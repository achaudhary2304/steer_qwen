"""Periodic validation-only audits using resident training models."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from concept_retrofit.io import save_json
from concept_retrofit.pipeline import evaluate, auc_metrics
from .steering import generate_with_model


def leakage_report(train_features, train_labels, validation_features, validation_labels):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    result = {}
    if train_labels is None or validation_labels is None:
        return result
    eligible = [c for c in range(train_labels.shape[1])
                if len(np.unique(train_labels[:, c])) == 2
                and len(np.unique(validation_labels[:, c])) == 2]
    for key in train_features:
        scores = np.zeros((len(validation_labels), len(eligible)))
        for i, c in enumerate(eligible):
            probe = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10, solver='lsqr'))
            probe.fit(train_features[key], train_labels[:, c])
            scores[:, i] = probe.decision_function(validation_features[key])
        result[key] = auc_metrics(scores, validation_labels[:, eligible])
    return result


@torch.no_grad()
def intervention_effects(model, tokenizer, module, corpus, cfg, concept_index):
    """Last next-token target per chunk; not semantic generation judging."""
    concept_id = corpus.ids[concept_index]
    result = {}
    for group, annotated in [('annotated', True), ('unassigned', False)]:
        values = [r for r in corpus.splits['validation'][:cfg.evaluation_documents]
                  if (concept_id in r['label_ids']) == annotated][:16]
        effects = {condition: [] for condition in ('amplify', 'suppress')}
        for row in values:
            inputs, _, valid = corpus.batch(tokenizer, [row], cfg.max_length, cfg.device)
            if not valid.item():
                continue
            length = int(inputs['attention_mask'].sum())
            if length < 2:
                continue
            h = model.hidden(inputs)[:, length - 2].float()
            target = inputs['input_ids'][:, length - 1]
            factual, _ = module(h, cfg.residual_scale)
            factual_logits = model.logits(factual)
            factual_logp = factual_logits.log_softmax(-1)
            factual_ce = F.cross_entropy(factual_logits, target)
            for condition, activation in [('amplify', 1.), ('suppress', 0.)]:
                changed, _ = module(h, cfg.residual_scale, {concept_index: activation})
                logits = model.logits(changed)
                effects[condition].append({
                    'next_token_nll_change': float(F.cross_entropy(logits, target) - factual_ce),
                    'kl_from_factual': float(F.kl_div(logits.log_softmax(-1), factual_logp,
                                                     log_target=True, reduction='sum')),
                    'mean_absolute_logit_change': float((logits - factual_logits).abs().mean()),
                    'argmax_changed': float(logits.argmax(-1).item() != factual_logits.argmax(-1).item())})
        result[group] = {condition: {'chunks': len(items), **{
            key: float(np.mean([item[key] for item in items]))
            for key in items[0]}} if items else {'chunks': 0}
            for condition, items in effects.items()}
    return result


@torch.no_grad()
def run_diagnostics(model, tokenizer, module, corpus, cfg, output, step, tokens, scale, report):
    milestone = tokens // cfg.diagnostics_every_tokens * cfg.diagnostics_every_tokens
    folder = Path(output) / 'diagnostics' / f'tokens-{milestone:09d}'
    print(f'DIAGNOSTICS_START step={step} tokens={tokens} out={folder}', flush=True)
    previous_training = module.training
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if cfg.device.startswith('cuda') else None
    try:
        module.eval()
        current_cfg = replace(cfg, residual_scale=scale)
        save_json(folder / 'validation-current.json', report)
        if abs(scale - cfg.residual_scale) > 1e-12:
            target_report, _, _ = evaluate(model, tokenizer, module, corpus, cfg, 'validation')
        else:
            target_report = report
        save_json(folder / 'validation-target.json', target_report)
        print(f"DIAGNOSTICS_METRICS tokens={tokens} current_scale={scale:.6f} "
              f"auc={report['annotation_detection']['macro_auc']} "
              f"ap={report['annotation_detection'].get('macro_average_precision')} "
              f"named_share={report['logit_magnitude_shares']['named']:.6f} "
              f"target_nll={target_report['nll']:.6f} target_kl={target_report['kl']:.6f}", flush=True)

        # Fit diagnostic readers on train, evaluate on validation; never touch test.
        small_cfg = replace(current_cfg, evaluation_documents=min(256, cfg.evaluation_documents))
        _, train_features, train_labels = evaluate(model, tokenizer, module, corpus, small_cfg, 'train')
        _, validation_features, validation_labels = evaluate(model, tokenizer, module, corpus, small_cfg, 'validation')
        leakage = leakage_report(train_features, train_labels, validation_features, validation_labels)
        save_json(folder / 'leakage.json', {'probe_fit_split': 'train', 'probe_score_split': 'validation',
            'document_limit_per_split': small_cfg.evaluation_documents, 'leakage_probes': leakage})

        counts = [sum(c in row['label_ids'] for row in corpus.splits['validation'][:cfg.evaluation_documents])
                  for c in corpus.ids]
        total = min(cfg.evaluation_documents, len(corpus.splits['validation']))
        selected = sorted((i for i, n in enumerate(counts) if 0 < n < total),
                          key=lambda i: (-counts[i], corpus.ids[i]))[:3]
        effects = []
        for index in selected:
            metadata = corpus.manifest['concepts'][index]
            print(f'DIAGNOSTICS_CONCEPT index={index} atlas_id={corpus.ids[index]}', flush=True)
            effects.append({'concept_index': index, 'concept_id': corpus.ids[index],
                'metadata': metadata, 'effects': intervention_effects(
                    model, tokenizer, module, corpus, current_cfg, index)})
            generate_with_model(current_cfg, corpus, model, tokenizer, module,
                                folder / f'steering-concept-{index}.json', concept_index=index,
                                prompts=['Write a short story about a quiet afternoon.',
                                    'Write a short passage that clearly illustrates this topic or style: '
                                    + str(metadata.get('name', '')) + '. '
                                    + str(metadata.get('description', ''))])
        save_json(folder / 'interventions.json', {'residual_scale': scale,
            'measurement': 'One last next-token target per chunk, up to 16 chunks per group.',
            'caveats': ['Unassigned labels are not verified negatives.',
                        'Prediction changes do not establish semantic steering success.',
                        'Concepts selected by validation support; not a comprehensive concept benchmark.'],
            'concepts': effects})
        if cfg.diagnostics_judge:
            from .judge import judge_folder, resolve_key
            try:
                judge_folder(folder)
            except Exception as error:
                # Judge/network failures are diagnostic failures, not reasons
                # to discard the saved checkpoint or halt GPU training.
                key = resolve_key()
                message = str(error).replace(key, '[REDACTED]') if key else str(error)
                save_json(folder / 'judge-status.json', {'state': 'error', 'error': message})
                print(f'JUDGE_ERROR {message}', flush=True)
        save_json(folder / 'status.json', {'state': 'complete', 'step': step,
            'training_tokens': tokens, 'milestone_tokens': milestone, 'split': 'validation',
            'current_residual_scale': scale, 'target_residual_scale': cfg.residual_scale})
        print(f'DIAGNOSTICS_COMPLETE step={step} tokens={tokens} out={folder}', flush=True)
    finally:
        module.train(previous_training)
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
