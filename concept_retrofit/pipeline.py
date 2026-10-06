"""Runnable staged Qwen experiment, with atomic checkpoints and validation gates."""
from dataclasses import asdict, dataclass, replace
import gc
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data.prepare import Corpus
from .io import digest, save_checkpoint, save_json
from .models import ConceptBottleneck
from .models.qwen import load_text
from .training.objectives import annotation_loss, chunk_logits, language_losses


@dataclass
class RunConfig:
    model: str = 'Qwen/Qwen3.5-0.8B'
    revision: str | None = '2fc06364715b967f1860aea9cf38778875588b17'
    device: str = 'cuda'
    seed: int = 1729
    max_length: int = 128
    batch_size: int = 1
    steps: int = 100
    training_tokens: int = 0
    evaluate_every: int = 25
    evaluation_documents: int = 128
    unknown_features: int = 192
    unknown_rank: int = 32
    known_topk: int = 8
    unknown_topk: int = 32
    top_layers: int = 6
    lora_rank: int = 8
    bottleneck_lr: float = 0.0003
    lora_lr: float = 0.00003
    residual_scale: float = 0.5
    ce_weight: float = 1.0
    kl_weight: float = 1.0
    concept_weight: float = 1.0
    reconstruction_weight: float = 0.1
    residual_weight: float = 0.01
    leakage_weight: float = 0.01
    positive_weight: float = 4.0
    maximum_validation_kl: float | None = 0.5
    maximum_nll_increase: float | None = 0.3
    minimum_validation_auc: float | None = None
    diagnostics_every_tokens: int = 0
    diagnostics_judge: bool = False
    cpu_threads: int = 4
    steering_mode: str = 'coefficient'
    steering_balanced: bool = False
    carry_optimizer: bool = False
    residual_warmup: bool = True
    steering_lexicon: str | None = None
    steering_tau: float = 0.02
    steering_start_layer: int = -1
    respond_weight: float = 1.0
    express_weight: float = 1.0
    suppression_strength: float = 1.0

    def validate(self):
        if self.steering_mode not in ('coefficient', 'layer'):
            raise ValueError('steering_mode must be coefficient or layer')
        if not math.isfinite(self.steering_tau) or self.steering_tau <= 0:
            raise ValueError('steering_tau must be finite and positive')
        if self.steering_start_layer < -1:
            raise ValueError('steering_start_layer must be -1 or a layer index')
        if self.steps < 1 or self.batch_size < 1 or self.max_length < 2:
            raise ValueError('steps/batch_size must be positive, max_length >= 2')
        if self.training_tokens < 0:
            raise ValueError('training_tokens must be nonnegative')
        if self.diagnostics_every_tokens < 0:
            raise ValueError('diagnostics_every_tokens must be nonnegative')
        if self.minimum_validation_auc is not None and not 0 <= self.minimum_validation_auc <= 1:
            raise ValueError('minimum_validation_auc must be in [0,1]')
        if self.evaluate_every < 1 or self.evaluation_documents < 1:
            raise ValueError('Evaluation limits must be positive')
        if not 0 <= self.residual_scale < 1:
            raise ValueError('Evaluation residual scale must be below 1 to avoid identity claims')
        if self.unknown_rank < 1 or self.lora_rank < 1 or self.top_layers < 1:
            raise ValueError('Invalid rank/layer counts')
        if self.bottleneck_lr <= 0 or self.lora_lr <= 0:
            raise ValueError('Learning rates must be positive')
        if any(getattr(self, name) < 0 for name in (
            'ce_weight', 'kl_weight', 'concept_weight', 'reconstruction_weight',
            'residual_weight', 'leakage_weight', 'respond_weight', 'express_weight', 'suppression_strength')):
            raise ValueError('Loss weights must be nonnegative')


def load_run_config(path):
    config = RunConfig(**json.loads(Path(path).read_text()))
    config.validate()
    return config


def resume_configs_match(saved, current):
    """Stop thresholds may change; training/data settings must remain identical."""
    gates = {'maximum_validation_kl', 'maximum_nll_increase', 'minimum_validation_auc',
             'diagnostics_every_tokens', 'diagnostics_judge'}
    saved = asdict(RunConfig(**saved))
    current = asdict(RunConfig(**current))
    return {k: v for k, v in saved.items() if k not in gates} == {
        k: v for k, v in current.items() if k not in gates}


def setup(cfg, corpus, stage):
    cfg.validate()
    torch.set_num_threads(cfg.cpu_threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model, tokenizer = load_text(cfg.model, cfg.revision, cfg.device)
    if stage in ('lora', 'steering'):
        model.add_lora(cfg.top_layers, cfg.lora_rank)
    count = len(corpus.ids)
    module = ConceptBottleneck(model.head.in_features, count, cfg.unknown_features,
        cfg.unknown_rank, min(cfg.known_topk, count), cfg.unknown_topk).to(cfg.device)
    return model, tokenizer, module


def auc_metrics(scores, labels):
    from sklearn.metrics import roc_auc_score, average_precision_score
    aucs, aps = [], []
    for c in range(labels.shape[1]):
        if len(np.unique(labels[:, c])) == 2:
            aucs.append(float(roc_auc_score(labels[:, c], scores[:, c])))
            aps.append(float(average_precision_score(labels[:, c], scores[:, c])))
    return {'macro_auc': float(np.mean(aucs)) if aucs else None,
            'macro_average_precision': float(np.mean(aps)) if aps else None,
            'eligible_concepts': len(aucs), 'total_concepts': labels.shape[1]}


@torch.no_grad()
def evaluate(model, tokenizer, module, corpus, cfg, split):
    module.eval()
    examples = corpus.splits[split][:cfg.evaluation_documents]
    total_tokens, sums, scores, labels_all = 0, {'nll': 0., 'base_nll': 0., 'kl': 0.}, [], []
    concept_total, concept_documents = 0., 0
    shares = torch.zeros(3)
    share_tokens = 0
    features = {'original': [], 'named': [], 'unknown': [], 'residual': []}
    for start in range(0, len(examples), cfg.batch_size):
        inputs, labels, label_valid = corpus.batch(tokenizer, examples[start:start+cfg.batch_size],
                                                   cfg.max_length, cfg.device)
        with model.teacher():
            teacher = model.hidden(inputs)
        hidden = model.hidden(inputs)
        reconstructed, parts = module(hidden.float(), cfg.residual_scale)
        ce, kl, base_ce, n = language_losses(model, teacher, reconstructed,
                                             inputs['input_ids'], inputs['attention_mask'])
        total_tokens += n
        for key, value in [('nll', ce), ('kl', kl), ('base_nll', base_ce)]:
            sums[key] += float(value) * n
        mask = inputs['attention_mask'].bool()
        if label_valid.any():
            n_valid = int(label_valid.sum())
            concept_total += float(annotation_loss(chunk_logits(parts.known_logits, mask), labels,
                                                   label_valid, cfg.positive_weight)) * n_valid
            concept_documents += n_valid
            scores.append(chunk_logits(parts.known_logits, mask)[label_valid].cpu().numpy())
            labels_all.append(labels[label_valid].cpu().numpy())
            for key, value in [('original', teacher.float()), ('named', parts.known_activations),
                               ('unknown', parts.unknown_hidden), ('residual', parts.residual)]:
                pooled = (value * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
                features[key].append(pooled[label_valid].cpu().numpy())
        # Exact additive magnitudes for student-chosen tokens, including the
        # actual scaled residual. No teacher-token or gate-one identity shortcut.
        valid = mask[:, :-1] & mask[:, 1:]
        hs = reconstructed[:, :-1][valid]
        components = [parts.known_hidden[:, :-1][valid], parts.unknown_hidden[:, :-1][valid],
                      cfg.residual_scale * parts.residual[:, :-1][valid]]
        for offset in range(0, len(hs), 16):
            predicted = model.logits(hs[offset:offset+16]).argmax(-1)
            w = model.head.weight[predicted].float()
            magnitudes = torch.stack([(x[offset:offset+16] * w).sum(-1).abs() for x in components], -1)
            shares += (magnitudes / magnitudes.sum(-1, keepdim=True).clamp_min(1e-12)).sum(0).cpu()
            share_tokens += len(predicted)
    result = {key: value / total_tokens for key, value in sums.items()}
    result.update({'tokens': total_tokens, 'documents': len(examples), 'split': split,
                   'annotation_loss': concept_total / max(1, concept_documents),
                   'concept_valid_documents': concept_documents,
                   'residual_scale': cfg.residual_scale,
                   'perplexity': math.exp(min(result['nll'], 700)),
                   'base_perplexity': math.exp(min(result['base_nll'], 700)),
                   'annotation_detection': auc_metrics(np.concatenate(scores), np.concatenate(labels_all))
                       if scores else {'macro_auc': None, 'eligible_concepts': 0},
                   'logit_magnitude_shares': dict(zip(('named', 'unknown', 'residual'),
                                                      (shares / max(1, share_tokens)).tolist())),
                   'caveats': ['Chunk labels are machine annotations, not token-level ground truth.',
                               'Magnitude shares can be inflated by cancellation.']})
    arrays = {k: np.concatenate(v) for k, v in features.items()} if scores else {}
    return result, arrays, np.concatenate(labels_all) if labels_all else None


def train_stage(data, output, cfg, stage='frozen', initialize=None, resume=False):
    corpus = Corpus(data)
    lexicon = None
    lexicon_hash = None
    if stage == 'steering':
        if not cfg.steering_lexicon or cfg.steering_mode != 'layer':
            raise ValueError('Steering stage requires a lexicon and steering_mode=layer')
        resource = json.loads(Path(cfg.steering_lexicon).read_text())
        if resource['manifest_sha256'] != digest(Path(data) / 'manifest.json'):
            raise ValueError('Steering lexicon belongs to another dataset')
        lexicon = resource['concepts']
        if not lexicon:
            raise ValueError('Empty steering lexicon')
        lexicon_hash = digest(cfg.steering_lexicon)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_hash = digest(Path(data) / 'manifest.json')
    if cfg.training_tokens and corpus.manifest.get('unique_training_target_tokens', 0) < cfg.training_tokens:
        raise ValueError('Dataset lacks the requested unique Atlas-token budget')
    if (output / 'last.pt').exists() and not resume:
        raise FileExistsError('Output contains a checkpoint; use --resume or a fresh directory')
    model, tokenizer, module = setup(cfg, corpus, stage)
    steering_pools = None
    if stage == 'steering' and cfg.steering_balanced:
        from .training.steering import balanced_pools
        steering_pools = balanced_pools(corpus, tokenizer, lexicon, cfg.max_length, resource['documents'])
        save_json(output / 'steering-pools.json', {'split': 'train',
            'sampling': 'Round-robin concepts; half the batch is eligible concept chunks, half random LM chunks',
            'documents_scanned': min(resource['documents'], len(corpus.splits['train'])),
            'unique_chunks_by_concept': {key: len(values) for key, values in steering_pools.items()}})
        print('STEERING_POOLS '+json.dumps({key:len(values) for key,values in steering_pools.items()}), flush=True)
    adapters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if initialize:
        previous = torch.load(initialize, map_location='cpu', weights_only=True)
        if previous['manifest_sha256'] != manifest_hash:
            raise ValueError('Initialization checkpoint uses a different concept vocabulary/dataset')
        if previous['config']['model'] != cfg.model or previous['config']['revision'] != cfg.revision:
            raise ValueError('Initialization checkpoint uses a different backbone')
        module.load_state_dict(previous['bottleneck'])
        for name, value in previous.get('adapters', {}).items():
            if name not in adapters:
                raise ValueError('Initialization adapters do not match the new stage')
            adapters[name].data.copy_(value.to(cfg.device))
    # A linear adversary tries to read labels from unknown/residual. It is
    # fitted on detached features, then the module/backbone tries to confuse it.
    adversary = nn.Linear(2 * model.head.in_features, len(corpus.ids)).to(cfg.device)
    adv_optimizer = torch.optim.AdamW(adversary.parameters(), lr=cfg.bottleneck_lr)
    groups = [{'params': module.parameters(), 'lr': cfg.bottleneck_lr}]
    if adapters:
        groups.append({'params': list(adapters.values()), 'lr': cfg.lora_lr})
    optimizer = torch.optim.AdamW(groups)
    if cfg.carry_optimizer and not resume:
        if not initialize:
            raise ValueError('Optimizer continuation requires an initialization checkpoint')
        optimizer.load_state_dict(previous['optimizer'])
        for group, lr in zip(optimizer.param_groups, [cfg.bottleneck_lr, cfg.lora_lr]):
            group['lr'] = lr
        adversary.load_state_dict(previous['adversary'])
        adv_optimizer.load_state_dict(previous['adv_optimizer'])
        for group in adv_optimizer.param_groups:
            group['lr'] = cfg.bottleneck_lr
    generator = torch.Generator().manual_seed(cfg.seed)
    first_step, tokens_seen, best = 0, 0, float('inf')
    steering_counts = {'positions': 0, 'steps_with_injection': 0, 'by_concept': {}}
    order, cursor = torch.randperm(len(corpus.splits['train']), generator=generator), 0
    if resume:
        state = torch.load(output / 'last.pt', map_location='cpu', weights_only=True)
        if not resume_configs_match(state['config'], asdict(cfg)) or state['manifest_sha256'] != manifest_hash or state['stage'] != stage:
            raise ValueError('Resume requires identical training configuration, data, and stage; only stop thresholds may change')
        if state.get('steering_lexicon_sha256') != lexicon_hash:
            raise ValueError('Steering lexicon changed since checkpoint')
        module.load_state_dict(state['bottleneck'])
        for name, value in state['adapters'].items():
            adapters[name].data.copy_(value.to(cfg.device))
        adversary.load_state_dict(state['adversary'])
        optimizer.load_state_dict(state['optimizer'])
        adv_optimizer.load_state_dict(state['adv_optimizer'])
        generator.set_state(state['sampler_rng'])
        torch.set_rng_state(state['rng'])
        if cfg.device.startswith('cuda'):
            torch.cuda.set_rng_state(state['cuda_rng'])
        first_step, tokens_seen, best = state['step'], state['tokens_seen'], state['best']
        steering_counts = state.get('steering_counts', steering_counts)
        if state.get('selection_policy') != 'target-residual-scale-v1':
            # Older runs selected checkpoints evaluated with a different scale
            # from the one used in warm-up. Do not reuse that selection score.
            best = float('inf')
        order, cursor = state['order'], state['cursor']
    save_json(output / 'config.json', asdict(cfg))
    save_json(output / 'adapter_targets.json', model.adapter_names)
    interval = max(1, math.ceil(cfg.steps / 100))
    history = list(json.loads((output / 'history.json').read_text())) if resume and (output / 'history.json').exists() else []
    for step in range(first_step, cfg.steps):
        if cfg.training_tokens and tokens_seen >= cfg.training_tokens:
            break
        module.train()
        if cursor >= len(order):
            order, cursor = torch.randperm(len(order), generator=generator), 0
        indices = order[cursor:cursor+cfg.batch_size].tolist()
        cursor += len(indices)
        preferred_concept = None
        if steering_pools:
            keys = sorted(steering_pools, key=int)
            key = keys[step % len(keys)]
            preferred_concept = int(key)
            pool = steering_pools[key]
            for slot in range(max(1, len(indices) // 2)):
                indices[slot] = pool[torch.randint(len(pool), (1,), generator=generator).item()]
        inputs, labels, label_valid = corpus.batch(tokenizer, [corpus.splits['train'][i] for i in indices],
                                                    cfg.max_length, cfg.device)
        with model.teacher():
            teacher = model.hidden(inputs)
        respond, express, injection_count, injected_concept = None, None, 0, None
        positions = None
        if stage == 'steering':
            from .training.steering import calibrated_direction, injection_positions, intervention_losses
            candidates = []
            entries = [(str(preferred_concept), lexicon[str(preferred_concept)])] if preferred_concept is not None else lexicon.items()
            for key, entry in entries:
                selected_positions = injection_positions(inputs, labels, label_valid, int(key), entry['tokens'])
                if selected_positions.any():
                    candidates.append((int(key), entry, selected_positions))
            if preferred_concept is not None:
                candidates = [item for item in candidates if item[0] == preferred_concept]
                if not candidates:
                    raise RuntimeError('Balanced sample lacks its promised steering positions')
            if candidates:
                selected = 0 if preferred_concept is not None else torch.randint(len(candidates), (1,), generator=generator).item()
                injected_concept, entry, positions = candidates[selected]
                direction, _, _ = calibrated_direction(model, module, injected_concept, cfg.steering_tau)
                start_layer = cfg.steering_start_layer if cfg.steering_start_layer >= 0 else len(model.backbone.layers)-cfg.top_layers
                with model.inject(direction, positions, start_layer):
                    hidden = model.hidden(inputs)
                injection_count = int(positions.sum())
                steering_counts['positions'] += injection_count
                steering_counts['steps_with_injection'] += 1
                key = str(injected_concept)
                steering_counts['by_concept'][key] = steering_counts['by_concept'].get(key, 0) + injection_count
            else:
                hidden = model.hidden(inputs)
        else:
            hidden = model.hidden(inputs)
        progress = min(1., tokens_seen / max(1, cfg.training_tokens * 0.5)) if cfg.training_tokens else min(1., (step+1) / max(1, cfg.steps * 0.5))
        if not cfg.residual_warmup:
            progress = 1.
        scale = cfg.residual_scale if stage == 'steering' or not cfg.residual_warmup else 1. - (1. - cfg.residual_scale) * progress
        reconstructed, parts = module(hidden.float(), scale)
        mask = inputs['attention_mask'].bool()
        ce, kl, _, n = language_losses(model, teacher, reconstructed, inputs['input_ids'], mask)
        concept = annotation_loss(chunk_logits(parts.known_logits, mask), labels, label_valid, cfg.positive_weight)
        valid_h = mask[..., None]
        recon = ((reconstructed - teacher.float()).square() * valid_h).sum() / (mask.sum() * hidden.shape[-1])
        residual = (parts.residual.square() * valid_h).sum() / (mask.sum() * hidden.shape[-1])
        leakage = hidden.sum() * 0
        if stage == 'lora' and cfg.leakage_weight and label_valid.any():
            features = torch.cat((parts.unknown_hidden, parts.residual), -1)
            features = (features * valid_h).sum(1) / mask.sum(1, keepdim=True)
            adv_optimizer.zero_grad(set_to_none=True)
            adv_loss = annotation_loss(adversary(features.detach()), labels, label_valid, cfg.positive_weight)
            adv_loss.backward()
            adv_optimizer.step()
            adversary.requires_grad_(False)
            leakage = annotation_loss(adversary(features), labels, label_valid, cfg.positive_weight)
        loss = (cfg.ce_weight * ce + cfg.kl_weight * kl + cfg.concept_weight * concept +
                cfg.reconstruction_weight * recon + cfg.residual_weight * progress * residual -
                (cfg.leakage_weight * leakage if stage == 'lora' else 0))
        if stage == 'steering':
            respond = hidden.sum() * 0
            express = hidden.sum() * 0
            if injection_count:
                respond, express = intervention_losses(model, parts, reconstructed, positions, injected_concept, entry['tokens'])
            # Match the dedicated phase's intent: language objective + respond
            # + express. Keep teacher KL as a retrofit preservation addition;
            # do not impose chunk/reconstruction/independence losses here.
            loss = cfg.ce_weight * ce + cfg.kl_weight * kl + cfg.respond_weight * respond + cfg.express_weight * express
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss; previous checkpoint retained')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(module.parameters()) + list(adapters.values()), 1.)
        optimizer.step()
        adversary.requires_grad_(True)
        tokens_seen += n
        if (step + 1) % interval == 0 or (cfg.training_tokens and tokens_seen % max(1, cfg.training_tokens // 100) < n):
            print(f'PROGRESS {stage} {step+1}/{cfg.steps} loss={loss.item():.5f} '
                  f'ce={ce.item():.5f} kl={kl.item():.5f} concept={concept.item():.5f} tokens={tokens_seen}', flush=True)
        if stage == 'steering' and ((step+1) % max(1, cfg.steps // 100) == 0):
            print(f'STEERING_TRAIN step={step+1}/{cfg.steps} concept={injected_concept} '
                  f'positions={injection_count} respond={float(respond):.6f} express={float(express):.6f} '
                  f'tau={cfg.steering_tau}', flush=True)
        complete = (step + 1 == cfg.steps or bool(cfg.training_tokens and tokens_seen >= cfg.training_tokens))
        milestone = bool(cfg.diagnostics_every_tokens and tokens_seen // cfg.diagnostics_every_tokens >
                         (tokens_seen - n) // cfg.diagnostics_every_tokens)
        if (step + 1) % cfg.evaluate_every == 0 or complete or milestone:
            # Gate the actual training path during warm-up, not a prematurely
            # compressed deployment path. Final reports still use cfg's target.
            report, _, _ = evaluate(model, tokenizer, module, corpus,
                                    replace(cfg, residual_scale=scale), 'validation')
            selection = report['nll'] + cfg.kl_weight * report['kl'] + cfg.concept_weight * report['annotation_loss']
            history.append({'step': step + 1, 'validation': report,
                            'steering_counts': json.loads(json.dumps(steering_counts)) if lexicon else None})
            # Comparing scores across residual scales favors near-identity
            # warm-up checkpoints. Only select best at the deployment scale.
            is_best = scale <= cfg.residual_scale + 1e-12 and selection < best
            if is_best:
                best = selection
            state = {'config': asdict(cfg), 'stage': stage, 'manifest_sha256': manifest_hash,
                'step': step+1, 'tokens_seen': tokens_seen, 'best': best,
                'steering_lexicon_sha256': lexicon_hash, 'steering_counts': steering_counts,
                'selection_policy': 'target-residual-scale-v1', 'training_residual_scale': scale,
                'bottleneck': {n: p.detach().cpu() for n, p in module.state_dict().items()},
                'adapters': {n: p.detach().cpu() for n, p in adapters.items()},
                'adversary': adversary.state_dict(), 'optimizer': optimizer.state_dict(),
                'adv_optimizer': adv_optimizer.state_dict(), 'sampler_rng': generator.get_state(),
                'order': order, 'cursor': cursor,
                'rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state() if cfg.device.startswith('cuda') else None}
            save_checkpoint(output / 'last.pt', state)
            if is_best:
                save_checkpoint(output / 'best.pt', state)
            save_json(output / 'history.json', history)
            nll_increase = report['nll'] - report['base_nll']
            failures = []
            if not math.isfinite(report['kl']):
                failures.append('nonfinite validation KL')
            elif cfg.maximum_validation_kl is not None and report['kl'] > cfg.maximum_validation_kl:
                failures.append(f"validation KL {report['kl']:.6f} > {cfg.maximum_validation_kl}")
            if not math.isfinite(nll_increase):
                failures.append('nonfinite validation NLL increase')
            elif cfg.maximum_nll_increase is not None and nll_increase > cfg.maximum_nll_increase:
                failures.append(f"validation NLL increase {nll_increase:.6f} > {cfg.maximum_nll_increase}")
            concept_auc = report['annotation_detection']['macro_auc']
            if cfg.minimum_validation_auc is not None:
                if concept_auc is None:
                    failures.append('validation concept AUC unavailable: no eligible concepts with both positive and negative examples')
                elif not math.isfinite(concept_auc) or concept_auc < cfg.minimum_validation_auc:
                    failures.append(f'validation concept AUC {concept_auc:.6f} < {cfg.minimum_validation_auc}')
            safe = not failures
            print(f"VALIDATION {stage} step={step+1} base_nll={report['base_nll']:.6f} "
                  f"nll={report['nll']:.6f} nll_increase={nll_increase:.6f} "
                  f"kl={report['kl']:.6f} concept_auc={concept_auc} residual_scale={scale:.6f} "
                  f"gate={'pass' if safe else 'fail'}", flush=True)
            save_json(output / 'status.json', {'stage': stage, 'step': step+1,
                'state': 'complete' if complete and safe else 'running' if safe else 'gate_failed',
                'gate_failures': failures,
                'training_tokens': tokens_seen, 'requested_training_tokens': cfg.training_tokens,
                'validation': report, 'steering_counts': steering_counts if lexicon else None})
            if not safe:
                raise RuntimeError('Validation gate failed: ' + '; '.join(failures) +
                                   f'; checkpoint saved in {output / "last.pt"}, pipeline stopped')
            if milestone:
                from .evaluation.diagnostics import run_diagnostics
                run_diagnostics(model, tokenizer, module, corpus, cfg, output,
                                step + 1, tokens_seen, scale, report)
    if stage == 'steering' and steering_counts['positions'] == 0:
        save_json(output / 'status.json', {'state': 'invalid_supervision', 'stage': stage,
                                          'steering_counts': steering_counts})
        raise RuntimeError('No token positions received steering supervision; inspect the lexicon and sampling')
    if cfg.training_tokens and tokens_seen < cfg.training_tokens:
        raise RuntimeError('Step limit reached before token budget; increase steps and restart with a new config')
    del model, module, adversary
    gc.collect()
    if cfg.device.startswith('cuda'):
        torch.cuda.empty_cache()
    return str(output / 'best.pt')


def load_trained(data, checkpoint):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = RunConfig(**state['config'])
    corpus = Corpus(data)
    if state['manifest_sha256'] != digest(Path(data) / 'manifest.json'):
        raise ValueError('Checkpoint/data manifest mismatch')
    model, tokenizer, module = setup(cfg, corpus, state['stage'])
    module.load_state_dict(state['bottleneck'])
    parameters = dict(model.named_parameters())
    for name, value in state['adapters'].items():
        parameters[name].data.copy_(value.to(cfg.device))
    return cfg, corpus, model, tokenizer, module


def evaluate_checkpoint(data, checkpoint, output):
    cfg, corpus, model, tokenizer, module = load_trained(data, checkpoint)
    # Validation is the probe-fitting split; the final report is held-out test.
    _, train_features, train_labels = evaluate(model, tokenizer, module, corpus, cfg, 'validation')
    report, test_features, test_labels = evaluate(model, tokenizer, module, corpus, cfg, 'test')
    report['leakage_probes'] = {}
    if train_labels is not None and test_labels is not None:
        from sklearn.linear_model import RidgeClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        for key in train_features:
            scores = np.zeros_like(test_labels)
            eligible = []
            for c in range(len(corpus.ids)):
                if len(np.unique(train_labels[:, c])) != 2 or len(np.unique(test_labels[:, c])) != 2:
                    continue
                probe = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10, solver='lsqr'))
                probe.fit(train_features[key], train_labels[:, c])
                scores[:, c] = probe.decision_function(test_features[key])
                eligible.append(c)
            report['leakage_probes'][key] = auc_metrics(scores[:, eligible], test_labels[:, eligible])
    report['probe_fit_split'] = 'validation; post-hoc diagnostic, not used in checkpoint selection'
    save_json(output, report)
    del model, module
    gc.collect()
    if cfg.device.startswith('cuda'):
        torch.cuda.empty_cache()
    return report


def probe_layers(data, cfg, output):
    corpus = Corpus(data)
    model, tokenizer = load_text(cfg.model, cfg.revision, cfg.device)
    count = len(model.backbone.layers)
    layers = sorted(set([0, count // 4, count // 2, 3 * count // 4, count]))
    matrices, labels_by_split = {}, {}
    with torch.no_grad():
        for split in ('train', 'validation'):
            values = {i: [] for i in layers}
            labels_all = []
            examples = corpus.splits[split][:cfg.evaluation_documents]
            for start in range(0, len(examples), cfg.batch_size):
                inputs, labels, valid = corpus.batch(tokenizer, examples[start:start+cfg.batch_size], cfg.max_length, cfg.device)
                hidden = model.hidden(inputs, all_layers=True)
                mask = inputs['attention_mask']
                for i in layers:
                    pooled = (hidden[i].float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
                    values[i].append(pooled[valid].cpu().numpy())
                labels_all.append(labels[valid].cpu().numpy())
                print(f'PROGRESS probe {split} {min(start+cfg.batch_size,len(examples))}/{len(examples)}', flush=True)
            matrices[split] = {i: np.concatenate(v) for i, v in values.items()}
            labels_by_split[split] = np.concatenate(labels_all)
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    results = {}
    for layer in layers:
        train_y, val_y = labels_by_split['train'], labels_by_split['validation']
        scores = np.zeros_like(val_y)
        eligible = [c for c in range(len(corpus.ids)) if len(np.unique(train_y[:, c])) == 2
                    and len(np.unique(val_y[:, c])) == 2]
        for c in eligible:
            clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10, solver='lsqr'))
            clf.fit(matrices['train'][layer], train_y[:, c])
            scores[:, c] = clf.decision_function(matrices['validation'][layer])
        results[str(layer)] = auc_metrics(scores[:, eligible], val_y[:, eligible])
    save_json(output, {'layers': results, 'selection_split': 'validation',
                      'insertion_policy': 'Final output before LM head, regardless of probe peak; probes diagnose depth only'})
    del model
    gc.collect()
    if cfg.device.startswith('cuda'):
        torch.cuda.empty_cache()
    return results
