"""Run matched pre/post intervention audits and a bounded steering pilot.

Usage: python scripts/steering_pilot.py --data ... --checkpoint ... --out ...
The source checkpoint is copied before any training; no running trainer changes.
"""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from concept_retrofit.io import save_json
from concept_retrofit.pipeline import load_run_config, load_trained, train_stage, evaluate
from concept_retrofit.training.steering import build_lexicon
from concept_retrofit.evaluation.content_steering import run as audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default='runs/steering-pilot')
    parser.add_argument('--config', default='configs/runnable/qwen35-08b-steering-pilot.json')
    parser.add_argument('--concepts', default='Music,Home cooking,Astronomy,Computer Science')
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--strengths', default='0.2,1,3')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lexicon-documents', type=int, default=20000)
    parser.add_argument('--expanded-prompts', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and not args.resume:
        raise FileExistsError('Use --resume or a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    source = out/'source.pt'
    if not source.exists():
        shutil.copyfile(args.checkpoint, source)
    cfg = load_run_config(args.config)
    cfg = replace(cfg, steering_lexicon=str(out/'lexicon.json'))
    cfg.validate()
    save_json(out/'pilot-config.json', asdict(cfg))
    names = [n.strip() for n in args.concepts.split(',')]
    strengths = tuple(float(s) for s in args.strengths.split(','))
    original_cfg, corpus, model, tokenizer, module = load_trained(args.data, source)
    if cfg.model != original_cfg.model or cfg.revision != original_cfg.revision:
        raise ValueError('Pilot config must use the source backbone')
    concept_map = {c['name'].casefold():i for i,c in enumerate(corpus.manifest['concepts'])}
    indices = [concept_map[n.casefold()] for n in names]
    if not Path(cfg.steering_lexicon).exists():
        build_lexicon(args.data, tokenizer, cfg.steering_lexicon, indices, max_documents=args.lexicon_documents)
    report, _, _ = evaluate(model, tokenizer, module, corpus,
                             replace(original_cfg, evaluation_documents=cfg.evaluation_documents), 'validation')
    save_json(out/'validation-before.json', report)
    del model, tokenizer, module, corpus
    gc.collect(); torch.cuda.empty_cache()
    # Matching layer interface before training isolates training from simply
    # adding the inference mechanisms. Keep the frozen source's stage/adapters.
    state = torch.load(source, map_location='cpu', weights_only=True)
    state['config'].update(steering_mode='layer', steering_tau=cfg.steering_tau,
        steering_inference_tau=cfg.steering_inference_tau,
        steering_start_layer=cfg.steering_start_layer, suppression_strength=cfg.suppression_strength)
    before = out/'before-layer.pt'
    if not before.exists():
        torch.save(state, before)
    del state
    for checkpoint, directory in [(before, out/'before')]:
        if not (directory/'audit.json').exists() or json.loads((directory/'audit.json').read_text())['state'] != 'complete':
            audit(args.data, checkpoint, directory, names, args.max_new_tokens, strengths, expanded_prompts=args.expanded_prompts)
    trained = out/'training'
    status = trained/'status.json'
    complete = status.exists() and json.loads(status.read_text())['state'] == 'complete'
    if not complete:
        train_stage(args.data, trained, cfg, 'steering', str(source),
                    resume=args.resume and (trained/'last.pt').exists())
    after = trained/'last.pt'
    cfg_after, corpus, model, tokenizer, module = load_trained(args.data, after)
    report, _, _ = evaluate(model, tokenizer, module, corpus, cfg_after, 'validation')
    save_json(out/'validation-after.json', report)
    del model, tokenizer, module, corpus
    gc.collect(); torch.cuda.empty_cache()
    if not (out/'after/audit.json').exists() or json.loads((out/'after/audit.json').read_text())['state'] != 'complete':
        audit(args.data, after, out/'after', names, args.max_new_tokens, strengths, expanded_prompts=args.expanded_prompts)
    save_json(out/'status.json', {'state':'complete', 'source_checkpoint':str(source),
                                 'annotation_policy':'train-only lift-based weak token positions',
                                 'config':asdict(cfg)})
    print(f'STEERING_PILOT_COMPLETE out={out}', flush=True)


if __name__ == '__main__':
    main()
