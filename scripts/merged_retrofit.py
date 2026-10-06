"""One continuous retrofit: interleave steering and capability phases.

Every phase inherits the preceding bottleneck, LoRA and optimizer states.
Known concept coverage is reported honestly: lexical proxy supervision may not
support every Atlas label. This is an AR adaptation, not a diffusion replica.
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
from concept_retrofit.pipeline import load_run_config, load_trained, train_stage, evaluate_checkpoint
from concept_retrofit.training.steering import build_lexicon
from concept_retrofit.evaluation.content_steering import run as audit


def phase_plan(steering_steps, capability_steps, final_capability_steps):
    if min(steering_steps,capability_steps,final_capability_steps) < 1:
        raise ValueError('Phase lengths must be positive')
    result = []
    for i in range(1,5):
        result += [(f'steering-{i:02d}', 'steering', steering_steps),
                   (f'capability-{i:02d}', 'lora', final_capability_steps if i == 4 else capability_steps)]
    return result


def evaluation_concepts(manifest, resource, count=32):
    """Deterministic content controls spread across available taxonomy groups."""
    eligible = {int(key) for key in resource['concepts']}
    candidates = {c['name']:i for i,c in enumerate(manifest['concepts'])
                  if i in eligible and c.get('concept_type') == 'content'}
    selected = [name for name in ('Music','Home cooking','Astronomy','Computer Science') if name in candidates]
    groups = {}
    for name,index in candidates.items():
        if name in selected:
            continue
        metadata = manifest['concepts'][index]
        group = str(metadata.get('taxonomy_lcc_path_primary') or metadata.get('taxonomy_lcc_primary') or 'other')
        groups.setdefault(group,[]).append(name)
    for values in groups.values():
        values.sort(key=lambda name:candidates[name])
    while len(selected) < count and any(groups.values()):
        for group in sorted(groups):
            if groups[group] and len(selected) < count:
                selected.append(groups[group].pop(0))
    if not selected:
        raise ValueError('No eligible content controls for semantic evaluation')
    return selected


def audit_once(data, checkpoint, folder, names):
    folder = Path(folder)
    attempts = [folder, *sorted(folder.parent.glob(folder.name+'-retry-*'))]
    for attempt in attempts:
        status = attempt/'audit.json'
        if status.exists() and json.loads(status.read_text())['state'] == 'complete':
            return str(attempt)
    destination = folder
    retry = 1
    while destination.exists():
        destination = folder.with_name(folder.name+f'-retry-{retry}')
        retry += 1
    audit(data,checkpoint,destination,names,64,(1.,),expanded_prompts=True)
    return str(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default='configs/runnable/qwen35-08b-steering-scaled.json')
    parser.add_argument('--steering-steps', type=int, default=5000)
    parser.add_argument('--capability-steps', type=int, default=2000)
    parser.add_argument('--final-capability-steps', type=int, default=5000)
    parser.add_argument('--lexicon-documents', type=int, default=100000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluation-concepts', type=int, default=32)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and not args.resume:
        raise FileExistsError('Use a fresh output or --resume')
    out.mkdir(parents=True, exist_ok=True)
    source = out/'source.pt'
    if not source.exists():
        shutil.copyfile(args.checkpoint, source)
    cfg = load_run_config(args.config)
    cfg = replace(cfg, steering_lexicon=str(out/'lexicon.json'), carry_optimizer=True,
                  residual_warmup=False, steering_mode='layer', steering_balanced=True,
                  training_tokens=0, diagnostics_every_tokens=0, diagnostics_judge=False)
    plan = phase_plan(args.steering_steps,args.capability_steps,args.final_capability_steps)
    specification = {'config':asdict(cfg),'phases':plan,'lexicon_documents':args.lexicon_documents,'evaluation_concepts':args.evaluation_concepts}
    spec_file = out/'plan.json'
    if spec_file.exists() and json.loads(spec_file.read_text()) != json.loads(json.dumps(specification)):
        raise ValueError('Merged resume requires the original plan')
    save_json(spec_file,specification)
    source_cfg, corpus, model, tokenizer, module = load_trained(args.data,source)
    known_slots = len(corpus.ids)
    if source_cfg.model != cfg.model or source_cfg.revision != cfg.revision:
        raise ValueError('Merged config must match the source backbone')
    state = torch.load(source,map_location='cpu',weights_only=True)
    if not state['adapters']:
        raise ValueError('Combined schedule must initialize from the main LoRA checkpoint')
    del state
    if not Path(cfg.steering_lexicon).exists():
        build_lexicon(args.data, tokenizer, cfg.steering_lexicon, range(len(corpus.ids)),
                      max_documents=args.lexicon_documents)
    resource = json.loads(Path(cfg.steering_lexicon).read_text())
    save_json(out/'coverage.json', {'known_slots':len(corpus.ids),
        'steering_eligible_concepts':len(resource['concepts']),
        'not_eligible_indices':[i for i in range(len(corpus.ids)) if str(i) not in resource['concepts']],
        'annotation_policy':resource['annotation_policy']})
    names = evaluation_concepts(corpus.manifest,resource,args.evaluation_concepts)
    monitor_names = names[:8]
    save_json(out/'evaluation-concepts.json',{'final':names,'phase_monitor':monitor_names})
    # Compare the SAME layer interface before/after on the main model.
    state = torch.load(source,map_location='cpu',weights_only=True)
    state['config'].update(steering_mode='layer',steering_tau=cfg.steering_tau,
        steering_inference_tau=cfg.steering_inference_tau,
        steering_start_layer=cfg.steering_start_layer,suppression_strength=cfg.suppression_strength)
    baseline = out/'before-layer.pt'
    if not baseline.exists(): torch.save(state,baseline)
    del state, model, tokenizer, module, corpus
    gc.collect();torch.cuda.empty_cache()
    audits = {'before':audit_once(args.data,baseline,out/'before',names)}
    save_json(out/'audits.json',audits)
    current = source
    for number,(name,stage,steps) in enumerate(plan,1):
        folder = out/name
        complete = (folder/'status.json').exists() and json.loads((folder/'status.json').read_text())['state'] == 'complete'
        save_json(out/'status.json', {'state':'running','phase':name,'phase_number':number,
            'total_phases':len(plan),'source_checkpoint':str(current),
            'steering_eligible_concepts':len(resource['concepts'])})
        if not complete:
            phase_cfg = replace(cfg,steps=steps,evaluate_every=min(500,steps),seed=cfg.seed+number)
            print(f'MERGED_PHASE {number}/{len(plan)} name={name} steps={steps} initialize={current}',flush=True)
            train_stage(args.data,folder,phase_cfg,stage,str(current),
                        resume=args.resume and (folder/'last.pt').exists())
        current = folder/'last.pt'
        # Matched monitoring detects steering gains or forgetting during
        # capability phases. Final test split is evaluated only at the end.
        audits[name] = audit_once(args.data,current,out/f'audit-{name}',monitor_names)
        save_json(out/'audits.json',audits)
    shutil.copyfile(current,out/'final.pt')
    audits['final'] = audit_once(args.data,out/'final.pt',out/'final-steering',names)
    save_json(out/'audits.json',audits)
    evaluate_checkpoint(args.data,out/'final.pt',out/'final-test.json')
    save_json(out/'status.json', {'state':'complete','final_checkpoint':str(out/'final.pt'),
        'phases':len(plan),'known_slots':known_slots,
        'steering_eligible_concepts':len(resource['concepts']),
        'copied_components':['known/unknown/residual output decomposition','sparse heads',
            'concept supervision','intervention respond/express objectives','calibrated layer injections',
            'gated suppression','interleaved steering and capability phases'],
        'remaining_differences':['autoregressive Qwen rather than diffusion','weak lexical token annotations',
            'mean chunk supervision rather than OR aggregation','scaled residual rather than paper residual dropout',
            'adversarial leakage rather than paper independence objective',
            'no claim of original Qwen training-data attribution']})
    print('MERGED_RETROFIT_COMPLETE '+str(out/'final.pt'),flush=True)


if __name__ == '__main__':
    main()
