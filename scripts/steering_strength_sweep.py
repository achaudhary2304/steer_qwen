"""Diagnose amplification collapse without retraining or altering checkpoints."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from concept_retrofit.io import save_json
from concept_retrofit.pipeline import load_trained
from concept_retrofit.evaluation.steering import generate_with_model
from concept_retrofit.evaluation.judge import judge_folder


def longest_word_run(text):
    words = text.casefold().split()
    best = current = 0
    previous = None
    for word in words:
        current = current + 1 if word == previous else 1
        best = max(best, current)
        previous = word
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--concepts', default='Music,Home cooking,Astronomy,Computer Science')
    parser.add_argument('--strengths', default='0.05,0.1,0.2,0.5,1')
    parser.add_argument('--max-new-tokens', type=int, default=48)
    parser.add_argument('--judge', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    strengths = tuple(float(s) for s in args.strengths.split(','))
    if len(strengths) != len(set(strengths)):
        raise ValueError('Duplicate strengths')
    cfg, corpus, model, tokenizer, module = load_trained(args.data, args.checkpoint)
    cfg = replace(cfg, steering_mode='layer')
    lookup = {c['name'].casefold():i for i,c in enumerate(corpus.manifest['concepts'])}
    selected = [{'concept_index':lookup[n.strip().casefold()],
                 'concept_id':corpus.ids[lookup[n.strip().casefold()]],
                 'metadata':corpus.manifest['concepts'][lookup[n.strip().casefold()]]}
                for n in args.concepts.split(',')]
    save_json(out/'status.json', {'state':'running','checkpoint':args.checkpoint,
        'concepts':[c['metadata']['name'] for c in selected], 'strengths':strengths,
        'base_tau':cfg.steering_tau, 'decoding':'greedy; unchanged across strengths'})
    rows, groups = [], []
    for strength in strengths:
        folder = out/f'strength-{strength:g}'
        folder.mkdir()
        save_json(folder/'interventions.json',{'concepts':selected})
        groups.append(folder)
    for concept in selected:
        reference = None
        for strength,folder in zip(strengths,groups):
            destination = folder/f'steering-concept-{concept["concept_index"]}.json'
            generate_with_model(cfg,corpus,model,tokenizer,module,destination,
                args.max_new_tokens,concept['concept_index'],
                ['Write a short story about someone spending an evening at home. Keep it under 70 words.'],
                strength,reference)
            examples = json.loads(destination.read_text())['examples']
            reference = {(e['prompt'],e['condition']):e for e in examples
                         if e['condition'] in ('base','retrofit')}
            rows.extend({'concept':concept['metadata']['name'],'strength':strength,
                         'tau':cfg.steering_tau*strength, 'condition':e['condition'],
                         'longest_word_run':longest_word_run(e['response']),
                         'response':e['response']} for e in examples)
            save_json(out/'examples.json',rows)
    del model,module,tokenizer,corpus
    gc.collect();torch.cuda.empty_cache()
    summaries = {}
    if args.judge:
        for strength,folder in zip(strengths,groups):
            summaries[str(strength)] = judge_folder(folder)
            save_json(out/'judge-summaries.json',summaries)
    status = json.loads((out/'status.json').read_text())
    status.update(state='complete', judged=bool(args.judge),
                  caveat='Small diagnostic on selected failures, not a held-out steering benchmark; repetition alone does not measure semantic success.')
    save_json(out/'status.json',status)
    print('STRENGTH_SWEEP_COMPLETE '+str(out),flush=True)


if __name__ == '__main__':
    main()
