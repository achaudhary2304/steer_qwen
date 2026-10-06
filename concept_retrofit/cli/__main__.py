"""Public commands for dataset preparation, training, and evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

from concept_retrofit.config import load_config
from concept_retrofit.training.synthetic_smoke import run, save_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="concept-retrofit")
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate-config", help="validate an experiment JSON config")
    validate.add_argument("--config", required=True)
    smoke = subcommands.add_parser("synthetic-smoke", help="run the end-to-end synthetic bottleneck smoke test")
    smoke.add_argument("--steps", type=int, default=500)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--out", default="runs/smoke/synthetic-report.json")
    prepare = subcommands.add_parser('prepare-atlas', help='build a reproducible bounded Atlas subset')
    prepare.add_argument('--out', required=True)
    prepare.add_argument('--source', help='optional local JSONL export for offline use')
    prepare.add_argument('--known', type=int, default=64)
    prepare.add_argument('--sample-rows', type=int, default=20000)
    prepare.add_argument('--scan-rows', type=int, default=100000)
    prepare.add_argument('--min-support', type=int, default=10)
    prepare.add_argument('--shards', default='0,17,35,53')
    prepare.add_argument('--training-token-budget', type=int, default=0)
    prepare.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    prepare.add_argument('--revision', default='2fc06364715b967f1860aea9cf38778875588b17')
    prepare.add_argument('--max-length', type=int, default=256)
    judge = subcommands.add_parser('judge', help='score saved periodic steering examples through Groq')
    judge.add_argument('--diagnostics', required=True, help='one tokens-XXXXXXXXX directory')
    content = subcommands.add_parser('content-steer', help='standalone content steering audit and Groq judging')
    content.add_argument('--data', required=True)
    content.add_argument('--checkpoint', required=True)
    content.add_argument('--out', required=True)
    content.add_argument('--concepts', default='Music,Home cooking,Astronomy,Computer Science')
    content.add_argument('--max-new-tokens', type=int, default=96)
    for name in ('probe', 'train', 'evaluate', 'steer', 'run-all'):
        command = subcommands.add_parser(name)
        command.add_argument('--data', required=True)
        command.add_argument('--out', required=True)
        if name in ('probe', 'train', 'run-all'):
            command.add_argument('--config', required=True, help='runnable RunConfig JSON')
        if name == 'train':
            command.add_argument('--stage', choices=('frozen', 'lora'), default='frozen')
            command.add_argument('--initialize')
        if name in ('train', 'run-all'):
            command.add_argument('--resume', action='store_true')
        if name in ('evaluate', 'steer'):
            command.add_argument('--checkpoint', required=True)
        if name == 'steer':
            command.add_argument('--concept-index', type=int, default=0)
            command.add_argument('--max-new-tokens', type=int, default=32)
    args = parser.parse_args()
    if args.command == "validate-config":
        import json
        if 'name' in json.loads(Path(args.config).read_text()):
            config = load_config(args.config)
            print(f"valid: {config.name}")
        else:
            from concept_retrofit.pipeline import load_run_config
            config = load_run_config(args.config)
            print(f'valid runnable config: {config.model}')
    elif args.command == "synthetic-smoke":
        report = run(steps=args.steps, device=args.device)
        save_report(report, args.out)
        print(f"synthetic smoke: {'PASS' if report.passed else 'FAIL'}")
        print(f"report: {args.out}")
        if not report.passed:
            raise SystemExit(1)
    elif args.command == 'content-steer':
        from concept_retrofit.evaluation.content_steering import run
        run(args.data, args.checkpoint, args.out, [n.strip() for n in args.concepts.split(',')], args.max_new_tokens)
    elif args.command == 'judge':
        from concept_retrofit.evaluation.judge import judge_folder
        judge_folder(args.diagnostics)
    elif args.command == 'prepare-atlas':
        from transformers import AutoTokenizer
        from concept_retrofit.data.prepare import prepare
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        prepare(args.out, args.sample_rows, args.known, args.min_support, source=args.source,
                scan_limit=args.scan_rows, shards=tuple(map(int, args.shards.split(','))),
                tokenizer=tokenizer, max_length=args.max_length,
                training_token_budget=args.training_token_budget)
    else:
        from concept_retrofit.pipeline import load_run_config, train_stage, evaluate_checkpoint, probe_layers, resume_configs_match
        from concept_retrofit.evaluation.steering import generate_examples
        if args.command == 'probe':
            probe_layers(args.data, load_run_config(args.config), args.out)
        elif args.command == 'train':
            train_stage(args.data, args.out, load_run_config(args.config), args.stage, args.initialize, args.resume)
        elif args.command == 'evaluate':
            evaluate_checkpoint(args.data, args.checkpoint, args.out)
        elif args.command == 'steer':
            generate_examples(args.data, args.checkpoint, args.out, args.max_new_tokens, args.concept_index)
        elif args.command == 'run-all':
            cfg = load_run_config(args.config)
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            import json
            from dataclasses import asdict
            from concept_retrofit.io import digest, save_json
            specification = {'config': asdict(cfg), 'manifest_sha256': digest(Path(args.data) / 'manifest.json')}
            if (out / 'run.json').exists():
                previous = json.loads((out / 'run.json').read_text())
                if previous['manifest_sha256'] != specification['manifest_sha256'] or not resume_configs_match(previous['config'], specification['config']):
                    raise ValueError('Output belongs to a different configuration or dataset')
                if not args.resume:
                    raise FileExistsError('Pipeline already exists; use --resume')
                save_json(out / 'run.json', specification)
            else:
                save_json(out / 'run.json', specification)
            if not args.resume or not (out / 'layer-probes.json').exists():
                probe_layers(args.data, cfg, out / 'layer-probes.json')
            for stage in ('frozen', 'lora'):
                stage_out = out / stage
                import json
                complete = (stage_out / 'status.json').exists() and json.loads(
                    (stage_out / 'status.json').read_text())['state'] == 'complete'
                if not (args.resume and complete):
                    train_stage(args.data, stage_out, cfg, stage,
                        str(out / 'frozen' / 'best.pt') if stage == 'lora' else None,
                        args.resume and (stage_out / 'last.pt').exists())
                evaluate_checkpoint(args.data, stage_out / 'best.pt', out / f'{stage}-test.json')
                generate_examples(args.data, stage_out / 'best.pt', out / f'{stage}-examples.json')
            print(f'PIPELINE_COMPLETE {out}', flush=True)


if __name__ == "__main__":
    main()
