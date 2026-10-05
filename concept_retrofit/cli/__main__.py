"""Small safe entry point while the production training loop is built."""
from __future__ import annotations

import argparse

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
    args = parser.parse_args()
    if args.command == "validate-config":
        config = load_config(args.config)
        print(f"valid: {config.name}")
    elif args.command == "synthetic-smoke":
        report = run(steps=args.steps, device=args.device)
        save_report(report, args.out)
        print(f"synthetic smoke: {'PASS' if report.passed else 'FAIL'}")
        print(f"report: {args.out}")
        if not report.passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
