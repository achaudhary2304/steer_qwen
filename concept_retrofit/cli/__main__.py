"""Small safe entry point while the production training loop is built."""
from __future__ import annotations

import argparse

from concept_retrofit.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="concept-retrofit")
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate-config", help="validate an experiment JSON config")
    validate.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.command == "validate-config":
        config = load_config(args.config)
        print(f"valid: {config.name}")


if __name__ == "__main__":
    main()
