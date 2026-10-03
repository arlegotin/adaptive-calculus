"""Read-only configuration CLI. Numerical and experiment commands are added later."""

import argparse
import json
from pathlib import Path

from . import __version__
from .config import PURPOSES, canonical_bytes, resolve_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ralup", description="RALUP research configuration and records")
    parser.add_argument("--version", action="version", version=f"ralup {__version__}")
    subcommands = parser.add_subparsers(dest="command")
    config = subcommands.add_parser("config", help="validate and resolve a configuration to stdout")
    config.add_argument("template", type=Path)
    config.add_argument("--evidence", type=Path, help="JSON measured inputs and hashed gate records")
    config.add_argument("--purpose", required=True, choices=PURPOSES)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        template = json.loads(args.template.read_text(encoding="utf-8"))
        evidence = json.loads(args.evidence.read_text(encoding="utf-8")) if args.evidence else {}
        resolved = resolve_config(template, evidence, purpose=args.purpose)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(canonical_bytes(resolved.payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
