"""Configuration and read-only environment CLI."""

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
    doctor = subcommands.add_parser("doctor", help="record read-only hardware and runtime discovery")
    doctor.add_argument("--output", type=Path, required=True)
    doctor.add_argument("--probe", choices=("cpu", "mps"), help="separate isolated FP32 operation gate")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "doctor":
        from .environment import probe_operations, write_environment_report
        try:
            ref = write_environment_report(args.output)
            result = {"path": str(ref.path), "sha256": ref.sha256}
            if args.probe:
                gate = probe_operations(args.probe)
                result["probe"] = {"status": gate.status, "blockers": list(gate.blockers),
                                   "evidence": [{"path": str(item.path), "sha256": item.sha256} for item in gate.evidence]}
            print(canonical_bytes(result).decode())
            return 0 if not args.probe or gate.status == "pass" else 1
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
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
