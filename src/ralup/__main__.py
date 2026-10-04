"""Configuration, environment discovery and numerical diagnostics CLI."""

import argparse
import json
from pathlib import Path

from . import __version__
from .config import PURPOSES, canonical_bytes, resolve_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ralup", description="RALUP research configuration and records")
    parser.add_argument("--version", action="version", version=f"ralup {__version__}")
    subcommands = parser.add_subparsers(dest="command")
    preparation = subcommands.add_parser("prepare-data", help="authenticate sources and reconstruct non-scoring provenance")
    preparation.add_argument("--config", type=Path, required=True)
    config = subcommands.add_parser("config", help="validate and resolve a configuration to stdout")
    config.add_argument("template", type=Path)
    config.add_argument("--evidence", type=Path, help="JSON measured inputs and hashed gate records")
    config.add_argument("--purpose", required=True, choices=PURPOSES)
    doctor = subcommands.add_parser("doctor", help="record read-only hardware and runtime discovery")
    doctor.add_argument("--output", type=Path, required=True)
    doctor.add_argument("--probe", choices=("cpu", "mps"), help="separate isolated FP32 operation gate")
    numerics = subcommands.add_parser("check-numerics", help="run implemented numerical oracle checks")
    numerics.add_argument("--suite", choices=("numerics", "quantizer"), default="numerics")
    numerics.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    numerics.add_argument("--output", type=Path, default=Path("artifacts/checks/results.json"))
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "prepare-data":
        from .data.source import prepare_data
        try:
            ref = prepare_data(args.config)
            report = json.loads(ref.path.read_bytes())
            print(canonical_bytes({"path": str(ref.path), "sha256": ref.sha256,
                                   "status": report["status"], "missing_stages": report["missing_stages"]}).decode())
            return 1  # H01 succeeds; the full preparation pipeline remains incomplete.
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    if args.command == "check-numerics":
        from .quantization.checks import check_numerics, check_quantizer
        try:
            check = check_quantizer if args.suite == "quantizer" else check_numerics
            ref = check(args.device, args.output)
            report = json.loads(ref.path.read_text(encoding="utf-8"))
            print(canonical_bytes({"path": str(ref.path), "sha256": ref.sha256,
                                   "status": report["status"], "blockers": report["blockers"]}).decode())
            return 0 if report["status"] == "pass" else 1
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
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
