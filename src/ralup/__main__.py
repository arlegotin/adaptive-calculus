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
    preparation = subcommands.add_parser("prepare-data", help="authenticate sources, audit duplicates and freeze non-scoring partitions")
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
    harness = subcommands.add_parser("test-harness", help="run implemented harness engineering checks")
    harness.add_argument("--suite", choices=("data", "model", "all"), default="all")
    harness.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="device for model checks")
    harness.add_argument("--output", type=Path, default=Path("artifacts/checks/data-results.json"))
    baselines = subcommands.add_parser("verify-baselines", help="run named baseline engineering fixtures")
    baselines.add_argument("--suite", choices=("adamw",), required=True)
    baselines.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    baselines.add_argument("--output", type=Path, default=Path("artifacts/checks/adamw-results.json"))
    run = subcommands.add_parser("run", help="bounded explore engine with explicit smoke or pilot-bootstrap purpose")
    run.add_argument("--phase", choices=("explore",), required=True)
    run.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "run":
        try:
            print(canonical_bytes(_run_explore(args.config)).decode())
            return 0
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    if args.command == "verify-baselines":
        from .baselines.checks import check_adamw
        try:
            ref = check_adamw(args.device, args.output)
            report = json.loads(ref.path.read_bytes())
            print(canonical_bytes({"path": str(ref.path), "sha256": ref.sha256,
                                   "status": report["status"], "blockers": report["blockers"]}).decode())
            return 0 if report["status"] == "pass" else 1
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    if args.command == "test-harness":
        from .data.checks import check_data
        try:
            if args.suite == "all":
                ref = _check_harness(args.device,args.output)
            elif args.suite == "model":
                from .models.checks import check_model
                ref = check_model(args.device, args.output)
            else:
                ref = check_data(args.output)
            report = json.loads(ref.path.read_bytes())
            print(canonical_bytes({"path": str(ref.path), "sha256": ref.sha256,
                                   "status": report["status"], "blockers": report["blockers"]}).decode())
            return 0 if report["status"] == "pass" else 1
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    if args.command == "prepare-data":
        from .data.partitions import prepare_bootstrap
        try:
            ref = prepare_bootstrap(args.config)
            report = json.loads(ref.path.read_bytes())
            print(canonical_bytes({"path": str(ref.path), "sha256": ref.sha256,
                                   "status": report["status"], "missing_stages": [stage for stage in report["missing_stages"] if stage != "H03 byte batches"]}).decode())
            return 1  # H01-H03 implemented; H04-H08/full WP02 remain pending.
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


def _check_harness(device, output):
    from .data.checks import check_data
    from .models.checks import check_model
    from .train import check_engine
    from .records import ArtifactRef, sha256_file
    output = Path(output)
    coverage = check_data(output.with_name(output.stem+"-coverage.json"))
    model = check_model(device,output.with_name(output.stem+"-model.json"))
    reports = {"coverage":json.loads(coverage.path.read_bytes()),"model":json.loads(model.path.read_bytes())}
    report = {"schema_version":1,"scope":"H06 synthetic harness engineering only",
              "scientific_gate":None,"device":device,"blockers":[],"checks":{}}
    for name,value in reports.items():
        report["checks"][name] = {"status":value["status"],"evidence":{"path":str((coverage if name == "coverage" else model).path),
                                                                      "sha256":(coverage if name == "coverage" else model).sha256}}
        report["blockers"].extend(value["blockers"])
    if not report["blockers"]:
        report["checks"].update(check_engine(device))
    report["status"] = "pass" if not report["blockers"] else "blocked"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(canonical_bytes(report)+b"\n")
    return ArtifactRef(output,sha256_file(output))


def _run_explore(path):
    """Explicit envelope; resolution errors never switch to bootstrap."""
    import hashlib
    import os
    import subprocess
    import uuid
    from datetime import datetime, timezone
    import torch
    from .data.articles import Article
    from .data.cursor import BatchCursor
    from .models.transformer import ByteTransformer, ModelConfig
    from .optim.base import MethodAdapter, SchedulePoint, default_parameter_groups
    from .optim.adamw import AdamWUpdateRule
    from .records import RunLedger, WorkLedger
    from .train import TrainingState, select_logical_batch, train_step
    import time
    project_start = time.perf_counter()
    envelope = json.loads(Path(path).read_bytes())
    if set(envelope)-{"config","evidence","purpose","pilot_target_budget","seed","output"}:
        raise ValueError("Unknown explore envelope field")
    purpose = envelope.get("purpose")
    template = envelope.get("config")
    if not isinstance(template,dict):
        raise ValueError("Explore envelope requires config object and explicit purpose")
    evidence = envelope.get("evidence",{})
    if "purpose" in template:
        template = dict(template)
        if template.pop("purpose") != purpose or evidence:
            raise ValueError("Resolved explore purpose/provenance mismatch")
        evidence = {"gates":template.pop("evidence_gates")}
    resolved = resolve_config(template,evidence,purpose=purpose)
    if purpose not in ("smoke","pilot-bootstrap"):
        raise ValueError("Final train/meta execution awaits later measured calibration; select pilot-bootstrap explicitly for bounded development")
    budget = envelope.get("pilot_target_budget")
    if type(budget) is not int or budget <= 0:
        raise ValueError("Explore requires an explicit finite positive pilot_target_budget byte envelope")
    seed = envelope.get("seed",606)
    if type(seed) is not int or seed < 0:
        raise ValueError("Explore seed must be a nonnegative integer")
    config = json.loads(canonical_bytes(resolved.payload))
    if config["synthetic"] and budget > config["training"]["continuation_steps"]*8192:
        raise ValueError("Synthetic envelope exceeds its bounded smoke steps")
    device = config["runtime"]["device"]
    if device == "mps" and (os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or not torch.backends.mps.is_available()):
        raise ValueError("Actual MPS requires availability and fallback disabled before interpreter startup")
    if config["synthetic"]:
        content = b"synthetic H06 byte fixture\n"*400
        cursor = BatchCursor.from_articles([Article(content,[],hashlib.sha256(content).hexdigest())],seed=seed,run_target_budget=budget)
    else:
        manifest = config["data"]["manifest"]
        if json.loads(Path(manifest["path"]).read_bytes())["revision"] != config["data"]["revision"]:
            raise ValueError("Bootstrap corpus revision differs from resolved identity")
        cursor = BatchCursor.from_manifest(Path(manifest["path"]),manifest["sha256"],partition="target-training",seed=seed,run_target_budget=budget)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = ByteTransformer(ModelConfig(**{key:config["model"][key] for key in ("layers","width","heads")})).to(device)
    model.configure_quantization({"name":config["training"]["recipe"],
                                  "rotation":"full_hadamard" if config["quantization"]["rotation"] == "hadamard" else "identity"})
    optimizer = config["optimizer_probe"]
    method = MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),
                           AdamWUpdateRule(betas=tuple(optimizer["betas"]),eps=optimizer["epsilon"]))
    run_id = "h06-"+uuid.uuid4().hex
    output = Path(envelope.get("output",f"artifacts/runs/{run_id}"))
    commit = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    diff = subprocess.check_output(["git","diff","HEAD"])
    artifact_hash = lambda name: config["runtime"][name]["sha256"] if config["runtime"][name] is not None else None
    disposition = "synthetic_engineering_only" if config["synthetic"] else "pilot_bootstrap_development_only"
    ledger = RunLedger(output,{"run_id":run_id,"created_utc":datetime.now(timezone.utc).isoformat(),
        "local_timezone":"Europe/Amsterdam","phase":"smoke" if config["synthetic"] else "explore",
        "source":{"commit":commit,"dirty_diff_sha256":hashlib.sha256(diff).hexdigest() if diff else None},
        "lock_sha256":artifact_hash("dependency_lock"),"environment_sha256":artifact_hash("environment"),
        "resolved_config":config,"resolved_config_sha256":resolved.sha256,
        "method":"adamw_original_W_byte_adapter","adaptation":"project_H06_bootstrap",
        "seeds":{"model":seed,"data":seed},"data_sha256":None if config["synthetic"] else config["data"]["manifest"]["sha256"],
        "warm_start_sha256":None,"policy_sha256":None,"lineage":{"parent":None,"resume":None},
        "endpoint":{"pilot_target_budget":budget,"disposition":disposition,"final_scientific_duration":False}})
    state = TrainingState(model,method,cursor=cursor,ledger=WorkLedger(ledger),microbatch=config["training"]["microbatch"],
                          schedule={"step":0,"point":SchedulePoint(optimizer["learning_rate"],optimizer["decay"],0.)})
    ledger.record_status("running",disposition=disposition)
    try:
        while cursor.remaining_run_targets:
            train_step(state,select_logical_batch(state))
    except Exception as error:
        ledger.record_status("failed",disposition=str(error))
        raise
    finally:
        project_elapsed = time.perf_counter()-project_start
        ledger.record({"step":state.schedule["step"],"stream":0,"unique":0,"additional_loss":0,"backward":0,"query":0,
                       "elapsed_seconds":0.,"status":"project_elapsed_observation","project_elapsed_seconds":project_elapsed})
    ledger.record_status("completed",disposition=disposition)
    return {"path":str(output),"synthetic":config["synthetic"],"disposition":disposition,
            "stream_targets":state.ledger.stream["targets"],"steps":state.schedule["step"],
            "checkpoint_published":False,"scientific_gate":None,"project_elapsed_seconds":project_elapsed}


if __name__ == "__main__":
    raise SystemExit(main())
