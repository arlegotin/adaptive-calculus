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
    baselines.add_argument("--suite", choices=("adamw", "muon", "soap-madam", "ewgs", "trs", "routing-rounding"), required=True)
    baselines.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    baselines.add_argument("--output", type=Path, default=Path("artifacts/checks/adamw-results.json"))
    run = subcommands.add_parser("run", help="bounded explore engine with explicit smoke or pilot-bootstrap purpose")
    run.add_argument("--phase", choices=("explore",), required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument('--suite', choices=('warmstart',), help='separate bounded partition warmstarts')
    calibration = subcommands.add_parser('calibrate', help='bounded measured engineering capacity; no scientific duration selection')
    calibration.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "run":
        try:
            print(canonical_bytes(_run_warmstarts(args.config) if args.suite == 'warmstart' else _run_explore(args.config)).decode())
            return 0
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    if args.command == 'calibrate':
        from .calibrate import calibrate
        try:
            resolved, envelope = _calibration_config(args.config)
            ref = calibrate(resolved, envelope.get('microbatches', [1, 2, 4]))
            print(canonical_bytes({'path': str(ref.path), 'sha256': ref.sha256}).decode())
            return 0
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            parser.error(str(error))
    if args.command == "verify-baselines":
        from .baselines.checks import check_adamw
        try:
            if args.suite == "routing-rounding":
                from .baselines.rounding import check_routing_rounding
                ref = check_routing_rounding(args.device,args.output)
            elif args.suite == "trs":
                from .baselines.trs import check_trs
                ref = check_trs(args.device,args.output)
            elif args.suite == "ewgs":
                from .baselines.ewgs import check_ewgs
                ref = check_ewgs(args.device,args.output)
            elif args.suite == "soap-madam":
                from .baselines.soap_madam_checks import check_soap_madam
                ref = check_soap_madam(args.device,args.output)
            elif args.suite == "muon":
                from .baselines.muon_checks import check_muon
                ref = check_muon(args.device,args.output)
            else:
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


def _calibration_config(path):
    envelope = json.loads(Path(path).read_bytes())
    if set(envelope) - {'config', 'evidence', 'purpose', 'microbatches'}:
        raise ValueError('Unknown calibration envelope field')
    if envelope.get('purpose') != 'calibration':
        raise ValueError('Explicit calibration purpose required')
    template = dict(envelope['config'])
    evidence = envelope.get('evidence', {})
    if 'purpose' in template:
        if template.pop('purpose') != 'calibration' or evidence:
            raise ValueError('Resolved calibration purpose/provenance mismatch')
        evidence = {'gates': template.pop('evidence_gates')}
    return resolve_config(template, evidence, purpose='calibration'), envelope


def _run_warmstarts(path):
    from .calibrate import train_warmstart
    resolved, _ = _calibration_config(path)
    seed = resolved.payload['calibration']['seed']
    refs = {partition: train_warmstart(resolved, partition, seed) for partition in ('meta_train', 'target_train')}
    return {key: {'path': str(ref.path), 'sha256': ref.sha256} for key, ref in refs.items()}


def _run_explore(path):
    """Explicit envelope; resolution errors never switch to bootstrap."""
    import hashlib
    import os
    import subprocess
    import uuid
    import random
    from datetime import datetime, timezone
    import torch
    from .data.articles import Article
    from .data.cursor import BatchCursor
    from .models.transformer import ByteTransformer, ModelConfig
    from .optim.base import MethodAdapter, SchedulePoint, default_parameter_groups
    from .optim.adamw import AdamWUpdateRule
    from .records import ArtifactRef, RunLedger, WorkLedger
    from .state import capture_snapshot, restore_snapshot, save_snapshot, load_snapshot
    from .train import TrainingState, select_logical_batch, train_step
    import time
    project_start = time.perf_counter()
    envelope = json.loads(Path(path).read_bytes())
    if set(envelope)-{"config","evidence","purpose","pilot_target_budget","seed","output","checkpoint_directory","resume","stop_after_steps","method"}:
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
    checkpoint_directory = envelope.get("checkpoint_directory")
    stop_after = envelope.get("stop_after_steps")
    if stop_after is not None and (type(stop_after) is not int or stop_after <= 0 or checkpoint_directory is None):
        raise ValueError("stop_after_steps requires a positive count and explicit checkpoint_directory")
    method_settings = envelope.get('method')
    if method_settings is not None:
        if not isinstance(method_settings,dict) or set(method_settings) != {'name','adaptation_label','config','learning_rate'}:
            raise ValueError('Matrix method requires explicit name, adaptation_label, config and learning_rate')
        import math
        if type(method_settings['learning_rate']) not in (int,float) or not math.isfinite(method_settings['learning_rate']) or method_settings['learning_rate'] < 0:
            raise ValueError('Matrix learning rate must be finite and nonnegative')
    scientific_configuration = {"resolved":config,"seed":seed,"pilot_target_budget":budget}
    if method_settings is not None: scientific_configuration['method'] = method_settings
    resumed_state = None
    if envelope.get("resume") is not None:
        resumed_state = restore_snapshot(load_snapshot(ArtifactRef(**envelope["resume"])),config["runtime"]["device"])
        if resumed_state.configuration != scientific_configuration:
            raise ValueError("Resume config/seed/data budget differs from complete snapshot")
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
    initialization = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng():
        torch.set_rng_state(initialization.get_state())
        model = ByteTransformer(ModelConfig(**{key:config["model"][key] for key in ("layers","width","heads")})).to(device)
        initialization.set_state(torch.get_rng_state())
    stream_seed = lambda name: int.from_bytes(hashlib.sha256(f"ralup-rng-v1:{seed}:{name}".encode()).digest()[:8],"little")
    rng_streams = {"initialization":initialization,
                   "quantizer":torch.Generator(device="cpu" if method_settings is not None and method_settings["name"] in ("qar_fp4_adjacent","stochastic_shadow_forward","stochastic_endpoint") else device).manual_seed(stream_seed("quantization")),
                   "meta":random.Random(stream_seed("meta")),"timing":random.Random(stream_seed("timing"))}
    model.configure_quantization({"name":config["training"]["recipe"],
                                  "rotation":"full_hadamard" if config["quantization"]["rotation"] == "hadamard" else "identity"})
    optimizer = config["optimizer_probe"]
    method = MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),
                           AdamWUpdateRule(betas=tuple(optimizer["betas"]),eps=optimizer["epsilon"]))
    matrix_lr = optimizer['learning_rate']
    method_name, adaptation = 'adamw_original_W_byte_adapter','project_H06_bootstrap'
    if method_settings is not None:
        from .baselines.registry import build_matrix_method, matrix_record, MethodRegistry
        registry = MethodRegistry()
        if method_settings['name'] in ('qar_fp4_adjacent','stochastic_shadow_forward','stochastic_endpoint'):
            from .baselines.qar import QarAdjacentAdapter
            from .baselines.rounding import RoundingAdapter
            from .baselines.registry import routing_record
            method=(QarAdjacentAdapter(method_settings['config']) if method_settings['name']=='qar_fp4_adjacent' else RoundingAdapter(method_settings['config'])).bind(dict(model.named_parameters()),default_parameter_groups(model))
            model.configure_quantization({'name':method.config['scale_recipe'],'rotation':method.config['rotation'],'backward':method.config['backward']})
            registry.register(routing_record(method_settings['name'],method.groups,method.config))
        elif method_settings['name']=='trs_backbone_adapter':
            from .baselines.trs import TrsBackboneAdapter,construct_backbone,validate_config
            from .baselines.registry import trs_record
            trs_config=validate_config(method_settings['config'])
            model.configure_quantization({**model._recipe,'name':trs_config['scale_recipe']})
            method=TrsBackboneAdapter(construct_backbone(trs_config['backbone']),trs_config).bind(dict(model.named_parameters()),default_parameter_groups(model))
            registry.register(trs_record('trs_backbone_adapter',method.groups,trs_config))
        elif method_settings['name']=='ewgs_fp4':
            from .baselines.ewgs import EwgsAdapter
            from .baselines.registry import ewgs_record
            model.configure_quantization({**model._recipe,'backward':'ewgs'})
            method=EwgsAdapter(method_settings['config']).bind(dict(model.named_parameters()),default_parameter_groups(model),method.rule)
            registry.register(ewgs_record('ewgs_fp4',method.groups,method.config))
        else:
            method = build_matrix_method(method_settings['name'],dict(model.named_parameters()),default_parameter_groups(model),method_settings['config'])
            registry.register(matrix_record(method_settings['name'],method.groups,method.rule))
        record = registry.require(method_settings['name'],adaptation_label=method_settings['adaptation_label'])
        method_name,adaptation = record.name,record.adaptation_label
        matrix_lr = method_settings['learning_rate']
    run_id = "h07-"+uuid.uuid4().hex
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
        "method":method_name,"adaptation":adaptation,
        "method_settings":json.loads(json.dumps({**method.metadata,"ordered_parameter_groups":[{"name":g.name,"parameter_names":list(g.parameter_names),"weight_decay":g.weight_decay} for g in method.groups]})),
        "seeds":{"model":seed,"data":seed},"data_sha256":None if config["synthetic"] else config["data"]["manifest"]["sha256"],
        "warm_start_sha256":None,"policy_sha256":None,"lineage":{"parent":None if resumed_state is None or resumed_state.ledger.run_ledger is None else resumed_state.ledger.run_ledger.run_id,
        "resume":None if resumed_state is None else envelope["resume"]["sha256"]},
        "endpoint":{"pilot_target_budget":budget,"disposition":disposition,"final_scientific_duration":False}})
    state = TrainingState(model,method,cursor=cursor,ledger=WorkLedger(ledger),microbatch=config["training"]["microbatch"],
                          schedule={"step":0,"point":SchedulePoint(matrix_lr,optimizer["decay"],0.)},
                          configuration=scientific_configuration,rng_streams=rng_streams,
                          process_audit=[{"kind":"launch","pid":os.getpid(),"process_elapsed_seconds":0.}])
    if resumed_state is not None:
        state = resumed_state
        state.ledger.run_ledger = ledger
        cursor = state.cursor
    checkpoint = None
    launch_step = state.schedule["step"]
    interrupted = False
    ledger.record_status("running",disposition=disposition)
    try:
        while cursor.remaining_run_targets:
            train_step(state,select_logical_batch(state))
            if checkpoint_directory is not None:
                state.process_audit[-1]["process_elapsed_seconds"] = time.perf_counter()-project_start
                checkpoint = save_snapshot(capture_snapshot(state),Path(checkpoint_directory))
            if stop_after is not None and state.schedule["step"]-launch_step >= stop_after:
                interrupted = bool(cursor.remaining_run_targets)
                break
        if checkpoint_directory is not None and checkpoint is None:
            checkpoint = save_snapshot(capture_snapshot(state),Path(checkpoint_directory))
    except Exception as error:
        ledger.record_status("failed",disposition=str(error))
        raise
    finally:
        project_elapsed = time.perf_counter()-project_start
        ledger.record({"step":state.schedule["step"],"stream":0,"unique":0,"additional_loss":0,"backward":0,"query":0,
                       "elapsed_seconds":0.,"status":"project_elapsed_observation","project_elapsed_seconds":project_elapsed})
    state.process_audit[-1]["process_elapsed_seconds"] = project_elapsed
    ledger.record_status("interrupted" if interrupted else "completed",disposition=disposition)
    return {"path":str(output),"synthetic":config["synthetic"],"disposition":disposition,
            "stream_targets":state.ledger.stream["targets"],"steps":state.schedule["step"],
            "checkpoint_published":checkpoint is not None,
            "checkpoint":None if checkpoint is None else {"path":str(checkpoint.path),"sha256":checkpoint.sha256},
            "interrupted":interrupted,"scientific_gate":None,"project_elapsed_seconds":project_elapsed}


if __name__ == "__main__":
    raise SystemExit(main())
