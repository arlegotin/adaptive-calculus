# RALUP

Research specification and implementation plans for a representation-aware learned update policy, based on the supplied research brief with cutoff **3 October 2026**.

The package currently implements validated configuration, hashed artifact references, and append-only run records. Training, numerical probes, benchmarking, and protected scoring remain planned work; no experimental improvement or hardware acceptance is claimed.

- [Research and system specification](docs/superpowers/specs/2026-10-03-ralup-design.md)
- [Implementation plan, dependencies, and requirement coverage](docs/superpowers/plans/2026-10-03-ralup.md)

The plan progresses through numerical foundations, the common training harness, comparator/policy implementations, and controlled evaluation. Its scientific claim is conditional on a frozen controller and a local quantized byte-Transformer workload. Hardware feasibility, baseline fidelity, improvement, and cost effectiveness remain to be measured.

Python 3.11 or later is required. The configuration and records modules use only the standard library. From the repository root, install locally into an existing isolated environment and run the implemented commands:

```sh
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m ralup --help
.venv/bin/python -m ralup --version
.venv/bin/python -m ralup config --help
.venv/bin/python -m ralup config configs/smoke.json --purpose smoke
.venv/bin/python -m pytest tests/unit/test_config_records.py -q
```

The local build requires setuptools; tests require pytest. `pyproject.toml` declares the build and test dependencies, without prescribing a numerical framework version. Before local installation, the same CLI works with `PYTHONPATH=src .venv/bin/python -m ralup --help`.

`config` reads a template and optional `--evidence` JSON, validates the requested purpose, and writes canonical resolved JSON to stdout. It does not write files or launch a run. `smoke.json` explicitly uses synthetic bytes, a tiny 1-layer/8-wide model, and two continuation steps; it cannot resolve for training, meta-training, or confirmation. The scientific proxy/core/meta templates keep measured settings null. A core template with no evidence therefore fails resolution for `--purpose train`.

Evidence supplies nested fields matching the template and a `gates` object. Each gate contains `status` (`pass`, `blocked`, `negative`, or `inconclusive`), an `evidence` list of `{ "path": "...", "sha256": "..." }` references, and a `blockers` list. A passing gate requires at least one valid artifact reference and no blockers. Evidence may fill null fields; it cannot override fixed fields. Artifact references validate path and digest syntax here; consumers must verify actual file bytes with `sha256_file` before using an artifact.

The required inputs depend on purpose:

| Purpose | Required gates and inputs |
| --- | --- |
| `doctor` | No measured prerequisites; permits inspection of unresolved templates |
| `prepare-data` | WP00, dependency lock, and environment record |
| `smoke` | Explicit synthetic profile with a finite budget of 1–16 steps |
| `train` | WP00/01/02/04; data revision/manifest, device/lock/environment/tolerances, rotation, measured microbatch/warm start/continuation/B0/recipe |
| `meta` | Training inputs plus WP06, controller, normalizer, and frozen snapshot-sampling proportions |
| `confirm` | Meta inputs plus WP07, B*, unique seeds matching n, analysis, and locked manifest references |

These are resolution purposes, not executable experiment commands. Confirmation configuration alone cannot activate protected scoring; the later scorer must verify the manifest and every planned final checkpoint.

`ResolvedConfig.payload` is a deeply immutable snapshot; its SHA-256 covers canonical JSON with sorted keys, UTF-8, compact separators, and finite numbers only. Templates and supplied evidence remain unchanged.

Optimizer/meta settings validate numeric types, positive rates/epsilon/sigma, two betas in `[0,1)`, positive integer horizon/pair counts, nonnegative AdamW decay, and zero outer Adam decay. Initial values are probes; valid tuned alternatives are allowed in a declared template. Null optimizer/meta settings block run resolution until supplied, while early inspection/data preparation can retain unresolved inputs.

`RunLedger(directory, metadata)` creates `run.json` exclusively; a second creation fails. `RunLedger(directory)` reopens existing metadata. Metadata records identity, UTC/IANA timezone, phase, source commit/dirty diff, lock/environment hashes, resolved config, method/adaptation, named seeds, data/warm-start/policy hashes, lineage, and endpoint. Completion dispositions are appended through `record_status` to `status.jsonl`, referenced by immutable `run.json`.

Supply `resolved_config` as the full JSON snapshot obtained from `json.loads(canonical_bytes(resolved.payload))` and `resolved_config_sha256` as `resolved.sha256`. Creation and reopening verify the digest, compatible phase/purpose, complete schema, required measured fields, and gate provenance. Run phase `explore` accepts configuration purposes `train` and `meta`; `confirm` requires purpose `confirm`; synthetic phase `smoke` requires purpose `smoke`. Revalidation uses the snapshot's purpose. Placeholder or unresolved production snapshots are rejected before publishing `run.json`.

`record(work)` requires `step`, `stream`, `unique`, `additional_loss`, `backward`, `query`, and `elapsed_seconds`. Counts refer to scored targets (raw-byte targets for the specified workload); categories may overlap and are never summed into a synthetic total. Metrics retain optional NLL/count, context work, subregion timings, memory counters, checkpoint hash, nonfinite count, and per-matrix telemetry. Unmeasured fields remain null. Writes flush and fsync; each run directory has one writer. Modified metadata or an incomplete ledger tail blocks further appends.

[Engineering decisions](decisions.md) distinguish specified constants from measured gates. [Progress](progress.json) records the current gate and exact verification command.
