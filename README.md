# RALUP

Research specification and implementation for a representation-aware learned update policy, based on the supplied research brief with cutoff **3 October 2026**.

Current research disposition: **exploratory and inconclusive**. WP04 has no accepted robust B0; WP06 needs revision after observed noise and horizon-dependent rank reversals. Final C/LR/B*, training budgets, a justified paired pilot, locked n and confirmation manifest are absent. Official test remains protected. Engineering task completion does not establish an improvement or efficiency claim.

All 32 engineering task gates are closed. The whole-branch review and its six-finding consolidated fix are accepted, with 32 focused regressions and 18 adjacent cases. A separate narrow task corrected invalid replication-cost fixtures without changing product code. Fresh CPU integration at `f2394fd` completed **1,287 passed, 76 skipped and 2 deselected**. See [current research status](docs/harness/current-research-status.md) and [final integration evidence](docs/harness/final-integration-evidence.json). Native availability was masked for this CPU suite; historical native failures and resource limits retain their original scope. No scientific advantage is established.

Complete-state restoration now requires both archived source identity and a matching live Python/numerical-runtime/platform/backend fingerprint. Historical archives remain readable for the separate authenticated model-only reset and DATA-only contracts. Measured pre-publication intervals remain cost lower bounds; complete deployment seconds require an authenticated enclosing boundary for every warm ancestor and child. Ranking exposes the warm-inclusive mean separately from its aggregate sum; offline META/tuning and unknown observer/project administration stay separate.

The current handoff is documented in [the E07 contract](docs/harness/e07-contract.md) and [the evidence index](docs/harness/e07-evidence.json). The index binds the local immutable evidence root, separately recreated core environment, plotting environment, raw report inputs, standalone figures, exact historical preservation and finite replay receipts. These are local research artifacts; source and dataset rights limitations remain recorded.

To regenerate a report from an immutable evidence file, use an explicit compatible interpreter and a new destination:

```sh
PYTHONPATH=src /path/to/locked/python -m ralup report --evidence /path/to/evidence.json --sha256 EXPECTED_SHA256 --output /new/report
sh reproduce.sh /path/to/locked/python /path/to/evidence.json EXPECTED_SHA256 /new/reproduction
```

`reproduce` authenticates evidence and regenerates the report twice, comparing exact output hashes. The separate source-strict numerical replay and interruption/resume receipts are linked in that evidence. It executes no arbitrary commands supplied by an evidence file and never opens protected scores. `--manifest` currently returns an explicit reopened gate for a confirmed study without a supported report-evidence adapter; this exploratory study has no confirmation manifest.

The package implements validated configuration, hashed artifact references, append-only run records, environment discovery, FP32 numerical checks, the common training harness, comparator and policy implementations, exploratory calibration, guarded confirmation infrastructure, packed export and reporting. Actual measurements retain their original producer, runtime and scope. Scientific finalist selection and confirmation remain unresolved; no experimental improvement is claimed.

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

`doctor` writes a redacted hardware/runtime report and prints its SHA-256. `--probe` separately launches a fresh interpreter with silent MPS fallback disabled and writes a uniquely named operation report. A blocked or negative probe returns exit status 1; missing optional operators remain individually recorded.

```sh
.venv/bin/python -m ralup doctor --output artifacts/environment/environment.json
.venv/bin/python -m ralup doctor --output artifacts/environment/environment.json --probe mps
```

The successful stable stack is locked for CPython 3.14 macOS arm64 in [requirements.lock](requirements.lock). The package can import without numerical dependencies; probes require the installed stack. [Asset licenses](licenses/ASSETS.md) and [wheel provenance](vendor_manifest.json) retain all dependency and bundled notice hashes. Operator-smoke bounds and measured errors are evidence for F02 only; format oracles, assistance bounds, full snapshots and scientific tolerances require later gates.

`check-numerics` executes the complete F03–F05 numerical foundation and writes a hashed report; `--suite quantizer` retains the F03/F04-only scope. CPU is the default device; actual MPS checks require fallback disabled before starting the interpreter. Blocked or negative checks return exit status 1.

```sh
.venv/bin/python -m ralup check-numerics --suite quantizer --device cpu --output artifacts/checks/quantizer-cpu.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m ralup check-numerics --suite quantizer --device mps --output artifacts/checks/quantizer-mps.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/unit/test_e2m1.py tests/integration/test_e2m1_mps.py -q
```

The quantizer returns reconstructed FP32 values and uint8 numerical ordinals. [Numerical provenance and measured limits](docs/numerics/e2m1.md) distinguish ordinals, actual encoding nibbles, significand parity, and FP32 scaled-coordinate rounding. Scales must broadcast, be finite and positive, and share the FP32 input device; row scales normally have shape `[rows,1]`. Original NaN/Inf inputs are rejected. Division overflow from finite inputs saturates; reconstruction overflow raises a storage error. A device integer/FP32 path preserves subnormal operands, underflow rounding and tiny direction signs on MPS. This implements simulated W4A32 rather than the full OCP MXFP4 block/scale format. The F04 interfaces and measured numerical limits are documented in [quantization context provenance](docs/numerics/quantization-context.md). Bounded actions and discriminating quadratic diagnostics now pass CPU and actual MPS numerical checks; [action contracts and measured bounds](docs/numerics/actions.md) record the evidence. Optimizer integration, source parity and scientific training gates remain next.

`hadamard`, `to_storage` and `from_storage` use explicit `identity` or
`full_hadamard` IDs. `compute_scales(u, recipe, previous)` returns detached FP32
scales; `fit_gaussian_e2m1()` returns the deterministic fit, independent integration
residual and conventions. The caller owns fixed-scale handoff and prepares one
context per logical step. Reuse it across fresh microbatch views:

```python
from ralup.quantization import prepare_context, quantized_view, to_storage

context = prepare_context(w, {"name": "row_fixed_max", "rotation": "full_hadamard"}, 0, rng=None)
# quantized_view returns storage Q(W@R); transform the input consistently.
output = to_storage(x, context.rotation) @ quantized_view(w, context).T
output.sum().backward()
# A second physical microbatch constructs another fresh view using context.

next_context = prepare_context(w, {
    "name": "row_fixed_max", "rotation": context.rotation,
    "previous_scales": context.scales,
    "previous_zero_groups": context.scale_zero_groups,
}, 1, rng=None)
```

`row_dynamic_max` recomputes at explicit preparation; row Gaussian reference is
`quest_fp4_row_reference` (trust backward by default). Whole-tensor sensitivity
recipes are `tensor_fixed_max`, `tensor_dynamic_max` and
`quest_fp4_tensor_reference`. Named backward options are `ste`, `trust` and
`clipped_ste`. Runtime tensor recipe fields are caller-owned state, separate from
canonical scientific JSON. All continuous state remains FP32; ordinals/flags are
uint8/bool. No optimizer history or supplied RNG is advanced. Supplying `codes`
reuses an externally generated realization; stochastic generation is reserved
for M05. The project refit, full-row rotation and physical trust threshold are
explicit adaptations of paper mathematics; QuEST release-fixture parity is
reserved for M07.

[Measured F04 evidence](docs/numerics/f04-evidence.json) retains CPU/MPS report hashes,
separate sandbox-blocked evidence and the current-suite verification command.

## Frozen confirmation gate

`python -m ralup plan-confirmation --freeze --design PATH` authenticates the E02
plan and refuses with named missing evidence while B0, final C/LR/B*, the genuine
paired pilot and locked n remain unresolved. It creates no scientific manifest
from engineering fixtures or caller eligibility flags. [E03 contracts](docs/harness/e03-contract.md)
describe the immutable fields, complete terminal inventory, append-only access
history and synthetic-only engineering verification.

The protected gateway is a procedural local-owner guard, not OS security.
Training/search cannot open protected targets through it; H01/H02 non-scoring
data audits remain separate. Protected inspection makes subsequent changed
claims exploratory and requires fresh independent confirmation.

`python -m ralup confirm --manifest artifacts/confirm_manifest.json` and
`python -m ralup analyze-confirmation --manifest artifacts/confirm_manifest.json`
authenticate the original E02 prerequisite disposition and exit 2 while the
scientific manifest is absent. **WP08 is not attempted; its prerequisites remain
blocked.** No protected WikiText score or positive scientific claim is produced.

[E04 execution and inference](docs/harness/e04-contract.md) implement locked
paired execution, full-state interruption/resume, one-way final scoring, retained
divergence/invalidation costs, and both two-sided 97.5% paired intervals.
Numerically executed synthetic byte fixtures exercise success, null, inconclusive
and divergence outcomes. Their endpoint analyses explicitly remain engineering
evidence and do not select a scientific n, budget, controller or comparator.

The E07 I1/I2 report fix preserves positive quality with unknown costs and authenticates declared operational primary ledgers. Recorded CPU/native replay predates that report-source change; focused report-contract checks cover the fix. Historical full-state checkpoints do not gain compatibility with the new source identity.
