# H07 complete resume and safe publication

This is snapshot/replay engineering on synthetic fixtures. It makes no capacity,
quality, duration, controller, scientific-gate or protected-test claim. H06's
completed transaction is the only capture boundary: staged selection, live work,
uncommitted proposals and failed state all reject capture.

`capture_snapshot(TrainingState)` owns detached, dense, contiguous CPU copies of
all arrays and safe metadata. `restore_snapshot(snapshot, device)` constructs a
new model, binds the adapter to its actual original-W Parameters, restores CPU
noncapturable AdamW scalar counters and device-resident original-W moments, and
restores detached quantization caches. Subsequent forwards construct fresh
surrogate graphs. Capturing and repeated restoration never share mutable state.
Parameter object identity is preserved through every subsequent adapter commit;
an object identity cannot span separate interpreter processes.

## State inventory

| Owner | Complete stored fields |
| --- | --- |
| Model | `ModelConfig(layers,width,heads)`; every named FP32 weight; every named gradient or explicit null; root and every child module training mode; `_recipe`, `_runtime`, `_contexts`, `_context_step` |
| Quantization context | `scales`, `codes`, `storage_values`, `storage_u`, `gradient_mask`, `zero_groups`, `scale_zero_groups`, `rotation`, `grouping`, `backward`, `recipe`, `step`, full metadata |
| Runtime and quantizer | All named runtime handoffs including previous fixed scales and initialization zero flags, warm-start/codes if supplied; complete `quantizer_state` |
| Method | Versioned method identity; ordered `ParameterGroup(name,parameter_names,weight_decay)` records; AdamW `betas`, `eps`; exact `optimizer_states`, `proposals`, `snapshot_schema`; registered method extension fields |
| Original-W optimizer | Every parameter's `step`, `exp_avg`, `exp_avg_sq`, or empty initial state; a completed state may not omit one parameter's moments; `optimizer_state` is reconstructed as the adapter's authoritative alias |
| Histories and features | Every history field, including matrix `previous_codes`, `two_steps_ago_codes`, `cached_codes`, `scales`, loss EMA and update counter; explicitly declared safe shared-field or registered method normalizer/features/refresh telemetry extensions are preserved |
| Schedule and packing | Full scheduler dictionary including step, `SchedulePoint(lr,weight_decay,progress)` and any safe carry; physical `microbatch`, `clip_norm`; CLI scientific configuration retains the fixed 8,192 logical target budget |
| Data cursor | Full H03 safe state: authenticated source identity and hashes; epoch, permutation position, compact uint32 permutation, carry score interval/mask derivation, cumulative/unique/repeated target counts, final run budget, authoritative `rng_states.data`; separate materialization/read counters are audit observations |
| RNG streams | CLI `rng_streams.initialization` (actual post-constructor CPU generator state), device-specific `quantizer`, independent `meta` and `timing`; authoritative data RNG remains cursor-owned; all caller-supplied named Random/torch generators; global Python, CPU torch, and available MPS torch state |
| Work ledger | Full `stream`, `passes`, exact cursor-free scored-offset `coverage`, `elapsed`, all completed and prior discarded-failure events including nonfinite/failure costs; attached immutable RunLedger metadata reference and location |
| Provenance | Complete canonical configuration and its SHA-256; declared authenticated ArtifactRefs; exact archived producer bytes plus hashes/references for every producer RALUP Python source file, dependency lock and frozen F02 environment artifact; independent strict current-runtime identity check; cursor fixture content or reauthenticated frozen H02 index/source/membership/exclusion/document identities |
| Process audit | Prior process rows; on restore a new UTC/pid/resume row with process elapsed zero and captured past step seconds; CLI records each new process's project elapsed separately |
| Publication | `ralup.training_snapshot.v2`, tensor checksum and exact dtype/shape manifest, manifest ArtifactRef, publication marker, declared milestone membership |

No policy/outer Adam/ES/controller owner exists in H06. H07 does not invent those
owners or silently serialize arbitrary future objects. Later task owners must
register their schemas before being checkpointable.

## Safe format and recovery

The F02-pinned `safetensors==0.8.0` container holds owned CPU arrays. Canonical
finite JSON records exact registered typed metadata. The loader verifies the
manifest ArtifactRef, tensor payload checksum, tensor key/shape/dtype manifest,
model/gradient/optimizer geometry, completed EMA/history/context/fixed-scale
handoff inventories and cross-owner geometry/types, and every actual external
artifact before transferring scientific arrays to the target device. Capture
also validates every supported child module type, field inventory and static
setting. Nonempty module/Parameter hooks, unknown child owners and unsupported
child context handoffs are rejected; callbacks are never serialized. Child
training modes are safe numeric metadata and are restored explicitly.
Unknown roots/nested schemas, executable values and nonfinite payloads fail
closed. No pickle, dynamic import, eval or artifact-selected constructor is used.

Producer authentication reads the archived bytes, independent of the current
checkout. `load_snapshot` can therefore inspect an authenticated historical
producer after its source/lock/environment paths have changed. Those archived
producer references, including matching configuration lock/environment refs,
authenticate against the archive; data and other external refs still authenticate
their actual objects. `restore_snapshot` separately requires exact current
source/lock/environment identities before full-state reconstruction. Version 2
has no implicit version-1 migration or arbitrary compatibility option. An
explicit versioned, tested old/new-identity migration record is required before
any future full-state compatibility change. Restricted model-only warm-start
import belongs to H08 and is absent here.

One writer owns a snapshot directory. Files are exclusively created and
flush/fsynced in a hidden same-filesystem staging directory. Exact producer
source/lock/environment bytes are separately flush/fsynced in a content-addressed
`producer/` directory, with per-file hash/length descriptors. That directory is
fsynced, atomically renamed to an immutable generation, and the store is
fsynced. The actual generation is then reloaded and verified before publishing
an atomically replaced, flush/fsynced `latest.json` reference. A flush/fsynced
`published.json` marker preserves each successfully published manifest digest
for authenticated fallback. Newly created directory parents are also fsynced.
The last two verified generations plus declared milestones survive retention;
invalid generations are never counted as valid retention slots.

`recover_snapshot(directory)` first authenticates latest, then checks published
markers newest first. It never promotes a complete but unpublished orphan or
rehashes a corrupted manifest into acceptance. A failure before the latest
replacement retains the previous valid generation. An exception after the
replacement can leave a fully verified new generation published; recovery
validates it again. The API does not promise that a raised post-publication I/O
error means publication did not happen. Hidden interrupted staging data is
ignored. Hash references authenticate integrity, not a malicious writer who can
replace both payload and its trusted expected references.

Mutable pointer/RNG/permutation/carry are copied. Large immutable corpus indexes
stay authenticated disk references: restoration reopens the H03 index rather
than constructing millions of block tensors. Snapshot authentication/index
reopening are real setup/I/O costs; H08 owns their capacity measurement.

## Explicit CLI save and resume

Existing `run --phase explore --config PATH` retains explicit smoke or
pilot-bootstrap authorization. Add these optional envelope fields:

```json
{
  "checkpoint_directory": "artifacts/runs/my-run/checkpoints",
  "stop_after_steps": 1,
  "resume": {"path": ".../manifest.json", "sha256": "..."}
}
```

`stop_after_steps` requires a positive count and checkpoint directory. The
runner saves after each completed logical step and can stop at that boundary;
resume must have the same canonical scientific configuration, seed and final
byte budget. It starts a new RunLedger with parent-run/checkpoint lineage while
retaining the old WorkLedger's cumulative passes, stream, elapsed and events.
Output identifies the actual checkpoint reference and whether execution stopped
with targets remaining. Resume does not reinterpret prior WP00 local evidence
as rights/parity acceptance of future methods.

## Registering later method state

Trusted method modules must register at import in every interpreter:

```python
register_state_schema("my_features.v1", MyFeatures,
                      ("mean", "variance", "count"))
register_method_schema("my_adapter.v1", MyAdapter,
                       ("optimizer_states", "proposals", "snapshot_schema",
                        "features", "refresh_telemetry"), MyAdapter,
                       shared_fields={"histories": ("normalizer", "pending_refresh")})
```

The dataclass registration lists every field in declared order. The adapter
registration lists every instance attribute except `parameters`, `groups` and
`rule`; its trusted constructor accepts `(parameters, groups, rule)`. Optional
`shared_fields` explicitly declares additional top-level `histories`,
`quantizer_state` or `runtime` owners under that method schema version. Required
completed-step matrix histories, EMA and handoffs remain mandatory; ordinary
initial state may omit them until a completed step exists. The `data` name is
reserved in root `rng_streams`; its single owner is `cursor.rng_states.data`. New nested
objects must be registered dataclasses or plain safe metadata/tensor arrays.
Bump schema names when changing ownership/meaning; do not discard unknown
fields, auto-register types, or import type paths supplied by an artifact. The
current update-rule/model schemas intentionally support H06 ByteTransformer and
original-W AdamW only; a later rule must add explicit validation/registration in
its own task. Cross-process tests must import the real method registration.

## Replay evidence and limits

CPU six-step uninterrupted versus three-step + atomic save/load + three-step
continuation is exact for fixed max, dynamic max and Quest reference recipes,
with Hadamard storage, nontrivial moments/histories/EMA/scheduler, partially
consumed target blocks and epoch rollover. Actual MPS uses the same zero
same-device duplicate-variation bounds predeclared in authenticated H04 evidence;
these bounds were never loosened. Fixed-scale replay also runs in a fresh CPU
and MPS interpreter. Weight-only reconstruction demonstrably diverges.

The scientific replay hash excludes wall-clock observations, new-process audit,
RunLedger identity/location and cursor read/materialization counters. It includes
all scientific arrays/RNG/coverage and deterministic completed/failure costs.
Past elapsed costs remain stored and are checked separately. Same-device exact
replay is not a CPU-versus-MPS bitwise-parity or large real-corpus resume claim.

Initial implementation verification at `71aa32a` (historical evidence):
`PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m
pytest -q` on actual local Metal: **489 passed in 41.79s**, no skips or failures.
This includes all 46 H07 tests and all four actual-MPS cases. Focused sandbox CPU:
**42 passed, 4 skipped in 6.55s** (actual MPS unavailable only in that sandbox).
The final full raw log is `artifacts/checks/h07-full-tests-final.txt`; its SHA-256
is recorded in the local H07 implementation report.

Review fix round 1 uses snapshot schema **v2** and has its own current evidence:
CPU covering files **79 passed, 4 sandbox-MPS skips in 8.09s**; authorized actual
MPS covering files **83 passed in 23.77s**, no skips/warnings/failures, fallback
zero before startup. The commands target `tests/unit/test_checkpoint_integrity.py`
and `tests/integration/test_resume.py`. Current source/test/log/lock/environment
hashes and initial full-suite historical evidence are recorded in
[h07-fix1-evidence.json](h07-fix1-evidence.json). No full-suite rerun was claimed
for this scoped round.

M01 extension: `matrix_adapter.v1` is an explicit trusted constructor, restricted to the model's exact projection manifest and no-decay auxiliary groups. Registered rule envelopes are `{schema:muon_project_fp32.v1|muon_plus_fp32.v1,config:<complete exact defaults-key inventory>}`. Constructors are selected only from a local two-entry table, never artifact-selected imports. Old AdamW `{betas,eps}` encoding remains unchanged. Unknown rule/schema/fields, mismatched named groups, cross-owner moment inventory/geometry, missing completed moments and uncommitted proposals reject before scientific transfer. Matrix state is original-W `{momentum}`, auxiliary state is original-W AdamW; only actual Adam scalar counters transfer back to CPU. Rule/adapter schema checks and source compatibility remain strict. No old-source full-state migration is inferred.
