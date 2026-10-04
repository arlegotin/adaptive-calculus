# H05 implementation report

Status: DONE. Implemented only S07/S16/S22 optimizer proposal, common method adapter and baseline registry/engineering verification. No trainer, snapshot assembler, ledger, controller, protected scorer, data selection or scientific gate changes.

## Implemented contracts and H06/H07 handoff

- `SchedulePoint(lr, weight_decay, progress)` validates finite/nonnegative scalar schedule values and progress in [0,1].
- `Proposal(reference_after_decay, gradient_displacement, new_optimizer_state)` contains detached, owned mathematical telemetry and new original-W moments. Its additional `authoritative_update` mapping carries the denominator and step size required for exact bypass rounding.
- `AdamWUpdateRule.propose` never modifies W, its gradient, or the supplied old state. Commit copies the already decayed reference into the existing Parameter and calls upstream-ordered `addcdiv_`; it neither recalculates moments nor replaces Parameter identity. S16 telemetry uses the explicit bias-corrected mathematical formula; adding W0+dW is deliberately not the exact bypass implementation. A rounding witness proves the distinction.
- `ParameterGroup(name, parameter_names, weight_decay)` records explicit ordering. `None` uses shared schedule decay; zero excludes decay. `default_parameter_groups(model)` uses H04's explicit six-matrix-per-layer manifest for internal projections and lexically records all remaining auxiliaries under zero decay, including input/position embeddings, norms and output head.
- `MethodAdapter(parameters, groups, rule)` owns `optimizer_states` keyed by manifest name and exposes the current `proposals`. No data/cursor dependency exists. `prepare_step` checks pending proposals. `backward_microbatch` calls the caller's shared `training_state.summed_nll(batch_slice)` and backpropagates its scalar SUM, returning summed NLL and forward/backward counts. H06 owns gradient clearing, quantized contexts, batch selection, divide-once normalization, clip-once, scheduling, transactions and ledgers.
- `collect_proposals(schedule)` is pure with respect to parameters/old moments. `commit_proposals(proposals)` validates all reference/displacement/new-state/authoritative operands and all actual rounded candidate commits before committing any real Parameter. It publishes new optimizer states only after all actual commits, clears pending proposals, and rejects repeat commits of the previous collection. `finish_step` delegates these two stages. Policy/exceptional adapters can override exposed stages; the ordinary identity action uses the authoritative rule commit.
- Adapter work records include ordinary forward/backward passes, declared extra forward/backward passes, optimizer updates and candidate validation updates. Candidate checks really execute a second commit on a detached clone and report those tensor updates; no hidden model passes are introduced. Further commit/runtime failures and later transaction failures require H06 rollback.
- Named schemas: `named.original_W.adamw.v1` stores per-parameter `step`, `exp_avg`, `exp_avg_sq`; step is CPU FP32 scalar for the selected noncapturable upstream path, moments match original W FP32 device/shape. `named.proposal.v1` names reference, displacement, new optimizer state and authoritative operands. `adapter_mutable_state=none.v1` declares no additional histories. Pending proposal tensors are named for H07; completed steps clear them. Rule configuration is separately recorded (`betas`, `eps`, and scalar-path settings), with ordered groups, schedule fields and extra-pass declarations exposed in adapter metadata. Later adapters must extend the state schemas and declare their extra passes.
- S22 `MethodRecord`/`MethodRegistry` require named family/source revision/adaptation, operations, ordered groups, schemas, license, settings and parity status. Unregistered names, missing/mismatched adaptation labels, duplicates and missing fields fail. Original fixtures, paper references and project adaptations have distinct kinds. License/operation/parity blocks are reported as blockers, never defeated baselines; passing an original fixture cannot authorize scientific adaptation claims.
- CLI `python -m ralup verify-baselines --suite adamw [--device cpu|mps] [--output ...]` writes hashed bounded fixture evidence. It pins Torch 2.14.1, records installed source/revision/settings and marks `scientific_gate=null` and fixture claim eligibility false.

## Exact upstream reference

Installed `.venv` Torch is **2.14.1**, git revision `5c4886908584029761b579af026dcfb627c84070`. `torch.optim.AdamW` inherits Adam and sets `decoupled_weight_decay=True`; in this version the chosen scalar implementation is `torch.optim.adam._single_tensor_adam`, not a separate `_single_tensor_adamw` symbol. Installed function source SHA256 is `213669f311feee0831e7a5a567a4a34e7c05a1e348268fbcb8eb65f2ef9acc9d`.

Selected path: FP32, Python scalar LR/betas, `foreach=False`, `fused=False`, `amsgrad=False`, `maximize=False`, `capturable=False`, `differentiable=False`, decoupled decay. The preserved primitive order is CPU FP32 step increment; optional `W.mul_(1-lr*weight_decay)` on an owned reference; `exp_avg.lerp_(grad,1-beta1)`; `exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=1-beta2)`; scalar bias corrections and step size; `(exp_avg_sq.sqrt()/sqrt(bias_correction2)).add_(eps)`; authoritative `W.addcdiv_(exp_avg,denominator,value=-step_size)` after copying the reference. No algebraic refactoring is accepted as bitwise commit parity.

## TDD and focused checks

1. RED before production packages/CLI existed:
   `.venv/bin/python -m pytest tests/unit/test_adamw_adapter.py tests/unit/test_method_registry.py -q`
   Result: **16 failed**, expected missing `ralup.optim` / `ralup.baselines` and unknown `verify-baselines` command.
2. GREEN after implementing optimizer, groups, adapter, registry and CLI: same command, **16 passed in 0.97s**.
3. Iteration findings: an initial telemetry calculation using zero-based upstream `addcdiv_` did not produce the intended CPU rounding witness. Using the explicitly specified S16 mathematical telemetry, while retaining separate exact authoritative operands, made the witness detect the prohibited telemetry-addition commit. JSON initially rejected `TorchVersion` and tuples; canonical metadata now converts these to plain JSON values. Both transient failures were resolved before GREEN.
4. Additional verification covered nonfinite stored moments blocking every candidate before real commits, duplicate commit rejection, and actual target-device fixture behavior. Sandbox focused run: **17 passed, 1 actual-MPS-unavailable skip**. Escalated actual-device focused run with `PYTORCH_ENABLE_MPS_FALLBACK=0`: **18 passed in 1.61s**, no skips.
5. RED for actual rule-configuration metadata:
   `.venv/bin/python -m pytest tests/unit/test_method_registry.py -k metadata -q`
   Result: **1 failed**, `default_registry() got an unexpected keyword argument 'rule'`.
   GREEN after recording actual custom beta/epsilon/scalar-path settings in registry and adapter metadata: focused suite **18 passed, 1 sandbox MPS skip in 1.22s**. This final metadata test is included in the final actual-device full suite below.

The fixed-gradient oracle tests cover four beta/epsilon/decay cases over 40 steps each, zero gradients, initial and later steps, a changed LR, full moment identity, input parameter/state nonmutation and Parameter identity. Separate coverage proves first nonzero-step rounding bypass, explicit default no-decay groups, SUM backward, finite preflight, deliberate moment publication, missing-gradient no-op, required adaptation and honest fixture claims.

## Required negative controls

Temporary source mutations ran `.venv/bin/python -m pytest tests/unit/test_adamw_adapter.py -k fixed_sequence -q --tb=short`, each under `try/finally` restoration:

- Moving decay after displacement: exit **1**, **3 failed, 1 passed, 9 deselected in 0.95s**. Zero-decay case correctly remains equal; nonzero-decay cases fail bitwise weight parity.
- Applying the first-moment update twice: exit **1**, **3 failed, 1 passed, 9 deselected in 0.96s**. Beta1 zero correctly makes repeated replacement idempotent; the other cases fail parity.
- Restored source focused check: **18 passed, 1 sandbox MPS skip in 1.01s**. Mutation stdout/failures are retained in `docs/evidence/2026-10-04-h05-negative-controls.json`; no test-only controls remain in production code.

## Final CPU and actual-MPS evidence

Final current source manifests:

- `PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m ralup verify-baselines --suite adamw --device cpu --output docs/evidence/2026-10-04-h05-adamw-cpu.json`
  Exit **0**, pass; manifest SHA256 `781d84b1913a32d04c6638e2046f6999038e419c16456a5fa2a221eded07a4a7`.
- Escalated local actual-MPS process, fallback disabled **before Torch startup**:
  `PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m ralup verify-baselines --suite adamw --device mps --output docs/evidence/2026-10-04-h05-adamw-mps.json`
  Exit **0**, pass; manifest SHA256 `6e37cfdb04b053079b0e0a8726b229d453190c79563fae583b398aec6d78e811`.

Each device checks four cases ×40 steps with **zero weight mismatches and zero optimizer-state mismatches against its same-device Torch scalar oracle**. Cross-device weights AND moments are separately checked under bounds fixed before measurement: `atol=2e-6`, `rtol=2e-6`. Largest recorded MPS-to-CPU weight difference is `3.4924596548080444e-10`. Cross-device bitwise identity is not claimed. These small fixed-gradient fixtures do not establish whole-model numerical equivalence, optimization quality, speed, scientific gate passage, or test-data outcomes.

## Final full suite and self-review

One final current full suite, escalated actual MPS with fallback disabled:

`PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest -q`

Exit **0**, **409 passed in 18.32s**, **0 skipped**, no warnings/errors. This includes all final H05 tests and the repository's current CPU/actual-MPS integration tests. Permanent source/test hashes and verification summary are retained in `docs/evidence/2026-10-04-h05-verification.json`.

Self-review read all new optimizer/registry/check code and CLI diff, checked explicit group coverage and no-decay semantics, original-coordinate state/Parameter identity, pure proposal ownership, exact source ordering, finite preflight and publication boundaries, schema/configuration names, extra work accounting, fixture claim limits, and restoration of both mutants. `git diff --check` passed. No unresolved correctness concerns. Full-step rollback/safe snapshot assembly remain deliberately owned by H06/H07.

## File inventory

- Created `src/ralup/optim/{__init__,base,adamw}.py`.
- Created `src/ralup/baselines/{__init__,registry,checks}.py`; checks is the necessary implementation for the requested CLI fixture command.
- Modified `src/ralup/__main__.py` only to register/dispatch `verify-baselines`.
- Created `tests/unit/test_adamw_adapter.py`, `tests/unit/test_method_registry.py`.
- Created four permanent evidence manifests under `docs/evidence/2026-10-04-h05-*`: CPU, actual MPS, negative controls and verification.
- Created this task report `.superpowers/sdd/2026-10-03-ralup/H05-report.md`.

Commit subject: `feat: add base update protocol and exact AdamW adapter`.
