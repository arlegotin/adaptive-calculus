# H06 logical engine and byte scorer

This is engineering infrastructure and synthetic evidence. It does not pass full WP02, calibrate capacity, select a final budget/backbone, train a controller, or unlock official test outcomes.

## Transaction boundary

`TrainingState(model, method, optimizer_state, quantizer_state, histories, schedule, cursor, rng_streams, ledger)` owns the shared state. Physical packing and optional clipping are explicit `microbatch` and `clip_norm` settings. The named optimizer state aliases `method.optimizer_states`; the H05 adapter owns ordered internal and auxiliary parameter groups, proposes all updates, validates all rounded candidates, then commits through its authoritative original-W rule. Scalar AdamW retains its `copy_`/`addcdiv_` operation ordering and Parameter objects.

For indexed training, call `select_logical_batch(state, target_budget=8192)`, then `train_step(state, batch)`. Selection is tentative. The authenticated corpus index, document metadata and current epoch permutation are shared as immutable data; staged carry and data RNG belong to the tentative cursor. Rollover allocates a new compact permutation before shuffling. Successful completion publishes the new pointer, carry, permutation and RNG; failure discards selection. The complete-step clock begins at selection and includes transaction cloning and failure restoration. It never serializes the corpus index at each step.

The optimizer entry rejects a zero selected-target batch before clearing gradients, preparing contexts, snapshotting, or advancing costs. It verifies the declared denominator against actual raw-byte masks. Each physical microbatch calls the adapter's ordinary backward hook, which calls `state.summed_nll(slice)`: a scalar **SUM** with a fresh surrogate graph and one shared detached numerical context. The engine divides accumulated gradients once by the selected logical targets, clips once if selected, and invokes one adapter finish. Only the engine commits shared loss/history, schedule, cursor and stream costs. Adapters cannot publish those fields early or replace Parameters. Fixed scales and their initialization zero flags persist; dynamic scales refresh on the next logical step. Cached-scale endpoint transitions and refreshed realization changes are reported separately.

Failure restores weights, gradients, original Parameter identities, named adapter moments/proposals and other adapter dictionary state, rule settings, model contexts/runtime handoff, quantizer state, histories, schedule, cursor, named RNGs, global Python/CPU/MPS RNGs, and scientific ledger/coverage state. It retains a failed-work event and elapsed cost, records observed nonfinite tensor elements, marks `failed`, and denies further training or checkpoint publication. H07 still owns durable complete-state snapshots and recovery; H06 publishes no checkpoints.

## Three cost views and adapter extension

`WorkLedger.stream` stores completed stream targets, unique targets and logical updates. `passes` stores launched loss, additional-loss, backward, query and forward-context targets; these categories overlap. `elapsed` stores complete-step elapsed samples including failures. The CLI's durable ledger separately records complete-project elapsed time, including setup and ledger I/O; this observation overlaps step timing and must not be added to it.

Ordinary `summed_nll` callbacks charge forward launches automatically. A scalar-loss autograd hook charges backward targets when backward starts, including failed passes. Exceptional methods must call `state.record_work(additional_loss=..., backward=..., query=..., forward_context=...)` for other launched passes, before that work. They must not duplicate counts already charged by ordinary callbacks. If an extra pass reuses `summed_nll`, only its additional/query classification needs an explicit charge; forward/backward totals are already automatic. H05 candidate-validation updates are also recorded and their time is included.

Indexed uniqueness comes directly from the exact cursor without a duplicate interval index. Cursor-free fixtures/API batches maintain coalesced scored-offset intervals keyed by `TargetBlock.document_id`, excluding context and unselected carry. These intervals roll back on failure and belong to H07 ledger state. The three-step repeated fixture records stream **36**, unique **12**, backward **36** and three completed updates.

## Scoring and protection

`score_blocks(model, blocks, evaluation_context={"partition": "fixture" | "development" | a training/query partition, "seed": ...})` constructs independent contexts from actual final weights and a local evaluation RNG. Fixed scales use the owned runtime handoff; dynamic scales recompute. It passes context overrides to forward, preserving the training cache, gradients, history and RNG. Targets are raw bytes 0..255; logits normalize over all 257 classes, with boundary 256 input-only.

Blocks sort by document identity, target interval and selected offsets. Loss arithmetic is FP32; evaluation explicitly transfers per-target losses to CPU, reduces in FP64 using `math.fsum`, and combines per-block/per-article sums in that stable order. BPB is `nll_sum / (log(2) * target_count)`. Duplicate evaluation targets are rejected. The scorer has no protected source loader and rejects an official-test evaluation context. H03's authenticated loader continues denying official test; only future E03 may add a protected scoring gateway.

## Explicit explore bootstrap

`python -m ralup run --phase explore --config PATH` accepts this envelope:

```json
{
  "config": {"...": "complete template or resolved config payload"},
  "purpose": "smoke",
  "pilot_target_budget": 8193,
  "seed": 606,
  "output": "artifacts/runs/my-synthetic-engine-check"
}
```

`config` is the entire schema payload, not the abbreviated object shown above. An unresolved payload can supply a separate `evidence` object. A resolved payload retains `purpose` and `evidence_gates`; envelope purpose must match. Synthetic smoke remains profile `smoke`, source `synthetic-bytes-v1`, explicitly labeled `synthetic_engineering_only`, and bounded by its existing finite smoke horizon. Its run ledger phase is `smoke` even though the engine entry command is `--phase explore`.

Real development bootstrap requires explicit `purpose: "pilot-bootstrap"`, authenticated data/runtime/rotation/microbatch/recipe/optimizer settings, and hashed accepted **WP00, WP01, H02, H04 and H05 engineering prerequisites**. Artifact hashes are rechecked; the H03 source loader authenticates the accepted frozen H02 corpus identity, revision, exclusions, membership and document identities. H02/H04/H05 labels authorize engineering bootstrap only; they are not WP02/WP04/WP06 scientific gate claims. The finite positive `pilot_target_budget` is a caller-selected development envelope, not an accepted warm-start/continuation/headroom duration. Final budget fields may remain null. Disposition is `pilot_bootstrap_development_only`; no checkpoint or scientific gate is claimed.

The CLI resolves the explicitly requested purpose. A failed train/meta resolution never becomes bootstrap. Final train/meta orchestration remains blocked pending later measured calibration and selection work. Data is target-training only in this runner; there is no protected score command. `python -m ralup test-harness` defaults to combined coverage/model/accumulation/scoring engineering checks; existing `--suite data` and `--suite model` remain available.

Actual local MPS commands must set `PYTORCH_ENABLE_MPS_FALLBACK=0` before starting the pinned `.venv` interpreter. CPU/MPS bounded parity, exact same-device duplicate evidence, and local sandbox unavailability are distinct scopes. Transaction overhead is real and awaits H08 measurement; synthetic timing is not a throughput/capacity claim.
