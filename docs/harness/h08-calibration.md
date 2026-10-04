# H08 calibration and separate warmstarts

`calibrate(config: ResolvedConfig, microbatches: list[int]) -> ArtifactRef` and
`train_warmstart(config: ResolvedConfig, partition: str, seed: int) -> ArtifactRef`
require an explicit `calibration` purpose. Real configurations reauthenticate
WP00, WP01, H02, H04, H05 and H07 evidence plus corpus/lock/environment identities.
Synthetic calibration is separately labeled engineering evidence. Neither path
selects B0, final warmstart/continuation duration, convergence or a final gate.

The optional `calibration` section holds output, seed, repetitions, finite
per-arm pilot bytes, finite warmstart bytes, declared continuation bytes, and a
bounded number of proposed duration doublings. Those byte envelopes are callers'
engineering bounds. Existing final budget fields remain unresolved. A real arm
always accumulates exactly 8,192 targets per logical step. Small fixture tests may
use a shorter explicitly synthetic envelope.

```sh
.venv/bin/python -m ralup calibrate --config configs/h08-synthetic.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m ralup calibrate --config configs/h08-real-proxy.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m ralup calibrate --config configs/h08-real-sustained.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m ralup run --phase explore --suite warmstart --config configs/h08-real-proxy.json
```

All actual MPS commands require the authorized process outside the local sandbox.
No precision change, CPU fallback, framework safety-limit disablement or protected
score is available through calibration.

Calibration starts packing at one. Each candidate measures all three implemented
quantization recipe arms in a seeded randomized order, using restored identical
initial weights, cursor and named/global RNG. Each arm then advances its byte
stream across successive logical transactions. Packing grows while the slowest
recipe arm's measured sustained throughput improves. The largest tested packing
is a bounded observed capacity, not a maximum-memory frontier. H06 stages the
cursor and copies, validates and commits the entire update inside synchronized
complete-step regions; its candidate-validation work is retained.

Warmup runs on a disposable H07 clone. The original scientific state hash must
match after restoring global RNG. Warmup passes and elapsed costs survive in the
project ledger. OOM failures use H06's complete rollback, retain failed passes,
halve only physical packing, and require identical document/scored offsets on
retry. All comparison arms are then measured again with the common smaller
packing. Other failures remain terminal and cannot publish a valid checkpoint.

Memory reserves `max(8 GiB, 25% RAM)` for the OS. Driver use must remain below 80%
of the framework recommended working set. The present implementation further
tightens both caps by 0.8 when the observed swap allocation is nonzero. That is a
conservative admission heuristic, not an assertion of current paging activity.
Before/after `vm_stat`, power and thermal observations are retained; pageout and
swapout deltas must be interpreted separately from counters since boot.
RSS, live Metal allocation and driver allocation are sampled separately every
50 ms during training. Their maxima are sampled observations; true Metal/driver
peaks and an additive unique RSS-plus-Metal total remain unavailable. A local
memory fit uses at least two measured packings, records residuals and retains the
reference attention's quadratic context dependence. Owned snapshot tensors have
owner paths, shapes, dtypes and byte inventories; these inventories describe
scientific arrays, not an additive process-memory prediction.

Actual initial and completed training states are transferred, saved, authenticated
and restored. Completed-state snapshots include moments, histories, caches and
RNG, so storage forecasts do not substitute a W-only snapshot. The first same-
process load has uncontrolled filesystem cache state; the separately timed cold
interpreter load includes process startup and authenticates the completed state.
Warm loading is measured subsequently. No OS cache purge is claimed. Snapshot
retention is forecast before writes against current free space. No user file is
deleted. SOAP storage/factorization measurements remain explicitly pending the
actual M02/M07 method.

Each run retains immutable RunLedger metadata and append-only event/status rows.
WorkLedger keeps overlapping stream, additional/query/backward/context passes,
and elapsed work independently. The report includes all warmup, reset, query,
checkpoint and failure events plus complete project elapsed observations in the
runtime ledger. `audit_queries` rejects mismatched observed and charged query
targets. `forecast_cost(measurements, design)` combines measured operation ranges
and the requested counts, retains prior project/parent costs and forecasts actual
snapshot retention. Those ranges are engineering observations, not confidence
intervals, full-program budgets or learning forecasts. Common warmstart costs
remain part of deployment ancestry; calibration/meta exploration remains project
work, and a later deployment owns its additional work ledger.

`train_warmstart` writes distinct immutable `meta_train` and `target_train`
artifacts. Their only real sources are H03's `meta-train` and `target-training`
partitions. Reports authenticate their complete H07 checkpoint, preserve cost
records including checkpoint transfer/write work, and leave learning sufficiency
and duration doubling selection to M07.

`import_warmstart(reference, fresh, partition=..., continuation_targets=...)`
authenticates the parent checkpoint and archived producer bytes without weakening
H07's strict full-state current-runtime check. It verifies exact model geometry,
parameter/dtype/tensor manifest and partition ancestry before copying W bitwise
into the supplied fresh current state. Generated warmstarts require the immutable
cost-complete warmstart report reference, rather than its inner checkpoint alone.
Fixture checkpoints can directly exercise the restricted importer.

The continuation contract preserves the authenticated parent permutation, epoch,
position, carry, cumulative/unique/repeated byte counts and cursor-owned data RNG.
Only its final byte envelope is extended by the declared continuation bytes.
All saved named/global RNG streams are restored. There is no duplicate root data
generator. Paired arms import the same artifact and use the same byte envelope,
so their cursor and randomness begin identically. The supplied current optimizer,
quantizer/history and schedule are fresh declared resets; carrying those owners
is a different experiment. The parent producer's original paths/hashes and stable
archived byte references remain explicit, allowing a current continuation snapshot
to authenticate its historical ancestry without relabeling source hashes. Frozen
final inference import belongs to E03 and has a different contract.

Measured evidence and limitations are in [h08-evidence.json](h08-evidence.json).
WP02 engineering checks are recorded pending independent review; no scientific
gate is asserted by the calibration artifacts.

The importer also authenticates the exact archived `models/transformer.py` and
`models/linear.py` bytes against the current model implementation and rejects
changed static module settings. Other archived producer files may differ under
this expressly restricted import; full-state restore still requires the exact
current complete runtime. This preserves model parameterization without claiming
arbitrary historical code compatibility.

Central calibration `cost.stream.unique` is the union of scored raw source-byte
offsets, identified by authenticated source/document IDs and half-open intervals.
`trajectory_unique_sum` retains summed independent-cursor increments. Runtime
ledger rows retain their run-local cursor meaning. Earlier measured reports are
immutable; authenticated `calibration-accounted-v2.json` sidecars reconcile their
scope and charge accounting/publication overhead without rerunning valid timing
blocks or relabeling their original producer archives.

The observed actual proxy has repeated-index MPS embedding-backward variation.
Initial paired warmstart W/cursor/RNG remain exact, but the first real post-step
bitwise check failed and remains retained. Historical tiny-fixture exact bounds
are unchanged. Root authorized a dated prospective, finite full-step duplicate/
strict-resume contract before method selection, with fixed FP32 ceilings and
independently frozen duplicate envelopes. Its enumerated shape/horizon and any
failed scope are explicit in the evidence; this does not establish a universal
backend bound. Longer horizons, core models, new methods, packing or runtime
changes require prospective characterization before comparisons.

The completed v4 wave failed all 12 enumerated numerical scopes. Cases03 and11
also violated exact cached/active code branches; case11 exceeded the fixed float
ceiling. Other failures include heldout errors above finite independent duplicate
maxima, which are not statistical prediction bounds and do not alone prove a
restore defect. Downstream numerical/scientific acceptance remains closed.
The original zero-bound failure and all prospective failures remain retained.

The evidence indexes case11's paired raw saved W/storage/scales and code-coordinate
audit. Available context7/context8 operands differ and straddle existing E2M1
midpoints; this is saved-input arithmetic, not new model execution or acceptance.
Case03's opposite-side local W was disposed before the later forensic amendment;
case11's earlier context6 numeric operands were never captured. These distinct
forensic limits remain explicit. A versioned read-only observer preserved later
paired producing-context evidence before authorized audited disposable-file
retention. Future numerical remedies require separate prospective design/review.

Warmstart cap enforcement was corrected after review I1: entry admission rejects
before the first transaction; sticky sampled violations are checked before and
after completed transactions and before snapshot/checkpoint/report publication.
Actual completed work and cap observations remain in the terminal failed ledger.
See [h08-fix1-evidence.json](h08-fix1-evidence.json) for the new current-source
focused validation and historical producer references. Prior calibration and
542/16-test evidence retain their original source identities. The original index's
still-open consolidation-log hash issue (M1) is explicitly deferred to E07/root
final; no new circular authentication receipt is claimed.
