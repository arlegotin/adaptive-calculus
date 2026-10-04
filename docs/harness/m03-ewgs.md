# M03 EWGS mechanics and named FP4 adaptation

The implementation independently codes the published EWGS equations. Pinned
source `cvlab-yonsei/EWGS@56c654cb893d53563eb352dd591d1450c34bdd15` was inspected
under GPL-v3; none was copied. Permanent registry metadata retains exact inspected
primitive/estimator/license hashes, copy status, independent local-research status,
and unresolved release reproduction. Uniform fixture parity, FP4 adaptation
mechanics, actual operation support, software rights and later scientific
eligibility are separate. A block does not defeat this baseline.

`ewgs_uniform_fixture` names the `[0,1]` primitive
`q=round(x*(L-1))/(L-1)`, whose backward is
`g*(1+delta*sign(g)*(x-q))`; clipping and physical affine maps lie outside the
original primitive. `ewgs_fp4` uses `z=clip(U/(6*s),-1,1)` and
`q=Q(U)/(6*s)` with cached detached scales, an inclusive representable-range
mask, and the physical upstream times the dimensionless residual factor. The
storage-to-original-W rotation acts once. Auxiliary parameters use ordinary
AdamW; all moments remain original-W.

The following conventions were declared before outcomes. Each adaptive estimate
covers one whole internal projection tensor, including all its scale groups.
It uses one complete logical training batch and its SUM divided by logical target
count, independently of ordinary physical microbatch packing. Its loss substitutes
one owned detached normalized quantized leaf, maps it to physical storage through
`leaf*6*cached_detached_scale`, and holds other numeric quantization caches fixed.
Hessian and gradient population std use this same leaf. Delta is
`max(0,(mean Rademacher trace/N)/(3*std_population(g)+1e-12))`.
Probes have a fixed explicitly configured count, with an independent adapter-owned
CPU-generator stream transferred to the actual device. Cadence is logical-step
modulo the configured positive integer, starting at step zero; frozen delta is
reused between refreshes. There is no release-style three-batch average, sample
std, or adaptive probe-stopping tolerance. These departures remain named.

`FIXTURE_CONFIG` is a declaration for bounded fixtures, not an implicit product
configuration: the constructor requires the entire field inventory. Fixed delta
is supported first. Adaptive mode requires explicit pilot relevance and operation
support; synthetic tests exercise the mechanism and establish no pilot relevance.
M07 owns later measured relevance. Tiny-shape actual MPS support cannot authorize
higher-order execution for an untested method/model/horizon/packing.

H06 retains ordinary SUM/divide/clip, selection, transaction and commit. Adaptive
passes call `record_work` before launches. Each extra-loss forward and its first
gradient have unique launch IDs and share one `execution_id`; their `extra_loss`
and `extra_loss_backward` classifications are one data exposure. Each HVP has
its own execution ID. `gradient_only_target_bytes` is zero in this implementation;
raw additional-loss/backward/loss-target counters overlap and must not be summed.
`ewgs_work` records these distinct classes and complete adapter extra elapsed time,
including failures. The analytic estimator reports owned leaf/probe array bytes
and measured elapsed time; no dense Hessian is formed. Peak retained autograd
graph allocation remains unmeasured, not zero. Whole-step time includes context,
probe construction/transfer, snapshot/rollback and ordinary work.

Trusted `ewgs_adapter.v2` owns explicit config, projection deltas, CPU probe RNG
bytes, last refresh step, complete estimator telemetry, original-W AdamW moments,
proposals and snapshot schemas. Reconstruction rejects unknown owners/rules,
forged estimator geometry/cost/delta conventions, missing projections and stale
cadence or context deltas before scientific transfer. Probe state stays on CPU.
Ephemeral `NormalizedLeafView` objects never enter the model cache or snapshot.
Full-runtime identity remains strict: no old-source migration is implied.
Actual adapter/estimator elapsed times are retained but excluded from scientific
replay hashing, while counts and RNG remain included.

Evidence: [m03-evidence.json](m03-evidence.json) points to NEW permanent owned
Data-store attempts with startup sources, commands, raw stdout, outcome/status,
measured costs and inventories. Storage ceilings/reserves are engineering safety
only; no science budget, protected outcome, calibration winner, baseline defeat,
release reproduction or full Cartesian neural gate is claimed.

## Review fix I1: actual HVP launches

For graph-free constant first derivatives, the zero Hessian is inferred without
a higher-order autograd call. Requested `probes` and allocated `probe_bytes`
remain explicit; `hvp_backward_passes` is zero, `extra_backward_passes` is one,
and `zero_hessian_inferred` is true. No HVP exposure ID or backward target bytes
are charged. For a higher-order graph, the HVP recorder runs immediately before
each actual derivative launch; finite trusted counts are exactly P HVPs and
1+P backwards, with the inference flag false. Inferred zero requires zero
trace/delta. Paired extra-loss forward/first-gradient IDs remain one exposure.

Trusted adapter and owned telemetry schemas are v2 because the complete nested
inventory now includes actual HVP count and zero-Hessian inference. Exact source
identity and finite constructors remain mandatory; no v1 compatibility waiver
or migration is supplied. The original v1 source, startup receipts and accepted
nonlinear/MPS outcomes remain immutable historical evidence.

[m03-fix1-evidence.json](m03-fix1-evidence.json) retains the narrow CPU affine
accounting/safe reconstruction/rejection regression and nonlinear analytic
branch sanity, including RED and GREEN startup helpers/product/tests/lock/F02,
stdout/status and scoped time. No accepted neural wave, MPS gate, full covering
or CLI suite was repeated for this accounting fix. Earlier startup dependency
omission, pilot relevance and saved-graph/cost unknowns retain their old scopes.
