# N01 dated numerical correction amendment (accepted 2026-10-04)

2026-10-04. Root authorizes this appended, separately reviewed correction after H08 clean review and before M01. This document is the task design to migrate into durable spec/plan records BEFORE product implementation. Implementation details not fixed below are ordinary worker choices within these contracts; deviations affecting mathematical semantics, scope or acceptance go to root before outcomes.

## Motivation and fixed mathematical contract

H08 isolated nonzero actual-input MPS embedding-backward variability from identical source/state/input/RNG, with exact forward outputs. The unchanged v4 prospective study completed all12 scopes and all12 failed their frozen acceptance. Two scopes additionally changed discrete cached codes; case11 also exceeded floating ceilings. Saved case11 local operands differed and straddled quantizer boundaries; none of385 retained context7/endpoint8 changed-code coordinates had identical local U/W and scale. Earlier case03 opposite-side values were discarded; context6 local inputs were never captured. These forensic limitations and every old failure/cost/source identity remain historical.

Candidate forward is the existing gather Y[n,:]=W[I[n],:]. Candidate derivative is the exact mathematical sum G_W[v,:]=sum_{n:I[n]=v}G_Y[n,:], realized as A.T@G_Y for constant selector A[n,v]=[I[n]=v]. Preserve FP32 model parameters, names/counts/initialization, boundary semantics and differentiability of the upstream gradient. This changes the shared numerical realization, not the architecture or mathematical objective. Determinism and faithful functional/higher-order behavior are empirical acceptance gates, not consequences claimed from matrix notation. A selected primitive is identical across all methods and explicitly identified in runtime/source/compatibility records.

## Prospective bounded verification scope

Use the current2-layer/width128/4-head proxy, original-W AdamW, FP32 actualMPS with fallback0, target_train authenticated parent and continuation cursor/RNG, logical8192 scored bytes, physical packing64, all three current recipes (row_fixed_max, row_dynamic_max, quest_fp4_row_reference),8-step horizon. This targets the observed failing real shape and difficult quest64 path without repeating the12-case partition/packing Cartesian study. CPU fixtures also require exact deterministic repeat/resume with their named complete scientific owners.

After candidate unit/parity/derivative tests pass, stabilize source and freeze/hash the complete resolved design before actual-shape outcomes. For each recipe run two independently reconstructed uninterrupted8-step arms U0/U1 and one4+4-step arm R with fresh-interpreter strict restore at4. Total scope72 continuation steps/589824 scored bytes, plus explicit finite component tests/smoke/warmup/timing envelopes, all charged. Authenticate exact same initial scientific tensors, named/global RNG, data cursor/input sequence, model/recipe, schedule and runtime. Compare every recorded scientific owner and summed NLL at each matched step; retain checkpoints at4/8 and exact hash manifests each step. Identity excludes elapsed/process/path/run IDs, retains all scientific state/work counters.

Selected ordered path must show zero same-device numerical variation in this scope: bitwise duplicate U0/U1 and matched U0/R scientific owners/NLL/gradients where the exact contract applies, with current CPU/MPS cross-device loss/gradient parity bounded by historical applicable atol=rtol1e-4. Existing specialized stricter action/rotation contracts remain unchanged. No stochastic envelope estimated after seeing outcomes replaces this proposed exact duplicate scope. If another backend operation causes nonzero variation, reject/diagnose the scope and seek a separately adjudicated prospective remedy; do not silently accept it. No universal exactness claim for other methods, core model or longer horizons follows.

Small component tests include repeated/absent indices, token256, analytic sum, a deliberately wrong derivative negative control, finite-difference/higher-order checks, and actual functional-gradient use. Reuse authenticated saved H08 component inputs as a read-only negative-control provenance comparison; do not require a stochastic old path to reproduce nonzero variation on every short repeat. New-source actual-input repeats test the candidate directly.

## Capacity, retention and provenance

Include selector/GEMM allocations/transfers and complete new-source steps in actual synchronized memory/time accounting shared by every method. Start physical packing1, observe a bounded geometric search around old64, then sustained randomized original/new-source or prospective recipe arms as appropriate; retain observed stopping criteria and short-block limitations. Specify finite timing envelopes from observed H08 wall durations BEFORE launch, not final scientific budgets. Changed performance invalidates use of old packing forecasts, not historical timings.

Retain exact startup producer identity, failed witnesses and BOTH local quantizer operands/scales/codes/context step for changed-code coordinates. New task-generated retained+scratch<=2GiB and free disk>=1GiB; bounded summaries/discard audits only for explicit reconstructible operationally completed copies. Existing H08 failures/warmstarts/user files never deleted. Preserve strict old full-state resume. Model-only historic parent compatibility requires explicit tested identical forward/parameterization/dtype/tensor manifest, cursor/RNG and expenditure authentication; otherwise generate charged new parents.

## Shared-file ownership and downstream handoffs

| Owners | Shared boundary and obligation |
|---|---|
| H04/N01 | transformer embedding implementation; preserve forward/names/counts/initialization and canonical model ownership |
| H07/N01 | trusted snapshot construction/validation and primitive/source identity; strict full-state compatibility, no arbitrary callable state |
| H08/N01 | model-only parent import, source/cursor/RNG/cost provenance, current capacity reports; historical failures remain closed |
| M01/M02 | every optimizer shares the selected derivative; new optimizer/factor state and changed costs require own validation |
| M07/M11 | accepted packing/runtime/scope and immutable historical parents; longer/different rollout scopes validated prospectively, scientific budgets measured |
| M03/M08/M10 | higher-order/functional-gradient compatibility; own adapters do not bypass the shared primitive or omit its costs |
| E01/E02/E03 | exact new-source identities, scope limits, full-state pool compatibility and new capacity/storage forecasts before lock/scoring |
| E07 | durable amendment, all old failures/forensic limits/attempt costs and accepted limited new evidence; scratch-free reproduction |

Wrong-decision cost: changed derivatives, functional behavior, backend reproducibility or resource cost could invalidate dependent new artifacts. Reject the new path and use a separately tested faithful alternative; never relabel old source hashes or enlarge observed bounds.

## Durable execution order and records

This accepted amendment is appended to the original 2026-10-03 drafting record. Execute **H08 review → N01 implementation/review → M01**. The original 31-task description is historical drafting text; N01 adds a separately reviewed correction. The prospective resolved design will be frozen in `docs/harness/n01-design.json` before new actual-scope outcomes. Durable results will be linked from `docs/harness/n01-evidence.json`; the complete worker report lives in `.superpowers/sdd/2026-10-03-ralup/N01-report.md`. No H08 scope, source or clock is relabeled.

## Dated execution supplements (2026-10-04)

The text above preserves the pre-product accepted design. Subsequent root rulings were versioned and frozen before their own affected work; they do not change the primary72-step numerical design, source, bounds or historical failures:

- [Whole-model parity supplement](../../harness/n01-parity-supplement.json): one matched actual8192-byte QuEST CPU/MPS loss/gradient pair; H06 logical-average gradients at unchanged1e-4, raw SUM discrepancies retained separately; two distinct charged query passes.
- [Storage-only canonicalization](../../harness/n01-storage-supplement.json): after completed recipe/all-owner exact PASS, independently authenticate entire decompressed payload equality before listing and disposing only redundant N01-owned U1/R copies; retain every complete same-stepU0, all contexts/checkpoints/arm metadata and failure records; preserve original compressed identities as historical aliases. Same2GiB cap/free1GiB floor.
- [Administrative-only recovery v3](../../harness/n01-administrative-recovery-v3.json): preserve the hold-induced original controller timeout and unavailable OS-exit/stdout facts; authenticate complete original QuESTU0 before launching only original remaining U1/R children. Earlier helper/design preflight failures remain versioned, and original/recovery/hold elapsed segments are charged once.
- [Corrected capacity recovery](../../harness/n01-capacity-recovery.json): the original capacity wrapper FAILED and lost search timings/selection/exact counts. Root replaced the unfulfilled sustained/search deliverable with one warmup plus six single-step recipe×[1,64] arms and one query.64 is provisional, no new selection. Lost completed upper55 plus partial upper7 and seven new completed steps imply completed upper62 and expressly revised full-step-equivalent upper69; original64 is not asserted to cover unknown failed attempts. No further margin/probes or thermal/sustained/optimality claims. M07 owns longer/per-method/core remeasurement and scientific budgets.

The [complete implementation report](../../harness/n01-results.md), [authenticated evidence](../../harness/n01-evidence.json) and [source-byte project coverage](../../harness/n01-project-coverage.json) record actual outcomes/costs and downstream limits. The derivative,72-step exact replay and composition parity passed; independent N01 review is still required before M01. All12 historical native-v4 H08 scopes remain FAILED.
