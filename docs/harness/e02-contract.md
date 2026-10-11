# E02 paired precision design

Research cutoff remains 3 October 2026. Receipts use actual automatic UTC and
local observations; the task directory is `e02-20261012` (local date).

## Statistical API

`paired_sd_upper(deltas, coverage)` uses the one-sided normal-theory bound
`s * sqrt((m-1)/chi2.ppf(1-coverage,m-1))`. Coverage is configurable and recorded.
Fewer than four complete finite differences, or zero/nonfinite sample SD, raises
`InadequatePilot`; there is no zero-variance small-n shortcut. The two contrasts
are candidate minus baseline and candidate minus equally informed learned LR.

`confirmation_n(sd_upper, delta_design, power=.95, interval_confidence=.975)`
finds the smallest integer n >= 2 whose noncentral-t probability reaches the
requested power. The success event is a negative true effect's interval upper
bound below zero, using t(.9875), not significance in the wrong direction. The
normal approximation starts with z(.9875)+z(.95). Each contrast has .95 power;
the plan uses their larger n, giving at least .90 joint power by union bound.
The SD upper bound's normal assumption and estimation uncertainty remain explicit.
A numerical search cap is an API range limitation, never resource impossibility.

`precision_curve` emits one row per prospective detectable BPB difference:
normal approximation and noncentral-t counts for both contrasts, maximum n,
actual attained powers, conditional expected-scale CI half-width forecasts,
paired-run seconds/storage forecasts and declared-envelope status. The CI width
is conditional on the upper SD, not a probabilistic width assurance. Unknown
costs/resources stay null. A scoped envelope exceedance does not establish
laptop runtime, global unaffordability or a scientific success threshold.

`pilot_diagnostics` preserves ordered differences, SD/upper SD, leave-one-out SD
range and ratio, leave-one-out mean sign/rank changes, maximum variance share,
and SD/scaled-MAD tail sensitivity. Thresholds are recorded (default LOO ratio2,
variance share.8, SD/MAD3). Four units cannot certify normality. Flags or missing
explicit tail review prevent a fixed engineering design; they call for a
prospectively justified development extension/review, never optional additions
after protected scoring.

## Planner and CLI

`plan_confirmation(pilot_runs: list[ArtifactRef], costs, precision_choices,
assumptions) -> ArtifactRef` authenticates input bytes and publishes a new,
fsynced, immutable canonical JSON artifact. Output reuse rejects. The CLI is:

```sh
python -m ralup plan-confirmation --pilot configs/e02-pilot-prerequisites.json --resources configs/e02-resources.json
```

The pilot envelope has `pilot_runs` references and `assumptions`. The resource
envelope has `costs`, `precision_choices`, `resources` and `output`. Inputs are
hashed into the resulting report. Success exit 0 means the disposition was
published; inspect `status` for the scientific outcome.

Current `scientific-development` inputs require the actual indexed E01
selection. Its original startup/finish, complete source archive, output bytes
and source identity are authenticated, then the indexed M07/M11 prerequisites
are re-derived and compared. Historical producer identity is retained, never
silently promoted to current-source model restore authority. The actual result
is `prerequisite_unresolved`: WP04 inconclusive, B0/B*/C/LR null, WP06
NEEDS_REVISION, zero actual paired units, no measured variance/curves or locked
n. All sixteen unresolved families and named advancement owners remain visible.
The actual resource envelope and precision choice are null; no budget is invented.

`synthetic-engineering` uses explicitly labeled
`ralup.engineering_paired_pilot.v1` records with seed, status and three finite
final development BPB fields (`candidate`, `baseline`, `learned_lr`). It requires
a complete ordered pilot seed inventory disjoint from tuning and proposed
confirmation seeds, upper-SD coverage, bootstrap seed, prospective target delta,
resource envelope, cost boundary and decision rationale. Its outputs are
`engineering_fixed_design`, `inadequate_pilot` or `insufficient_resources`.
Even the successful fixture has `confirmation_eligible=false`, `locked_n=null`,
and only `engineering_n`. Caller flags or fabricated scientific labels cannot
upgrade engineering data. Failure/missing outcomes preserve every referenced
attempt, disallow finite-subset deletion and prevent n selection.

Sensitivity/failure specifications are present before a future freeze: joint
paired-seed percentile bootstrap (10000 resamples), exact sign test, paired
signed-rank sign-enumeration sensitivity with an explicit large-n permutation
rule, zeros/ties treatment, two-sided .025 comparison alpha, substantive finite
budget divergence, documented implementation invalidation and no replacement or
optional seed addition. These are protocol declarations, not executed sensitivity
results. Four-seed pilot extension needs new prospective rationale and receipts.

## Scientific applicability and costs

An exploratory development pilot may precede a confirmation manifest and does
not require an upstream `confirmation_eligible=true` flag. It still needs real
frozen arm/config/data/budget identities, independent complete paired seed
lineages disjoint from tuning, actual final development scores and all retained
failures/costs. Planned-contrast variance additionally requires same-scope genuinely
trained/development-selected C/LR/comparator; provisional policies yield only
labeled diagnostic variance. A usable locked design also needs accepted robust
B0/recipe applicability and finalized E01 selection authority. No such producer
exists now. E01/reopened WP06 owns that scientific advancement; a future genuine
pilot producer integration remains necessary. This task implements no speculative
positive scientific authority and prescribes no new training wave.

Complete deployment forecasts must include each arm's attributable warm start
and all targets/additional loss/query/gradient/HVP/update/transfer/reset/snapshot
and failure work. Shared-parent project costs are deduplicated separately through
E01/E07 accounting. Unknown historical/engineering work stays unknown. E02 test
launcher times are engineering costs; synthetic fixture differences and assigned
costs are not scientific measurements. No protected scores are accessed.

## Recorder storage correction (E02 review I1)

The original 16 MiB routine-attempt forecast omitted the encoded complete-store
inventory: startup receipts were about 38–40 MB and complete attempts about
40–41 MB. Original receipts remain immutable, with their actual underforecasts
reported in the fix evidence. The observed attempts stayed inside the store/free
reserves, but their original preflight calculation was incomplete.

The recorder now captures exact source/config/helper bytes, runtime observation
and Git diff in memory, serializes the complete startup including inventory, and
converges its self-reported size fields before creating any attempt directory.
Admission includes those exact fixed bytes plus a separately recorded child
output allowance (default 16 MiB, prospectively configurable) and finish-receipt
allowance (8 MiB). An explicit caller forecast below that complete sum rejects
before directory creation; absent one, the computed complete sum is used.
The 16 MiB source/Git transaction reserve is a distinct boundary, not the total
Data artifact allowance.

After the child, actual outputs are checked against their declared allowance;
an overrun preserves the produced objects and marks the attempt unsuccessful.
The finish receipt has a separate pre-write size/reserve check. Complete-store
bytes and both free-space reserves are checked again after finish publication.
These allowances are scoped forecasts for known bounded commands, not a general
filesystem quota or a promise to preempt arbitrary child writes.
