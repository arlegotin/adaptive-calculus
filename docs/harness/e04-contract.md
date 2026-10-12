# E04 paired confirmation execution and inference

The research cutoff is 3 October 2026. New receipts use actual UTC and local
timestamps. This task crossed UTC midnight: its RED attempt is under
`e04-20261011`, and subsequent attempts are under `e04-20261012` in the authorized
permanent Data evidence store. Earlier artifacts and producer limitations remain
unchanged.

## Actual scientific disposition

**WP08: not attempted, prerequisite blocked.** Both required actual commands
exit 2, authenticate the original indexed E02 startup/finish/design/archive and
re-derive its prerequisite disposition through the E03 public authority boundary.
No `artifacts/confirm_manifest.json` exists or is manufactured. WP04 remains
inconclusive/B0 null; WP06 remains NEEDS_REVISION; final C/LR/B*, training budget,
fresh paired pilot and scientific locked n remain unresolved. E02 has zero actual
scientific paired pilot units. Their owning integration is E01/reopened WP06,
then E02. No official-test payload is opened or scored.

```sh
python -m ralup confirm --manifest artifacts/confirm_manifest.json
python -m ralup analyze-confirmation --manifest artifacts/confirm_manifest.json
```

Synthetic numerical execution is deliberately typed engineering evidence with
`confirmation_eligible=false`. It cannot advance those prerequisites, validate
the scientific optimizer/controller implementation, or establish WP08. There
is no automatic real model, seed, budget, controller, or dependency fallback.

## APIs and frozen numerical fixtures

`ralup.runner` exports `run_confirmation(ConfirmationManifest) -> list[ArtifactRef]`,
`terminal_inventory`, `score_confirmation(manifest, inventory) -> ArtifactRef`,
`load_confirmation_manifest`, and `invalidate_confirmation`.
`ralup.analysis.confirm` exports `PairedResults`, its authenticated artifact
readback constructor, and `analyze_confirmation(results, manifest) -> dict`.

The supported synthetic engine is explicit fixed-budget byte-unigram SGD with
256 logits, declared uniform four-bit logit quantization, quantized forward
probabilities and a straight-through logit gradient. Fixture arms use declared
SGD rates; their names C/Bstar/LR are engineering counterparts, not trained RALUP
controllers or evidence that a scientific B* exists. Training text is explicitly
synthetic. Final scoring actually executes the frozen quantized logits through
the common raw-byte SUM-NLL scorer. These are computed scores, not injected
positive outcomes or assumed percentage improvements.

Each fresh seed uses separate named initialization/data/quantizer/timing stream
identities. The timing stream generates frozen randomized arm permutations;
runtime changes to those permutations reject. One numerical warm start per seed
is saved once. Every arm begins from the same bit-for-bit full synthetic weights,
momentum, cursor, schedule, quantizer state, histories and named stream identities.
The tiny deterministic engine consumes randomness only at initialization and
initial byte-schedule offset; continuation has no hidden/global RNG draws.

Every source/lock/environment/runtime identity is validated under the current
strict E03/H07 contract. States do not restore old MODELONLY or historical full
snapshots, drop moments, or treat historical normalizer DATA as new training
authority. No native or actual-model learning claim is made by CPU fixtures.

The endpoint is exact frozen target bytes and continuation steps, never a
best protected checkpoint. Immutable JSON state envelopes retain parent-state,
config and manifest bindings after each step. Interrupted runs resume from the
last saved full state; originals and uncertain partial work remain. Covering
tests compare resumed and uninterrupted weights, momentum, schedule/cursor and
stream identities. Fixed budgets and the complete planned seed inventory remain
unchanged. Divergence receives a substantive failure receipt with costs and no
invented final checkpoint. It is never replaced. Implementation failure amends
the manifest and invalidates every bug-affected arm/unit; explicit adjudication
also retains affected terminal references. Documented versioned reruns need a new
freeze, rather than continuing the consumed manifest until significance.

## One-way delayed scoring

No protected fixture payload is read before all exact final terminal, cost,
full-state and quantized-export identities validate. The existing public E03
gateway reserves one scorer, authenticates the payload and materializes all
blocks, then appends `payload_ready`. Result publication/readback use its public
locked helpers; reservation alone never authorizes a result. Incomplete or
amended inventories reject before payload reading. Pending, failed or interrupted
scoring cannot retry the reservation. A repeated successful score returns the
existing verified result only with the original exact inventory binding.

Scorer interruptions retain completed physical block passes and counts plus
unknown partial work and completed arm runs. Results and analyses are immutable
JSON; result publication is append-only in the E03 audit. An advisory runner lock
serializes execution/scoring. This remains procedural local-owner enforcement,
not OS security or a signature-based trusted artifact store.

## Inference and failure reporting

One fresh complete target-training seed is one independent unit. The planned n
always remains the frozen seed count. Duplicate arms, additional seeds and
reused complete-unit lineage reject. Chunks, per-article metrics and timing
repetitions never change n. Per-seed final BPB, candidate-minus-comparator deltas,
signs and marginal successful BPB are reported.

Each co-primary contrast uses `t.ppf(.9875, n-1)` in
`mean(delta) ± t * sample_sd / sqrt(n)`: a two-sided 97.5% interval per comparison.
Both upper bounds must be below zero for synthetic H1+H2 support. Ratios are
`2**mean(delta)` with transformed endpoints. This normal-theory paired inference
has its usual distributional assumptions; constant differences are mathematics
fixtures, not adequate pilot variance or real confirmatory authority. The result
always separately reports `scientific_confirmation_established=false`.

Missing, nonfinite, divergent or implementation-invalidated arms deny an all-run
finite-BPB claim. Finite-success-conditioned intervals are separately named.
Failure probabilities use the planned seed denominator. The predeclared ordinal
sensitivity ranks divergence worse than finite completion, ties two divergences,
and reports unresolved units. Seeded joint paired percentile bootstrap,
exact binomial signs and exact signed-rank sign enumeration with tied midranks
(seeded plus-one Monte Carlo above 50 nonzero pairs) remain sensitivities. They
never replace the two primary t upper bounds. n<2 cannot produce a paired interval.

## Costs and evidence

Actual synthetic physical loss/backward passes, byte exposures, updates,
quantizer passes, final per-block evaluation, checkpoint/export writes and
wall intervals are retained. Query/gradient-only/HVP/feature-scan counts are zero
because this declared engine does not perform them; scientific feature costs are
not estimated from that absence. Raw classifications do not create duplicate
physical passes. Repeated training exposures union the actual synthetic byte
offsets; they never become global WikiText unique counts. Failed/interrupted work
stays in the ledger. Wall intervals use a shared host clock and union nested or
overlapping intervals rather than summing ancestors.

Shared warm-start cost deduplicates once in the project ledger. Every arm's
deployment view includes attributable warm-start work, all actual online update
work and final evaluation/checkpoint costs. The mean complete deployment boundary
is explicit, with a full-budget availability flag; failed runs retain diagnostic
expenditure without establishing a completed deployment comparison. Actual
quantized export bytes are measured from the written artifact. Full-state export
and quantized export are distinct. RSS/MPS counters, synchronized throughput and
time to an undefined development target are unmeasured and remain null.
Offline search/meta-training and enclosing engineering cost remain separate and
unknown, not zero. Final endpoint quality/cost points do not imply full-project
efficiency or a matched-total/wall scientific positive finding.

`e04-attempt.py` admits the complete encoded store inventory, source archive,
startup/runtime/diff plus bounded child and finish outputs before permanent
creation or child imports. It retains 64 GiB Data free and a 32 GiB complete-store
cap, plus 2 GiB for later E07 environment/scratch and 882 MiB for M01 temporary
work. The current environment occupies about 997 MiB on Git; E07 must still make
its own exact recreation forecast. The small-source Git transaction remains
at most 16 MiB with at least 512 MiB retained. Every claimed attempt has exact
source/runtime, stdout/stderr and immutable finish/cost/storage receipts.
The index is `e04-evidence.json`; the full task report is E04-report.md in the SDD
directory and its immutable administrative copy. No accepted native/Cartesian or
whole suite is repeated solely for packaging.
