# E03 frozen confirmation and protected access

Research cutoff remains 3 October 2026. The new permanent evidence path uses the
actual local 12 October date, `e03-20261012`; receipts use automatic UTC.

`ConfirmationManifest` lives in `ralup.config`; `ralup.evaluate` exports
`freeze_confirmation(inputs) -> ArtifactRef`,
`validate_confirmation(reference, inventory) -> ConfirmationManifest`, and
`open_protected_targets(*, phase, manifest_ref, inventory) -> Iterable[TargetBlock]`.
The typed errors `IntegrityError`, `IncompleteConfirmation` and
`ProtectedAccessError` live in `ralup.records` and are exported by evaluate.

## Actual scientific gate

```sh
python -m ralup plan-confirmation --freeze --design /Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/e02-20261012/plan-confirmation-v1/confirmation-plan.json
```

This command refuses before publishing a manifest. It authenticates the indexed
original E02 startup, finish, output and complete source archive, then re-derives
the original E01 prerequisites through the existing E02 boundary. Those historical
producer bytes establish the authentic current refusal; they do not establish
current-source complete-state compatibility.

Missing evidence is named: accepted robust B0/recipe applicability (WP04
inconclusive), finalized trained/development-selected C and learned LR (WP06
NEEDS_REVISION), selected B*, actual independent four-unit final-arm development
pilot with full outcomes/failure/cost receipts, adequate precision/tail/resource
evidence and locked n, and final immutable seed/arm/config/budget/checkpoint
inventory. All sixteen unresolved families remain in the original E02 artifact;
the CLI does not manufacture finalists or terminal checkpoints.

A synthetic E02 engineering design never becomes usable scientific authority,
even if a caller changes its eligibility or n flags. The current producer offers
no positive scientific schema. Genuine future authority remains E01/reopened
WP06, then E02, followed by integration of those authenticated final artifacts.
An exploratory development pilot may precede a manifest/eligibility flag; it
still needs genuine frozen arms and paired independent lineages, final development
outcomes and complete receipts. Provisional arm variance remains diagnostic only.
No new training wave is prescribed by E03.

## Synthetic contract for engineering verification

Positive E03 fixtures are explicitly `ralup.synthetic_confirmation.v1`, with
`scope=synthetic-engineering` and `confirmation_eligible=false`. They permit only
`partition=fixture` and a named `synthetic-protected.json` payload of schema
`ralup.synthetic_protected.v1`. They cannot authorize an official-test partition.
Source-authentic scientific production integration is deliberately still missing,
not approximated by synthetic markers or callers' claims.

All S34 fields are present: claim; dataset manifest and payload SHA-256 refs;
model, quantizer and backbone refs; policy bytes/hash ref; normalizer ref;
B* selection and learned-LR refs; ordered seeds/arms and randomized per-seed arm
permutations; endpoint/target budget; n; analysis bytes/hash ref; the two-comparison
Bonferroni rule (97.5% two-sided intervals, both upper bounds below zero); substantive
divergence/no replacement/no optional seed rule; allowed cold/warm timing variants;
and all resolved arm config hashes. The contract also binds all current
`src/ralup/**/*.py`, pinned lock and archived environment identities, actual
Python/Torch/machine/OS build/fallback identity, automatic creation UTC and cutoff.
Frozen objects own immutable nested maps/tuples. Canonical UTF-8 JSON uses sorted
keys and fixed separators, rejects nonfinite values, and hashes exact bytes.
Manifests have no alternate whitespace/newline encoding. Publication is exclusive
and fsynced; an existing output/history cannot be reused or overwritten.

`inventory` must be the complete list of `{seed, arm, terminal: {path, sha256}}`
references. No readiness boolean is accepted. Every terminal receipt binds the
manifest, exact seed/arm, resolved config hash, frozen endpoint, source/runtime,
independent-unit/shared-parent lineage, cost receipt, status, checkpoint and
failure. Within-seed parents match; different seeds cannot reuse units/parents.
Completed synthetic arms require exact final checkpoint metadata/hash with the
same seed/arm/config/endpoint/source/runtime. Divergence is a declared substantive
terminal failure with a retained failure/cost receipt and no invented final
checkpoint. Missing, duplicated, extra, running or changed arms deny access.
These are software fixtures, not evidence of scientific seed independence.

## One-way access and audit

Phase, full immutable contract/dependencies, source/runtime and complete terminal
inventory are checked before any protected payload read. The payload itself is
hashed from the exact parsed bytes after admission and before target construction;
a changed payload produces a retained failure with no retry. Metadata checks
cannot verify payload bytes without reading them. H01/H02 reconstruction audits
remain separate, non-scoring interfaces and confer no scorer permission.

A hash-linked append-only `.access.jsonl` records the original freeze, denied
openings, single scorer reservation, payload failures, amendments and result refs.
Sequence, previous event hash, manifest hash and exact canonical event bytes are
validated on every access; events have distinct IDs and automatic UTC. A local
advisory writer lock serializes check/append. The first scorer reservation is
retained even if reading fails. Repeated opening fails; it never silently replaces
a result. `record_protected_result(ref, result_reference_dict)` permits one result
only after successful opening, authenticating its manifest binding;
`existing_protected_result(ref)` returns the existing verified ArtifactRef or
fails on changed result bytes. E04 owns actual execution/scoring/inference.

`record_confirmation_amendment(ref, amendment)` preserves amendments separately
from the frozen manifest. Before inspection it requires a new freeze; after
inspection changed claims are labeled exploratory and require fresh independent
confirmation. Amended freezes cannot activate/restart the scorer.

This is procedural enforcement for a local owner, not OS security. An owner can
rewrite files or bypass Python code; hash linkage is not a signature, external
trusted anchor or filesystem quota. It does not turn public-corpus reuse into a
fresh private dataset or remove that scientific limitation.

## Evidence and storage

`docs/harness/e03-attempt.py` follows the accepted corrected E02 recorder: it
captures exact source/tests/helper/config/lock/environment bytes, runtime/diff,
and complete store inventory; converges the encoded startup's own byte counts;
and admits their full sum plus declared bounded child/finish output allowances
BEFORE creating the attempt directory or importing the child. The default child
allowance is 16 MiB, finish 8 MiB. The complete forecast is computed, not a naive
16 MiB full-artifact allowance. The separate source/Git transaction allowance
remains 16 MiB with 512 MiB retained. Whole-store cap is 32 GiB and Data free
reserve 64 GiB before and after writes, including a post-finish store audit.
Overruns retain evidence and fail status; this is a scoped forecast/audit for
bounded commands, not preemption of arbitrary writers.

Raw RED/GREEN, affected CPU checks and standalone actual CLI refusal are indexed
in `e03-evidence.json`. No actual WikiText protected payload is opened or scored,
no scientific numerical/native wave is repeated, and no dependency is upgraded.
Unknown engineering work remains unknown; synthetic exposures are not scientific
or globally unique corpus counts. E07 owns total project accounting/migration.
