# Publication preparation

This checkout prepares RALUP version 0.1.0 for public reuse. No push, publication, repository-visibility change or package upload has occurred as part of this preparation. The project repository URL and citation metadata identify the intended source location; they are not evidence of publication.

## Fresh-checkout verification

Run from the repository root in a new isolated environment. Follow [public development](public-development.md) for the base install, platform-specific CPU numerical/test dependencies, `pip check`, the portable CPU pytest suite, numerical checks and harness checks. Base installation supports the standard-library configuration CLI without numerical dependencies. Linux numerical installation selects CPU PyTorch wheels explicitly. The dependency bounds describe compatible API families; they do not claim validation of every permitted dependency combination or every platform.

The public suite uses small independently authored byte fixtures and defaults to excluding opt-in historical local-evidence cases. Its CPU switch explicitly excludes native execution. Fresh test setup may create a real environment discovery record with engineering-test scope; production source/runtime authentication and protected-scoring gates remain enforced. A local discovery report does not become an archived operation gate or scientific proof.

For packaging review, build without altering an archived environment:

```sh
python -m pip install build
python -m build --wheel --outdir artifacts/public-development/dist
```

Inspect the resulting wheel's `dist-info/METADATA` and `dist-info/licenses/`: they must include the mixed-material SPDX expression, both project license documents, project notices and the full SOAP MIT notice. Numerical dependency wheels and corpus assets are not bundled. The [licensing record](../LICENSING.md) gives exact copied-license hashes; [third-party notices](../THIRD_PARTY_NOTICES.md) gives the rights boundary. A successful build is packaging verification, not legal clearance.

## Scientific limitations

RALUP has no demonstrated improvement. The two provisional one-step ES proposals share one proxy META task; they improved shorter endpoints but worsened the remaining 496-step endpoint by +0.006914/+0.013553 bits per byte against the initialized policy. They are development diagnostics, not independent meta-training replications or confirmation. WP04 remains inconclusive and WP06 needs revision. Robust B0, final candidate/learned-LR/strongest-competitor selection, final scientific budgets, a fresh paired pilot, justified confirmation n and a locked manifest remain absent. Official test has not been scored.

Forward E2M1 FP4 is simulated with FP32 shadows, arithmetic and optimizer state; gradients use ordinary backpropagation. No native low-bit performance or memory benefit, large-model generalization, or scientific efficiency advantage is established. See [current findings](harness/current-research-status.md) and the [scientific design](superpowers/specs/2026-10-03-ralup-design.md).

## Historical records and archive

The [engineering handoff](harness/engineering-handoff.md) preserves the previous root README's bytes after a short archive note. Links and commands inside the original text retain repository-root context, so they are historical references rather than relocated instructions. Use the current public development guide for working fresh-checkout commands.

Tracked evidence indexes and the historical lock preserve exact hashes and local path scope. Paths to retained environments, corpora, immutable evidence roots and the detached execution worktree require their original local layout; a fresh clone does not provide those assets. No bulk relocation or rewrite of those identity records occurred. The historical `requirements.lock` is pip's platform-specific hash-locked input, not JSON or a portable installation recipe. Historical source identities, runtime compatibility refusals and native/resource failures keep their original scope.

## Third-party scope and attribution

The copied SOAP reference retains its full upstream MIT notice. Independent equation implementations do not imply permission to copy restricted upstream source or establish release reproduction. Dependency notices are scoped to their recorded versions. WikiText's unresolved supplied-license discrepancy and absent article histories remain recorded; no corpus is redistributed and no legal clearance is claimed. Existing [asset rights](../licenses/ASSETS.md) and [vendor records](../vendor_manifest.json) remain unchanged.

Original code uses AGPL-3.0-only and original writing uses CC BY 4.0; external material retains its own terms. Use [CITATION.cff](../CITATION.cff) and identify the exact source commit for scholarly attribution. Citation requests for independent reuse of ideas or facts add no supplemental copyright-license conditions.
