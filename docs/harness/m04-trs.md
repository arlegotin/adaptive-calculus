# M04 TRS mechanics and logical-step adaptation

The two registered identities are `trs_release_fixture` /
`released_multiplicative_coupled_Adam_clean_equations` and
`trs_backbone_adapter` / `project_FP4_logical_step_decoupled_backbone`.
These are independently implemented equations. No GPL author source was copied
or executed, and bounded mechanics parity does not establish author-release
reproduction, redistribution permission, pilot relevance or scientific quality.
Pinned source revision is `cvlab-yonsei/TRS@958dedd1baeb60814cb4162ce794bc9f41cdf4a0`;
file inspection hashes and permanent release/paper/scale metadata are in
`baselines.registry.trs_record` and the authenticated verification reports.

Release initializes `r=1`, `K=first target_rate`, and previous normalized values
at zero. The first observed transition is ignored and the first effective LR is
alpha. Later it computes `K=mu*K+(1-mu)*k`, then
`r=mu*r+(1-mu)*r*R/(K+1e-8)`, and effective LR `alpha*r`.
The disabled `mu=0` source branch instead uses the ordinary source group LR,
which its training script uses for the target rate. The paper v4 §3 Eq.11 uses
additive `max(0,U+eta*(R-K))`; that is a different comparator and was not silently
substituted. Fixture defaults are declared release/development conventions,
never tuning winners.

The small release fixture has FP32 latent weights/Adam moments, standard bias
corrections and coupled decay before moments. Its fixed scalar scale is
`3*sample_std(initial W)`. For declared nonbinary bit widths, normalized values
are `round(clamp(W/(abs(sW)+1e-6)*2**(b-1),Qn,Qp))/2**(b-1)` with declared
symmetry. The one-bit transform, trainable activation scale, CNN/data pipeline,
AMSGrad and missing-transition ordinary-parameter branch are outside this
bounded fixture. Scalar feedback is the required Python-float interface
(binary64 host arithmetic); the inspected release stores FP32 scalar tensors.
This is equation/fixture parity within the recorded bounds, not bitwise author
software reproduction.

The FP4 adapter uses the shared quantization context's
`storage_values / cached_scales / 6`, including its declared rotation. It compares
numerical normalized levels, so physical scale multiplication and duplicate
signed-zero encoding IDs do not invent transitions. This named adaptation
requires the fixed row-max recipe; it does not inherit the release's tensor
sample-std scale or its uniform levels. It measures the pre-update cached
quantized values, matching the forward-before-update timing. A single prepared
logical context covers every microbatch, with initial previous values zero.

Proposal collection owns candidates and leaves committed feedback/history
unchanged. `finish_step` publishes feedback exactly once together with the
original-W backbone update. H06 remains the shared loss-EMA/code-history,
cursor/schedule/ledger transaction boundary and rolls back all method fields on
any later failure. Evaluation invokes no feedback hook and preserves the full
scientific hash. Target schedules (`constant` or `linear` with explicit
`start/end`) are a separately registered configuration owner from the ordinary
`SchedulePoint.lr`. Linear endpoints return their declared values exactly.
Decay uses ordinary LR on the pre-displacement reference. Feedback changes only
the gradient displacement. No target rate is substituted for the backbone's
ordinary LR, except the explicitly disabled release-feedback branch described
above.

Supported backbones are explicit trusted original-W AdamW, project FP32 Muon,
and paper-equation FP32 Muon+. Configuration includes a complete named backbone
envelope; auxiliary parameters use ordinary no-decay AdamW and matrix backbones'
separate auxiliary LR. SOAP, Madam and arbitrary unregistered `BaseUpdateRule`
compositions are unsupported. This does not restrict future B0 selection:
selecting an unsupported backbone requires later owned composition/evidence or
an honestly narrowed comparator claim. Unsupported/licensing/operation/parity
blocks are never defeated baselines.

Trusted snapshot registrations are `trs_state.v1` and `trs_adapter.v1` under the
existing `ralup.training_snapshot.v2`. They preserve every scalar/code count,
complete backbone/auxiliary optimizer moments, pending-code/proposal inventory
(which must be empty at capture), config and existing shared owners. They check
feedback geometry/count/context, actual wrapped-rule config and exact ordered
projection groups before restoration. Unknown/executable/nonfinite owners and
missing feedback states fail closed. Constructor selection remains a trusted
local table and strict current source/lock/F02 compatibility remains unchanged;
no historical full-state migration is inferred. Public config/metadata sequences
are JSON lists; the canonical validator was not relaxed.

CLI supports `verify-baselines --suite trs --device cpu|mps` and the existing
bounded `run --phase explore` envelope with method name
`trs_backbone_adapter`, matching adaptation label, complete TRS config, and
ordinary `learning_rate`. The adapter's `FIXTURE_CONFIG` is an explicit example
of the complete config. This does not resolve null scientific budgets or
protected/science gates.

The single scoped covering run reports **165 passed in 11.10s**, no skips,
warnings or failed tests, including the new actual-MPS accumulation/resume and
fresh-interpreter CPU trusted reconstruction. Its original recording helper
exited **1**, because it misclassified pytest's successful IntEnum status. That
receipt remains unchanged. A corrected-helper-only IntEnum regression exited
0; both requested CPU and actual-MPS CLI runs subsequently exited 0 with status
pass and no blockers. No redundant covering rerun was made.

The additive-feedback mutant fails at `r=.99996` versus the released
`.9996153836908285`; the per-microbatch mutant fails observation counts 5 versus
2. A separate public-Muon-config RED caught tuple betas; converting sequences at
the TRS public boundary fixes canonical metadata and config reconstruction.
The covering/CLI product and test hashes agree exactly. Weight fixture bounds
are `atol=5e-7, rtol=1e-6`; scalar mathematical fixture `rtol=1e-12`;
packing parameter bounds `atol=3e-7, rtol=3e-5`, with exact feedback/counts and
same-device resume hashes. Observed packing max errors are
`1.0477378964424133e-08` CPU and `1.4901161193847656e-08` MPS. This is a 1-layer,
width-8, 2-head, 16-target synthetic scope; no full-family, real-corpus, target
model/horizon, large packing, CPU/MPS bitwise or universal deterministic claim.

[Evidence pointers and exact scope/cost/inventory](m04-evidence.json) authenticate
all original receipts. Bulk assets remain in the NEW permanent task store
`/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m04-20261004`.
Every new attempt retains pre-start helper/product/test/lock/F02 bytes, stdout,
status and complete process costs, including failures and negative controls.
The nine experiment/helper launches total 26.232679416425526 seconds scoped wall;
child wall is already included. The administrative inventory generation has its
own separate receipt. Earlier reading/editing/tool/session/project costs remain
unknown, never zero or a complete project total. No old artifacts were moved,
deleted, overwritten or rehashed into acceptance.
