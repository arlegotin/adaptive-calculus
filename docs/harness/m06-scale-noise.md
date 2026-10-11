# M06 learned scale, damping and matrix masking contract

This is an independently written equation implementation and bounded engineering
comparison. No protected outcomes, tuning winner, pilot relevance or scientific
eligibility is established. Numerical and operation blocks are not defeated
baselines. Registry statuses remain unverified by default; fixture passes do not
silently promote the deployed adaptation.

`learned_scale_view(W, log_s, contract)` returns physical `s*C(W/s)` in identity
storage coordinates. The uniform fixture uses integer nearest-even levels;
the FP4 adaptation uses the canonical E2M1 levels and tie rule. The scale
surrogate is `C-W/s` strictly inside the declared code range and the saturated
code at/beyond either endpoint. Its group reduction is multiplied once by
`1/sqrt(N*Cmax)` and then by `s` for the log chain. This analytic surrogate is
the target; finite differences of the hard forward are deliberately rejected.
Clipped STE has zero W derivative at/beyond the bounds; full STE is a separately
configurable view convention.

The deployed learned adapter supports the full FP4 range `[-6,6]`, row or whole
tensor groups, identity storage, an independently configurable scale Adam LR,
betas/epsilon, gradient clipping and bounded log scale. Restricted code ranges
are standalone equation fixtures only. Their export/deployment is unsupported.
The default initialization is max-absolute/Cmax with zero groups initialized to
one, followed by declared log-range clipping; this departs from LSQ's
mean-absolute initialization. Matching arms consume the actual exported FP32
scales, full E2M1 recipe and training-only information contract. The task test
binds a fixed arm to these actual scales and checks identical deterministic
contexts and scores. There is no scale reset on final export.

The adapter owns real trainable log-scale Parameters, their gradients, every
scale Adam moment/CPU counter, normalizer, pending logical target count,
detached physical-scale cache, finite producing backend (`cpu` or `mps`), and
validation-cost telemetry. Original-W Parameters retain object identity and
their backbone moments remain in original W. Fresh functional views override
exactly the six projections in the existing decoder; the shared byte scorer,
W4A32 activation arithmetic and target masks are unchanged. The common H06
SUM backward charges ordinary passes once. H06 divides W gradients once;
the method divides scale gradients once by the same logical target count and
clips them independently. Scale and damping work launch no additional data
forward/backward pass, and report their actual optimizer/penalty/state work.

The completed context records the scales used by that logical forward; runtime,
quantizer and exported scales record the newly committed physical scales. H07
allows only this declared learned-owner difference. CPU artifact validation
checks finite arrays, geometry/range/config and exact physical cache/runtime/
quantizer ownership. At live capture and same-backend restore it additionally
checks `exp(log_s)==physical_cache` bitwise on the declared producing backend
before exposing usable state/reference. Cross-backend restore fails closed.
Existing strict source/lock/F02 identity checks remain mandatory. Validation
checks, bytes and seconds are retained as engineering telemetry and excluded
from scientific replay hashing, like existing process/wall costs. Rollback
rebinds the original scale Parameter objects and their stored gradients.
Successful transactions also bind the complete post-context method-owned
Parameter inventory and identities before backward/commit. First-step scale
construction remains inside the transaction; pre-existing owners must survive
preparation, and replacement, removal or addition after binding rejects the
step before publication and follows the same rollback and charged-work path.

The first actual MPS attempt failed because CPU validation recomputed exp and
demanded equality to MPS exp. That failure remains immutable. An exp-only
diagnostic of 1,024 explicitly specified inputs in the default range observed
a maximum difference of two ULP; this is observed variation, not a universal
bound. No numeric/replay tolerance was changed. The physical-cache correction
implements the root's backend-owner ruling instead of replacing producer bytes
with a CPU exponent result. Failed neural operands were not retained by the
original gate and remain unavailable; they were not reconstructed.

`damping_penalty(W,Q,lambda)` is `lambda/2*||W-stopgrad(Q)||²`, an S24 project
adaptation absent from the masking paper. It shares the fixed FP4 grid and adds
one complete logical penalty irrespective of packing, in the same backward
graph. Simple masking accepts a matrix with probability .5 and multiplies its
gradient displacement by two. Practical Magma always uses probability .5 and
multiplies accepted displacement by alignment without doubling. Alignment uses
the **current** dense first moment and gradient, with `.9/.1` EMA of
`sigmoid(cos/temperature)`, and advances even when a matrix is rejected. The
dedicated CPU mask generator/state is completely owned. Decay is applied
separately at ordinary LR; every base moment advances on both decisions.
Zero-vector cosine zero, initial alignment .5, rescaled finite norm arithmetic,
matrix projection scope and gradient-only decay separation are project choices.
The paper's differently normalized theoretical rule is not substituted.

Adapters support explicit original-W AdamW, Muon and Muon+ backbones with their
no-decay auxiliary groups. SOAP, Madam, arbitrary rules and storage rotations
are unsupported compositions requiring later owned evidence. This is not a
restriction on future B0 selection.

Primary sources and rights:

- [LSQ, arXiv:1902.08153v3](https://arxiv.org/html/1902.08153v3),
  2020-05-07, equations 1–3/5 and section 2.2. The article's arXiv distribution
  license supplies no author software reuse grant. Product equations are
  independently implemented; no author software was copied. Log-scale Adam,
  FP4/grouping, initialization and byte-decoder integration are named departures.
- [SkipUpdate/Magma, arXiv:2602.15322v1](https://arxiv.org/html/2602.15322v1),
  2026-02-17, practical Algorithm 1/section 3. Primary audit HTML SHA-256
  `ae119060283877a05e1a05b46be45ab916eaa7cf6f3a637764af07ecb9294744`;
  authenticated CC BY 4.0 legal-code SHA-256
  `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411`.
  The licensed article is attributed and modifications are disclosed above;
  no code grant or release parity is inferred. The authoritative primary
  extraction remains `.superpowers/sdd/2026-10-03-ralup/source-audit/masking/`
  (`report.md`, `evidence.json`, `equations.json`). No article/code bytes were
  vendored into product modules.

Public methods are `lsq_uniform_fixture`, `learned_fp4_scale`, `damping_fp4`,
`simple_matrix_mask`, and `magma_matrix_mask`. The common CLI binds the four
deployable controls with explicit adaptation labels; `verify-baselines --suite
scale-noise --device cpu|mps --output PATH` publishes canonical JSON lists and
scoped analytic/mask/state/operation results. The gate owns only a 1x8x2
decoder, eight synthetic targets, two logical steps and enumerated packing.
Actual MPS requires fallback zero before startup. Complete-state equality is
same-device evidence; neither a universal MPS capacity/replay assertion nor
CPU/MPS bitwise parity follows.

All attempts, including RED, mutants, the MPS failure and its correction,
retain startup source/test/helper/lock/F02 bytes, native streams, status and
scoped complete-process costs in NEW permanent Data paths. Costs for loss and
backward classifications can overlap the same execution; they are not summed
as independent passes. Scale/mask wall measurements include the adapter's
backbone update and overlap whole-step clocks. Cache bytes are explicit; unknown
enclosing/editing/tool costs remain unknown. Source/config/Git receipts are
small guarded transactions; old evidence and frozen artifact gates are unchanged.
