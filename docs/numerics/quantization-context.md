# Project quantization contexts and numerical provenance

This independently written portable implementation follows S07 and S13–S15 of
the [system design](../superpowers/specs/2026-10-03-ralup-design.md). It implements
simulated E2M1 W4A32. All weights, arithmetic, gradients, scales, cached forward
values and masks are FP32. Discrete ordinals are uint8 and flags are boolean.
Optimizer moments stay in original W coordinates; these modules do not own or
modify optimizer state.

`to_storage(W, rotation)` returns U=W@R. `from_storage(dU, rotation)` returns
dU@R.T. Rotation IDs are `identity` and `full_hadamard`; the latter is the complete
last axis of positive power-of-two width, with orthonormal normalization
1/sqrt(width). `quantized_view(W, context)` returns **storage Q(U)**. A linear
caller must compute `(X@R)@quantized_view(W, context).T`. The fresh autograd
transform applies R.T in backward. Reusing a context across separately constructed
microbatch graphs needs no retained graph.

`prepare_context(W, recipe, step, rng)` owns cloned detached U, scales, exact
forward ordinals and reconstructed values, frozen gradient masks, rotation,
grouping, backward IDs and logical step metadata. The caller prepares it once
per logical step and keeps it through all physical microbatches. No hidden global
step cache or history exists. Mutating W after preparation leaves the numerical
forward frozen until explicit next-step preparation. Context tensors are owned
snapshots and should be treated as read-only. Frozen dataclass/metadata prevent
field rebinding; PyTorch does not enforce tensor immutability.

Runtime recipe fields are `name`, `rotation`, `backward`, `warm_start`,
`previous_scales`, `previous_zero_groups` and `codes`. Tensor-bearing runtime
recipes are state handoffs; they are not canonical scientific JSON configuration.
`row_fixed_max` and `tensor_fixed_max` initialize `max(abs(U))/6` from explicit
warm-start W, or the first supplied W. The caller owns this lifetime and passes
`previous_scales` thereafter; scales are shape/device checked and cloned.
`previous_zero_groups` retains scale-initialization zero flags separately from
current weight zero flags. Without those flags, initialization provenance is
explicitly unknown (`null`), never inferred from current W. Warm start and
previous scales cannot both be supplied. Dynamic recipes recompute from current
W on explicit context preparation and reject previous scales. A final evaluation
context uses the same pure operation without history or RNG advance.

Row grouping is the last axis with `[out,1]` scales. Whole-tensor sensitivity
recipes have `[1,1]` scales. `quest_fp4_row_reference` and
`quest_fp4_tensor_reference` fit the Gaussian codebook and use s=a*RMS(group)/6;
trust is the default backward. RMS divides by the group maximum before squaring,
avoiding needless overflow/underflow for inputs such as 1e30 or 1e-30. Automatic initialization/dynamic scales assign true zero
groups positive sentinel s=1 with an explicit zero flag. Reused fixed scales
remain frozen even if a current row becomes zero. Integer
FP32 bit tests preserve nonzero classification on MPS. Representable positive
subnormal scales, weights, transform outputs and incoming gradients are supported
with device integer bits and normal-range FP32 operations. `fp32.py` provides
exact mantissa/exponent decomposition, nearest-even power-of-two reconstruction,
exact rational rounding for tiny x/6 and x/sqrt(2), and preserved FP32 subtraction
and products. Tiny division/product rounding is performed once with integer
significands; rounding a normal intermediate into a subnormal can otherwise
introduce a second-rounding error. Row maxima/scale positivity are tested by bits.
Stable RMS normalizes tiny inputs through their decoded significands; Gaussian
scales always use preserved product arithmetic. Group sparsity can make a scale
subnormal even when its maximum is normal and above any fixed magnitude cutoff;
preserving every Gaussian product avoids such input-based assumptions. Sparse
256-value row and whole-tensor groups with maximum bits 0x2000000 both retain
scale bits 2042958 on CPU/MPS.

Hadamard keeps the ordinary normalized butterfly unchanged. Near-zero butterfly
pairs use the gradual-underflow helpers; custom autograd applies the same
orthonormal inverse to incoming gradients, including subnormals. Trust compares
preserved physical abs(Q(U)-U) bits against scale bits. Clipped STE retains the
FP32 abs(U/s)<=6 rule using normal significand division. Masked backward selects
integer gradient bits to preserve tiny incoming gradients. All tensor operations
remain on the requested device with FP32 arithmetic and integer tensors; there
are no CPU tensor copies, higher precision production tensors or fallback.
A nonzero group's computed scale that rounds to zero is unrepresentable and
raises explicitly rather than being mislabeled all-zero. Nonfinite input,
nonfinite transform output and genuinely overflowing chosen reconstructions also
raise. Independently integrated Gaussian fitting remains a CPU float64 oracle.

S14's analytic normal moments use midpoint cells with infinite exterior tails:
M0=Phi(u)-Phi(l), M1=phi(l)-phi(u),
M2=M0+l*phi(l)-u*phi(u), with infinite products zero. Positive-tail intervals
use survival-CDF differences. `fit_gaussian_e2m1()` expands a strict three-point
bracket from a=(.5,1,2), minimizes the sum of cell squared errors in log(a) with
Brent, and refines the stationary point within that bracket using the analytic
derivative. It independently integrates each squared-error cell with deterministic
QUADPACK and records residuals and conventions. A cached scalar fit constant
avoids fitting per logical step; it contains no parameter state.

The [independent CPU float64 golden audit](gaussian-e2m1-reference.json) was
produced before implementation using direct quadrature minimization and an
independent analytic root check. Its stationary a≈2.922475071373709 and
MSE≈0.01268490413871003 are verification values, never the fitting algorithm.
Flat-objective minimizer positions can differ by a few float64 ulps or about 1e-8;
objective residuals are a more stable numerical measure. The original audit is
`.superpowers/sdd/2026-10-03-ralup/numerics-reference/numerics-report.md`.

Plain `ste` passes the incoming storage gradient. `trust` multiplies it by
`abs(Q(U)-U)<=s`, the physical project half-gap threshold. Separately named
`clipped_ste` uses `abs(U/s)<=6`. Masks are frozen at context preparation.
A custom autograd function returns exact cached values without the cancellation
error of adding/subtracting a straight-through residual. Supplying `codes`
accepts one externally generated realization and owns a detached copy. The
supplied branch validates actual weights/scales and reconstructs only the chosen
codes; it never reconstructs an unused nearest result. Thus a finite supplied
endpoint remains valid even when a different nearest endpoint would overflow.
Nonzero supplied ordinals in an all-zero group are rejected to preserve its
zero forward contract. Drawing
stochastic realizations is reserved for M05; F04 never consumes `rng`.

## QuEST release distinction

Primary mathematics: [QuEST v2 §§3.1–3.2](https://arxiv.org/html/2502.05003v2).
Inspected source: [base_linear.py at c295afda142c03a89d2af04653881beccf459aa1](https://github.com/IST-DASLab/QuEST/blob/c295afda142c03a89d2af04653881beccf459aa1/src/models/quantization/base_linear.py).
The saved source audit is
`.superpowers/sdd/2026-10-03-ralup/source-audit/baseline-sources.md`.
Training quantizers use last-axis row/token RMS; the inspected CUDA scaling
kernel also groups per row. That grouping is resolved for those paths. The release
uses block-diagonal 128-point Hadamard; this project uses full-row Hadamard.
The release hardcodes a=2.92247856, uses duplicate zero levels, RMS epsilon and
first-index ties. The project instead fits unique E2M1 levels and uses S13 parity.

The release `FP4TrustQuantizer` replaces Gaussian Lloyd levels without recomputing
`self.trust`; half/Hadamard subclasses likewise compute a local `trust` without
assigning it. Thus its default inherited Gaussian threshold is about .31857661
RMS, distinct from the FP4 half-gap and this project's physical s. These project
recipes are explicitly labeled `project_S14_refit`; they are not release
reproduction. Source-fixture parity is reserved for M07. No inspected upstream
code or restricted kernel is copied into the implementation.

`check-numerics --suite quantizer` runs actual F03 and F04 checks on the requested
device, with golden objective/integration comparisons, involution and gradients,
fixed/dynamic/zero scales, trust/clipped/STE gradients, detached caches and two
independent backward calls. Bounds are frozen in the checker before device
measurements. Reports identify grouping and coordinates. Success covers these
implemented F03/F04 checks only. The default `check-numerics` additionally runs
F05 assistance and mechanism invariants; see [actions and measured bounds](actions.md). Actual MPS requires fallback disabled before interpreter startup; a
sandbox-blocked report cannot stand in for an MPS pass.
