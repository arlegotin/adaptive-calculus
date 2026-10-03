# Bounded storage actions and tiny mechanism evidence

F05 implements S17/S18 in `ralup.policy.actions`. All input, shadow and
arithmetic tensors are FP32. `U0` already includes the caller's common decay;
`dA` is the original optimizer update mapped into storage coordinates. Moments
remain original-coordinate caller state. Exponential geometry uses
`a=sign(U0)*dA/max(RMS(U0),1e-30)`,
`c=RMS(dA)/RMS(U0*a)` (zero when that denominator is zero), and
`dM=U0*expm1(clip(c*a,-.1,.1))`. The blend is
`h*((1-alpha)*dA+alpha*dM)`. There is no norm matching after exponentiation or
assistance. Multiplication preserves signs and zeros; the additive branch can
escape zero. This geometry is not Madam or fractional calculus.

Assistance recomputes deterministic `Q(U0)` and `Q(U0+d)` at the **cached**
context scales. It uses ordinal adjacency and the physical midpoint of adjacent
code levels. Nonzero update, in-bound U0 and an existing directional neighbor
are required. Eligibility further requires unchanged code, positive midpoint
distance rho, and `abs(d)>=tau*rho`. The candidate is exactly
`b+sign(d)*min(gap/1024,abs(d)/1024)`. Acceptance requires the immediate neighbor
code, finite format bounds, a representable directional nudge beyond b, and
`abs(candidate-U0)<=(1/tau+1/1024)*abs(d)` within the frozen FP32 bound below.
Rejected or ineligible proposals retain the additive shadow. Tiny nudges that
round to the midpoint are counted, never enlarged. The validated device integer
rescaling helpers preserve representable FP32 subnormal shadows on actual MPS.

At `tau==1`, assistance directly returns `U0+d` without quantization. For
`(alpha,h,tau)=(0,1,1)`, `compose_action` explicitly permits bypass to the
backbone's authoritative **original-coordinate** commit. It guarantees its
storage expression is exactly `U0+dA`; it cannot reconstruct optimizer-specific
rounding from that displacement. H05/M09 own backbone integration and identity
commit/resume checks. Metadata also identifies coordinates, logical context
step, scale lifecycle, baseline crossings, feasibility, eligibility,
accepted/rejected masks and counts, unrepresentable nudges, and measured bound
excess. No code-center reset or context mutation occurs.

## Frozen fixture tolerances

Before F05 CPU/MPS probes, assistance fixed absolute tolerance zero and relative
tolerance `2*finfo(float32).eps = 2.384185791015625e-7` **relative to the
amplification bound**, inherited from F03's measured reconstruction contract.
A large base weight never supplies extra absolute tolerance. Raw candidate and
accepted excess remain reported; the numerical CLI independently audits accepted
displacements in CPU float64 after the device calls. Geometry uses relative
`2e-6`, absolute zero, inherited from F04's small FP32 algebra fixtures (F02
observed maximum operator error `1.1920928955078125e-7`). Duplicate assistance
requires bit-exact output/masks: allowance zero, consistent with F02's measured
zero duplicate variability. These are numerical fixture bounds, not neural
loss/gradient/resume or future optimizer integration tolerances.

CPU and actual MPS, fallback disabled, each checked 100,000 deterministic
coordinates per tau `.125,.25,.5,.75,1`, all 14 midpoint ties in both directions,
endpoint/saturation controls, cache refresh, sign/zero/exponent cap, identity and
decay separation. Accepted counts are `41076,34094,20120,8549,0`; no random
eligible proposal was rejected; accepted float64 bound excess, bound-tolerance
violations, tie mismatches and duplicate differences were all zero. The explicit
scale bit-pattern 12, U0 bit-pattern 12, update bit-pattern 1 case rejected one
unrepresentable nudge and retained shadow bit-pattern 13 on both devices.
Geometry maximum absolute error against the independent float64 S17 fixture was
`6.827305498724456e-9` CPU and `9.62342726618104e-8` actual MPS.

## Discriminating diagnostics

`run_quadratic_diagnostics(seed,horizons)` returns deterministic CPU float64
mechanism diagnostics. Unit-scale `U0=1,d=.04,tau=.125` gives shadow
`1.2500390625` and next code `1.5`: squared error for target 1.1 worsens
`.01 -> .16`; target 1.8 improves `.64 -> .09`. Naive negative-weight
multiplication and unconditional midpoint nudging each fail a dedicated mutation
control. Boundary assistance therefore has both favorable and unfavorable cases.

Positive q is a reparameterization: `q*w=w*exp(log(q))` for q>0, checked
at q `.5,.9,.99,.999,1` with CPU float64 relative tolerance four epsilon and
exact q=1 identity. A separate seeded 1,024-sample signed-w/epsilon fixture
compares Python math exp with NumPy exp, records the input hash and errors,
and never takes a logarithm of signed w. There is no q policy action. The separate cheap Jackson
quadratic check uses `D_q f(w)=(f(q*w)-f(w))/((q-1)*w)` for
`f(w)=.5*(w-target)**2`, whose analytical identity is
`w-target+.5*(q-1)*w`. It records independent difference quotients, the q→1
ordinary derivative, and explicit continuous extension `-target` at w=0.
Original Appendix A is independently rerun: `w=1.7,target=-.2,q=1.3` gives
direct `2.1550000000000007` versus derived `2.155`. Its finite-step correction
is not a new information source; no per-neural-parameter finite-q evaluation
is introduced. Difference-quotient absolute tolerance is `4e-12`, from float64 cancellation
error (the smallest tested nonzero denominator is `.0005`, and numerator
objective magnitudes are below one). The q=.999 derivative limit error is
bounded analytically by `.5*abs(q-1)*max(abs(w))`; no empirical limit tolerance
is substituted. FP32-nextafter, persistent shadows versus center reset,
stochastic unbiased expectation (100,000 seeded diagnostic draws, no new
sampling gate), and power-of-two scale homogeneity are also recorded.
Stochastic production quantization remains M05-owned.

Coupled SPD fixtures cover dimensions 2/4/8 and condition numbers 1/10/100. Each
enumerates all `3**dimension` immediate ordinal moves `{-1,0,+1}` and then K
common persistent-shadow, quantized-gradient baseline steps with rate
`.1/condition_number`. Baseline is the all-zero immediate action. The oracle
knows the quadratic and full candidate rollouts; each record includes its
privilege, `3**dimension*(K+1)` objective evaluations, `3**dimension*K` baseline
steps, matrix/target/fixture hash and trajectories. Caller horizons are honored;
CLI diagnostic defaults are `[0,1,4]` and seed `20261004`, yielding 27 fixture
records. These are tiny diagnostics, not neural/calibration budgets.

The default `python -m ralup check-numerics --output artifacts/checks/results.json`
runs F03–F05 and writes environment, source hashes, controls and invariant
results. `--suite quantizer` preserves F03/F04-only checking. Actual MPS requires
`PYTORCH_ENABLE_MPS_FALLBACK=0` before startup and device access. A sandbox
unavailable result is retained as blocked evidence, never an MPS pass.
[f05-evidence.json](f05-evidence.json) retains actual evidence hashes; local raw
reports stay in ignored artifacts. Numerical fixtures pass CPU and actual MPS;
review acceptance and future H/M harness, optimizer, source-parity and scientific
training gates remain independent. No dataset, training or protected scoring
was performed.
