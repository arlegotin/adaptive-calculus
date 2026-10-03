"""S17/S18 proposals on U0 after common decay, with frozen context scales.

There is no optimizer state here. Callers inverse-map storage actions and keep
moments in original coordinates. Identity callers may commit their authoritative
original-coordinate backbone result directly (H05/M09 own that arithmetic).
"""
import math
import torch
from torch import Tensor
from ..quantization.context import QuantizationContext
from ..quantization.formats import E2M1_THRESHOLDS, _preserved_reconstruction, directional_neighbor, quantize_e2m1
from ..quantization.fp32 import add_rescaled, bits, multiply_rescaled

# Frozen from F03's two-epsilon FP32 reconstruction contract before F05 probes.
# Relative to the displacement bound, never relative to a large base weight.
BOUND_RTOL = 2 * torch.finfo(torch.float32).eps
BOUND_ATOL = 0.


def _validate(u0, d):
    if u0.dtype != torch.float32 or d.dtype != torch.float32:
        raise TypeError('Action tensors must be float32')
    if u0.shape != d.shape or u0.device != d.device or not u0.numel():
        raise ValueError('Action tensors must have matching nonempty shape/device')
    if not bool(((bits(u0) & 0x7fffffff) < 0x7f800000).all()) or not bool(((bits(d) & 0x7fffffff) < 0x7f800000).all()):
        raise ValueError('Action tensors must be finite')


def _abs(x):
    return (bits(x) & 0x7fffffff).view(torch.float32)


def _rms(x):
    maximum = _abs(x).amax()
    if int(bits(maximum)) == 0:
        return maximum
    return maximum * (x/maximum).square().mean().sqrt()


@torch.no_grad()
def multiplicative_proposal(u0: Tensor, da: Tensor) -> Tensor:
    """Sign-aware capped exponential displacement; no later norm matching."""
    _validate(u0, da)
    r = _rms(u0).clamp_min(1e-30)
    a = torch.sign(u0)*da/r
    z = u0*a
    rz = _rms(z)
    c = _rms(da)/rz if int(bits(rz)) != 0 else torch.zeros_like(rz)
    return u0*torch.expm1((c*a).clamp(-.1, .1))


@torch.no_grad()
def assist_boundaries(u0: Tensor, d: Tensor, tau: float, context: QuantizationContext) -> tuple[Tensor, dict]:
    """Return shadow U, accepted/rejected masks and measured displacement excess.

    Q(U0), Q(U0+d) are deterministic nearest-even at context.scales, regardless
    of any cached stochastic realization. Scales and context are never refreshed
    or mutated here. Unrepresentable nudges retain the additive shadow.
    """
    _validate(u0, d)
    if not math.isfinite(tau) or not 0 < tau <= 1:
        raise ValueError('Assistance tau must be in (0,1]')
    if context.scales.device != u0.device:
        raise ValueError('Context and actions must share device')
    scales = torch.broadcast_to(context.scales, context.storage_u.shape)
    if scales.numel() != u0.numel():
        raise ValueError('Action shape must match context storage coordinate count')
    scales = scales.reshape(u0.shape).contiguous()
    empty = torch.zeros_like(u0, dtype=torch.bool)
    info = {'bound_rtol': BOUND_RTOL, 'bound_atol': BOUND_ATOL, 'context_step': context.step,
            'scale_source': context.metadata['scale_source'], 'scales_refreshed': False,
            'assistance_bypassed': tau == 1., 'feasible_mask': empty, 'eligible_mask': empty,
            'accepted_mask': empty, 'rejected_mask': empty, 'accepted_count': 0, 'rejected_count': 0,
            'unrepresentable_nudge_count': 0, 'bound_tolerance_exceeded_count': 0,
            'observed_bound_excess_max': 0., 'accepted_bound_excess_max': 0.}
    # Exact bypass: preserve caller-visible FP32 expression/order, no quantization.
    if tau == 1.:
        return u0+d, info
    # Existing bit-safe primitives preserve actual MPS representable subnormals.
    tiny = bool(((bits(u0) & 0x7fffffff) < 0x800000).any()) or bool(((bits(d) & 0x7fffffff) < 0x800000).any()) or bool((bits(scales) < 0x1000000).any())
    add = add_rescaled if tiny else lambda a, b, subtract=False: a-b if subtract else a+b
    mul = multiply_rescaled if tiny else lambda a, b: a*b
    y = add(u0, d)
    q0, codes0 = quantize_e2m1(u0, scales)
    _, codes1 = quantize_e2m1(y, scales)
    neighbor, exists = directional_neighbor(codes0, d)
    qn = _preserved_reconstruction(neighbor.to(torch.int64), bits(scales))
    # Physical midpoint = exact code-space midpoint * cached scale. Avoid
    # averaging rounded reconstructed levels at the bottom of FP32's range.
    thresholds = torch.tensor(E2M1_THRESHOLDS, device=u0.device)
    index = torch.minimum(codes0, neighbor).long().clamp_max(13)
    b = mul(scales, thresholds[index])
    rho = _abs(add(b, u0, subtract=True))
    gap = _abs(add(qn, q0, subtract=True))
    upper = _abs(_preserved_reconstruction(torch.full_like(codes0, 14).long(), bits(scales)))
    within = (bits(_abs(u0)) <= bits(upper))
    feasible = exists & within
    eligible = feasible & (codes1 == codes0) & (bits(rho) > 0) & (bits(_abs(d)) >= bits(mul(rho, tau)))
    margin = mul(torch.minimum(gap, _abs(d)), 1/1024)
    signed_margin = (bits(margin) | (bits(d) & -2147483648)).view(torch.float32)
    candidate = add(b, signed_margin)
    _, candidate_codes = quantize_e2m1(candidate, scales)
    displacement = _abs(add(candidate, u0, subtract=True))
    bound = mul(_abs(d), 1/tau+1/1024)
    allowed = mul(bound, 1+BOUND_RTOL)
    representable = bits(_abs(add(candidate, b, subtract=True))) > 0
    bound_ok = bits(displacement) <= bits(allowed)
    accept = eligible & representable & (candidate_codes == neighbor) & (bits(_abs(candidate)) <= bits(upper)) & bound_ok
    reject = eligible & ~accept
    excess = (displacement-bound).clamp_min(0.)
    info.update({'feasible_mask': feasible, 'eligible_mask': eligible, 'accepted_mask': accept,
                 'rejected_mask': reject, 'initial_codes': codes0, 'baseline_codes': codes1,
                 'neighbor_codes': neighbor, 'candidate_codes': candidate_codes,
                 'accepted_count': int(accept.sum()), 'rejected_count': int(reject.sum()),
                 'unrepresentable_nudge_count': int((eligible & ~representable).sum()),
                 'bound_tolerance_exceeded_count': int((eligible & ~bound_ok).sum()),
                 'observed_bound_excess_max': float(torch.where(eligible, excess, 0.).max()),
                 'accepted_bound_excess_max': float(torch.where(accept, excess, 0.).max()),
                 'baseline_crossing_count': int((codes1 != codes0).sum()),
                 'infeasible_count': int((~feasible).sum())})
    return torch.where(accept, candidate, y), info


@torch.no_grad()
def compose_action(u0: Tensor, da: Tensor, alpha: float, h: float, tau: float, context: QuantizationContext) -> tuple[Tensor, dict]:
    """Blend only the update; common decay belongs in U0 before this call."""
    _validate(u0, da)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1 or not math.isfinite(h) or h < 0:
        raise ValueError('Action requires alpha in [0,1] and finite nonnegative h')
    d = da if alpha == 0 and h == 1 else h*((1-alpha)*da+alpha*multiplicative_proposal(u0, da))
    out, info = assist_boundaries(u0, d, tau, context)
    identity = alpha == 0 and h == 1 and tau == 1
    info.update({'identity_action': identity, 'authoritative_original_commit_bypass_permitted': identity,
                 'identity_commit_coordinates': 'original_W', 'output_coordinates': 'storage_U',
                 'decay_reference': 'already_in_U0; update_only_amplification',
                 'post_exponential_normalization': False, 'post_assistance_normalization': False})
    return out, info
