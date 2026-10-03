"""E2M1 numerical ordinals, canonical encoding nibbles, and significand parity.

OCP Microscaling Formats v1.0: S EE M, exponent bias 1. Ordinal 7 is
positive zero (nibble 0); the negative-zero nibble 8 has no distinct ordinal.
Source evidence and project conversion choices live in docs/numerics/.
"""
import torch
from torch import Tensor

E2M1_VALUES = (-6., -4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 6.)
E2M1_NIBBLES = (15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7)
# Least-significant significand bit, independently of ordinal parity.
E2M1_EVEN = (False, True, False, True, False, True, False, True,
             False, True, False, True, False, True, False)
E2M1_THRESHOLDS = (-5., -3.5, -2.5, -1.75, -1.25, -.75, -.25,
                  .25, .75, 1.25, 1.75, 2.5, 3.5, 5.)


def _fp32_pair(x: Tensor, scale: Tensor) -> None:
    if x.dtype != torch.float32 or scale.dtype != torch.float32:
        raise TypeError('E2M1 tensor inputs and scales must be float32')
    if x.device != scale.device:
        raise ValueError('E2M1 tensor inputs and scales must share a device')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('E2M1 rejects nonfinite original inputs')
    if not bool((torch.isfinite(scale) & (scale > 0)).all()):
        raise ValueError('E2M1 scales must be finite and positive')


@torch.no_grad()
def quantize_e2m1(x: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    """Round x/scale in FP32; return reconstructed FP32 values and uint8 ordinals.

    Scalar validation reductions synchronize to raise errors. Tensor data stays
    on its device; 14 sequential comparisons use O(P) storage, never [P,15].
    Finite division overflow saturates normally. Reconstruction overflow is a
    storage error, rather than an alternate rounding/clipping rule. Generic
    PyTorch broadcasting applies (row scales should have shape [rows,1]).
    """
    _fp32_pair(x, scale)
    scaled = x / scale
    codes = torch.zeros_like(scaled, dtype=torch.int64)
    for upper, threshold in enumerate(E2M1_THRESHOLDS, start=1):
        crosses = scaled >= threshold if E2M1_EVEN[upper] else scaled > threshold
        codes.add_(crosses)
    levels = torch.tensor(E2M1_VALUES, device=x.device, dtype=torch.float32)
    values = levels[codes] * scale
    if not bool(torch.isfinite(values).all()):
        raise ValueError('E2M1 reconstruction overflow in float32 storage')
    # Explicit canonicalization includes input -0 and underflowed products.
    values = torch.where(values == 0, torch.zeros_like(values), values)
    return values, codes.to(torch.uint8)


@torch.no_grad()
def directional_neighbor(codes: Tensor, direction: Tensor) -> tuple[Tensor, Tensor]:
    """Immediate ordinal in sign(direction); zero/outward directions infeasible.

    An infeasible neighbor retains its input ordinal, including endpoint codes.
    Ordinals are numerical adjacency indices, never encoding nibbles.
    """
    if codes.dtype != torch.uint8:
        raise TypeError('E2M1 neighbor codes must be uint8 ordinals')
    if direction.dtype != torch.float32:
        raise TypeError('E2M1 neighbor directions must be float32')
    if codes.device != direction.device:
        raise ValueError('Codes and directions must share a device')
    if not bool((codes <= 14).all()):
        raise ValueError('E2M1 ordinals must lie in [0,14]')
    if not bool(torch.isfinite(direction).all()):
        raise ValueError('E2M1 neighbor directions must be finite')
    step = torch.sign(direction).to(torch.int64)
    candidate = codes.to(torch.int64) + step
    feasible = (step != 0) & (candidate >= 0) & (candidate <= 14)
    return torch.where(feasible, candidate, codes.to(torch.int64)).to(torch.uint8), feasible
