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


def _finite_bits(bits: Tensor) -> Tensor:
    return (bits & 0x7fffffff) < 0x7f800000


def _significand_exponent(bits: Tensor) -> tuple[Tensor, Tensor]:
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7fffff
    significand = torch.where(exponent == 0, fraction, fraction | 0x800000)
    return significand, torch.where(exponent == 0, -149, exponent - 150)


def _preserved_ratio(xbits: Tensor, sbits: Tensor) -> Tensor:
    # Integer significands are exactly representable in FP32. Divide them in
    # the normal range, then shift by an exact normal power of two. Exponent
    # gaps beyond ±28 already imply zero-cell/saturation for all significands.
    xsig, xexp = _significand_exponent(xbits)
    ssig, sexp = _significand_exponent(sbits)
    shift = (xexp - sexp).clamp(-28, 28)
    power = ((shift + 127) << 23).view(torch.float32)
    magnitude = (xsig.to(torch.float32) / ssig.to(torch.float32)) * power
    return torch.where(xbits < 0, -magnitude, magnitude)


def _preserved_reconstruction(codes: Tensor, sbits: Tensor) -> Tensor:
    # Each E2M1 magnitude is an integer / 2. Exact products fit in 28 bits.
    # Integer-to-FP32 conversion supplies one nearest-even rounding for normal
    # outputs; exponent assembly avoids an MPS denormal multiply/flush.
    sig, exponent = _significand_exponent(sbits)
    twice_magnitudes = torch.tensor((12, 8, 6, 4, 3, 2, 1, 0, 1, 2, 3, 4, 6, 8, 12),
                                   dtype=torch.int32, device=codes.device)
    product = sig * twice_magnitudes[codes]
    raw = product.to(torch.float32).view(torch.int32)
    output_exponent = ((raw >> 23) & 255) + exponent - 1
    if bool(((product != 0) & (output_exponent >= 255)).any()):
        raise ValueError('E2M1 reconstruction overflow in float32 storage')
    normal = (raw & 0x7fffff) | (output_exponent << 23)
    # Subnormal results can only have unit exponent -150: round product/2
    # directly, avoiding double rounding. Odd halfway products round to even.
    subnormal = (product >> 1) + ((product & 3) == 3).to(torch.int32)
    magnitude = torch.where(output_exponent > 0, normal, subnormal)
    magnitude = torch.where(product == 0, 0, magnitude)
    sign = torch.where((codes < 7) & (magnitude != 0), -2147483648, 0).to(torch.int32)
    return (magnitude | sign).view(torch.float32)


def _fp32_pair(x: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    if x.dtype != torch.float32 or scale.dtype != torch.float32:
        raise TypeError('E2M1 tensor inputs and scales must be float32')
    if x.device != scale.device:
        raise ValueError('E2M1 tensor inputs and scales must share a device')
    xbits, sbits = x.view(torch.int32), scale.view(torch.int32)
    if not bool(_finite_bits(xbits).all()):
        raise ValueError('E2M1 rejects nonfinite original inputs')
    if not bool(((sbits > 0) & (sbits < 0x7f800000)).all()):
        raise ValueError('E2M1 scales must be finite and positive')
    return xbits, sbits


@torch.no_grad()
def quantize_e2m1(x: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    """Round x/scale in FP32; return reconstructed FP32 values and uint8 ordinals.

    Scalar validation reductions synchronize to raise errors. Tensor data stays
    on its device; 14 sequential comparisons use O(P) storage, never [P,15].
    Subnormal operands/outputs use device integer bit decoding plus normal FP32
    arithmetic to preserve gradual underflow on backends that flush denormals.
    Finite division overflow saturates normally. Reconstruction overflow is a
    storage error, rather than an alternate rounding/clipping rule. Generic
    PyTorch broadcasting applies (row scales should have shape [rows,1]).
    """
    xbits, sbits = _fp32_pair(x, scale)
    subnormal_input = ((xbits & 0x7fffffff) > 0) & ((xbits & 0x7fffffff) < 0x800000)
    subnormal_scale = sbits < 0x800000
    exceptional = bool(subnormal_input.any()) or bool(subnormal_scale.any())
    scaled = _preserved_ratio(xbits, sbits) if exceptional else x / scale
    codes = torch.zeros_like(scaled, dtype=torch.int64)
    for upper, threshold in enumerate(E2M1_THRESHOLDS, start=1):
        crosses = scaled >= threshold if E2M1_EVEN[upper] else scaled > threshold
        codes.add_(crosses)
    levels = torch.tensor(E2M1_VALUES, device=x.device, dtype=torch.float32)
    # Even a normal scale at exponent -126 can reconstruct a subnormal .5s.
    values = (_preserved_reconstruction(codes, sbits) if bool((sbits < 0x1000000).any())
              else levels[codes] * scale)
    value_bits = values.view(torch.int32)
    if not bool(_finite_bits(value_bits).all()):
        raise ValueError('E2M1 reconstruction overflow in float32 storage')
    # Explicit canonicalization includes input -0 and underflowed products.
    values = torch.where((value_bits & 0x7fffffff) == 0, 0, value_bits).view(torch.float32)
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
    direction_bits = direction.view(torch.int32)
    if not bool(_finite_bits(direction_bits).all()):
        raise ValueError('E2M1 neighbor directions must be finite')
    step = torch.where((direction_bits & 0x7fffffff) == 0, 0,
                       torch.where(direction_bits < 0, -1, 1)).to(torch.int64)
    candidate = codes.to(torch.int64) + step
    feasible = (step != 0) & (candidate >= 0) & (candidate <= 14)
    return torch.where(feasible, candidate, codes.to(torch.int64)).to(torch.uint8), feasible
