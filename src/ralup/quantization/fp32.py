"""Device FP32 bit rescaling for gradual underflow without denormal arithmetic.

Only integer tensors and normal-range FP32 arithmetic are used. The helpers
assemble rounded subnormal results with integer bits; no host tensor copies or
higher precision tensor arithmetic occur. Consumers supply their autograd rule.
"""
import torch
from torch import Tensor

from .formats import _significand_exponent


def bits(x: Tensor) -> Tensor:
    return x.contiguous().view(torch.int32)


def frexp_bits(x: Tensor) -> tuple[Tensor, Tensor]:
    """Return exact normal mantissa m and integer e, x=m*2**e (zero m=0)."""
    xb = bits(x)
    sig, exponent = _significand_exponent(xb)
    raw = sig.to(torch.float32).view(torch.int32)
    mantissa_bits = (raw & 0x7fffff) | (126 << 23) | (xb & -2147483648)
    mantissa_bits = torch.where(sig == 0, 0, mantissa_bits)
    e = ((raw >> 23) & 255) - 126 + exponent
    return mantissa_bits.view(torch.float32), e


def rescale_pow2(x: Tensor, shift: Tensor | int) -> Tensor:
    """Round x*2**shift to FP32 via bit assembly, with nearest-even underflow.

    Overflow yields signed infinity so the consumer can reject a nonfinite
    output, or compare an overflowing absolute error to a finite threshold.
    """
    xb = bits(x)
    sig, exponent = _significand_exponent(xb)
    raw = sig.to(torch.float32).view(torch.int32)
    target = ((raw >> 23) & 255) + exponent + shift
    significand = ((raw & 0x7fffff) | 0x800000).to(torch.int64)
    right = (1-target).clamp(1, 31).to(torch.int64)
    quotient = significand >> right
    remainder = significand & ((torch.ones_like(right) << right)-1)
    half = torch.ones_like(right) << (right-1)
    rounded = quotient + ((remainder > half) | ((remainder == half) & ((quotient & 1) != 0))).to(torch.int64)
    rounded = torch.where(target < -23, 0, rounded).to(torch.int32)
    normal = (raw & 0x7fffff) | (target.clamp(1, 254) << 23)
    magnitude = torch.where(target > 0, normal, rounded)
    magnitude = torch.where(target >= 255, 0x7f800000, magnitude)
    magnitude = torch.where(sig == 0, 0, magnitude)
    return (magnitude | (xb & -2147483648)).view(torch.float32)


def multiply_rescaled(x: Tensor, factor: Tensor | float) -> Tensor:
    """FP32 product with exact integer nearest-even rounding of tiny results."""
    factor = torch.as_tensor(factor, dtype=torch.float32, device=x.device)
    xb, fb = bits(x), bits(factor)
    xs, xe = _significand_exponent(xb)
    fs, fe = _significand_exponent(fb)
    product = xs.to(torch.int64)*fs.to(torch.int64)  # at most 48 significant bits
    ordinary = rescale_pow2(product.to(torch.float32), xe+fe)
    target = ((bits(product.to(torch.float32)) >> 23) & 255)+xe+fe
    right = (-(xe+fe+149)).clamp(1, 62).to(torch.int64)
    quotient = product >> right
    remainder = product & ((torch.ones_like(right) << right)-1)
    half = torch.ones_like(right) << (right-1)
    rounded = quotient + ((remainder > half) | ((remainder == half) & ((quotient & 1) != 0))).to(torch.int64)
    rounded = torch.where(xe+fe+149 < -62, 0, rounded).to(torch.int32)
    magnitude = torch.where(target <= 0, rounded, bits(ordinary) & 0x7fffffff)
    return (magnitude | ((xb ^ fb) & -2147483648)).view(torch.float32)


def divide_sqrt2(x: Tensor) -> Tensor:
    """FP32 x/sqrt(2), with exact rational rounding for subnormal quotients.

    FP32 sqrt(2) has significand 11863283 and unit exponent -23.
    For |x|<2*minimum-normal the quotient in 2**-149 units is
    significand(x)*2**(unit_exponent(x)+172)/11863283. The numerator
    fits int64, so division/remainder supply one rounding, without double
    rounding a normal intermediate into a subnormal result.
    """
    import math
    xb = bits(x)
    magnitude = xb & 0x7fffffff
    sig, exponent = _significand_exponent(xb)
    numerator = sig.to(torch.int64) << (exponent+172).clamp(0, 24).to(torch.int64)
    quotient, remainder = numerator // 11863283, numerator % 11863283
    rounded = quotient + (2*remainder > 11863283).to(torch.int64)
    ordinary = bits(x/math.sqrt(2.))
    tiny_bits = rounded.to(torch.int32) | (xb & -2147483648)
    return torch.where(magnitude < 0x1000000, tiny_bits, ordinary).view(torch.float32)


def divide_six(x: Tensor) -> Tensor:
    """Positive x/6, with exact rational nearest-even subnormal rounding."""
    xb = bits(x)
    small = xb < 0x2000000  # x<8*minimum-normal contains every subnormal x/6.
    sig, exponent = _significand_exponent(xb)
    units = sig.to(torch.int64) << (exponent+149).clamp(0, 3).to(torch.int64)
    quotient, remainder = units // 6, units % 6
    rounded = quotient + ((remainder > 3) | ((remainder == 3) & ((quotient & 1) != 0))).to(torch.int64)
    # A rounded unit count >=2**23 is already a minimum-exponent normal bit pattern.
    ordinary_bits = bits(x/6)
    return torch.where(small, rounded.to(torch.int32), ordinary_bits).view(torch.float32)


def add_rescaled(a: Tensor, b: Tensor, *, subtract: bool = False) -> Tensor:
    """FP32 addition/subtraction with preserved gradual underflow in the result."""
    ma, ea = frexp_bits(a)
    mb, eb = frexp_bits(b)
    common = torch.maximum(ea, eb)
    aa = rescale_pow2(ma, ea-common)
    bb = rescale_pow2(mb, eb-common)
    result = aa-bb if subtract else aa+bb
    return rescale_pow2(result, common)
