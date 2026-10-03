"""Project full-last-axis Sylvester rotation, never an optimizer-state transform.

S13 uses U=W@R and linear output (X@R)@Q(U).T. Identity and the full
orthonormal Hadamard are self-inverse. QuEST's 128-block release rotation is
not this recipe; see docs/numerics/quantization-context.md for provenance.
"""
import math

import torch
from torch import Tensor

ROTATIONS = ('identity', 'full_hadamard')


def hadamard(x: Tensor) -> Tensor:
    """Orthonormal butterfly along the complete last axis; preserves autograd."""
    if x.dtype != torch.float32:
        raise TypeError('Hadamard requires float32 tensors')
    if x.ndim < 1 or x.shape[-1] < 1 or x.shape[-1] & (x.shape[-1]-1):
        raise ValueError('Hadamard last-axis width must be a positive power of two')
    bits = x.detach().contiguous().view(torch.int32) & 0x7fffffff
    if not bool((bits < 0x7f800000).all()):
        raise ValueError('Hadamard requires finite inputs')
    if bool(((bits > 0) & (bits < 0x800000)).any()):
        raise ValueError('Hadamard subnormal input risks device underflow')
    result = x
    stride = 1
    while stride < x.shape[-1]:
        groups = result.reshape(*x.shape[:-1], -1, 2, stride)
        # Normalize before addition to avoid unnecessary intermediate overflow.
        left = groups[..., 0, :] / math.sqrt(2.)
        right = groups[..., 1, :] / math.sqrt(2.)
        result = torch.stack((left+right, left-right), dim=-2).reshape_as(x)
        stride *= 2
    outbits = result.detach().contiguous().view(torch.int32) & 0x7fffffff
    if not bool((outbits < 0x7f800000).all()):
        raise ValueError('Hadamard overflow in float32 storage')
    if bool(((bits.amax(dim=-1) > 0) & (outbits.amax(dim=-1) == 0)).any()):
        raise ValueError('Hadamard nonzero row underflow in float32 storage')
    if bool(((outbits > 0) & (outbits < 0x800000)).any()):
        raise ValueError('Hadamard subnormal output risks device underflow')
    return result


def to_storage(x: Tensor, rotation: str) -> Tensor:
    """Map original-coordinate FP32 values to U. Do not apply to moments."""
    if rotation not in ROTATIONS:
        raise ValueError(f'Unknown project rotation: {rotation}')
    if x.dtype != torch.float32:
        raise TypeError('Storage transforms require float32 tensors')
    return x if rotation == 'identity' else hadamard(x)


def from_storage(u: Tensor, rotation: str) -> Tensor:
    """Map storage-coordinate FP32 displacements back with R transpose."""
    return to_storage(u, rotation)
