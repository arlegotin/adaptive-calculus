"""Fresh storage-coordinate surrogate graphs over cached detached forward data."""
import torch
from torch import Tensor

from .formats import _preserved_ratio
from .fp32 import add_rescaled, bits

BACKWARD_RULES = ('ste', 'trust', 'clipped_ste')


class _CachedSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, values, mask):
        ctx.save_for_backward(mask)
        return values.clone()

    @staticmethod
    def backward(ctx, upstream):
        (mask,) = ctx.saved_tensors
        gradient = torch.where(mask != 0, bits(upstream), 0).view(torch.float32)
        return gradient, None, None


def surrogate_view(u: Tensor, values: Tensor, mask: Tensor) -> Tensor:
    """Exact cached values in forward; fresh mask graph in every invocation.

    u is freshly transformed from W. Autograd through that orthonormal transform
    applies R transpose, leaving parameter gradients/moments in W coordinates.
    """
    return _CachedSurrogate.apply(u, values, mask)


@torch.no_grad()
def backward_mask(u: Tensor, values: Tensor, scales: Tensor, rule: str) -> Tensor:
    if rule not in BACKWARD_RULES:
        raise ValueError(f'Unknown backward rule: {rule}')
    if rule == 'ste':
        return torch.ones_like(u)
    scale_bits = bits(scales)
    if rule == 'trust':
        # Physical outer half-gap is s, not the inherited release Lloyd threshold.
        difference = add_rescaled(values, u, subtract=True)
        return ((bits(difference) & 0x7fffffff) <= scale_bits).to(torch.float32)
    # Separately declared clipped STE: derivatives only inside representable range.
    # Division avoids 6*s overflow for caller-supplied large fixed scales.
    return (_preserved_ratio(bits(u), scale_bits).abs() <= 6.).to(torch.float32)
