"""Fresh storage-coordinate surrogate graphs over cached detached forward data."""
import torch
from torch import Tensor

BACKWARD_RULES = ('ste', 'trust', 'clipped_ste')


class _CachedSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, values, mask):
        ctx.save_for_backward(mask)
        return values.clone()

    @staticmethod
    def backward(ctx, upstream):
        (mask,) = ctx.saved_tensors
        return upstream*mask, None, None


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
    scale_bits = scales.detach().contiguous().view(torch.int32)
    if bool((scale_bits < 0x800000).any()):
        raise ValueError('Subnormal surrogate scale risks device mask underflow; use plain STE or a normal scale')
    if rule == 'trust':
        # Physical outer half-gap is s, not the inherited release Lloyd threshold.
        return ((values-u).abs() <= scales).to(torch.float32)
    # Separately declared clipped STE: derivatives only inside representable range.
    # Division avoids 6*s overflow for caller-supplied large fixed scales.
    return ((u/scales).abs() <= 6.).to(torch.float32)
