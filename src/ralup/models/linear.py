"""Original-coordinate FP32 projections with optional detached storage caches."""
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..quantization.context import QuantizationContext, quantized_view
from ..quantization.rotation import to_storage


class QuantizedLinear(nn.Module):
    """Bias-free projection; its W Parameter is never replaced or rotated.

    Contexts contain numeric snapshots only. Each call builds an independent
    surrogate graph into W. Activation arithmetic remains FP32.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.bias = None
        self.context: QuantizationContext | None = None
        nn.init.normal_(self.weight, std=.02)

    def forward(self, x: Tensor, context: QuantizationContext | None = None) -> Tensor:
        context = self.context if context is None else context
        if context is None:
            return F.linear(x, self.weight)
        from ..baselines.ewgs import NormalizedLeafView
        values = context.storage_values if type(context) is NormalizedLeafView else quantized_view(self.weight, context)
        return F.linear(to_storage(x, context.rotation), values)
