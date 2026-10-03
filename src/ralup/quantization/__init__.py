"""Encoding-aware simulated E2M1 W4A32 numerical primitives."""
from .formats import directional_neighbor, quantize_e2m1
from .oracle import e2m1_oracle
from .context import QuantizationContext, prepare_context, quantized_view
from .rotation import from_storage, hadamard, to_storage
from .scales import compute_scales, fit_gaussian_e2m1

__all__ = ['directional_neighbor', 'e2m1_oracle', 'quantize_e2m1',
           'QuantizationContext', 'prepare_context', 'quantized_view',
           'from_storage', 'hadamard', 'to_storage', 'compute_scales', 'fit_gaussian_e2m1']
