"""Logical-step numerical caches, with fresh autograd views for each microbatch.

quantized_view returns STORAGE Q(U), where U=W@R. A linear caller must use
(X@R)@quantized_view(W, context).T. This interface never rotates moments.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from .formats import _preserved_reconstruction, quantize_e2m1
from .rotation import to_storage
from .scales import compute_scales, group_zero_flags, grouping_for
from .surrogates import BACKWARD_RULES, backward_mask, surrogate_view


@dataclass(frozen=True)
class QuantizationContext:
    """Caller-owned detached snapshot; treat its tensors as read-only.

    Continuous state is FP32; numerical ordinals/zero flags are uint8/bool.
    No Parameter, graph, optimizer state or RNG state is owned here.
    """
    scales: Tensor
    codes: Tensor
    storage_values: Tensor
    storage_u: Tensor
    gradient_mask: Tensor
    zero_groups: Tensor
    scale_zero_groups: Tensor | None
    rotation: str
    grouping: str
    backward: str
    recipe: str
    step: int
    metadata: Mapping

    def numeric_tensors(self) -> tuple[Tensor, ...]:
        return (self.scales, self.codes, self.storage_values, self.storage_u,
                self.gradient_mask, self.zero_groups) + (() if self.scale_zero_groups is None else (self.scale_zero_groups,))


@torch.no_grad()
def prepare_context(w: Tensor, recipe: dict, step: int, rng: object) -> QuantizationContext:
    """Prepare once at a logical-step start; reuse this context in all microbatches.

    recipe keys: name, rotation (identity/full_hadamard), backward
    (ste/trust/clipped_ste), previous_scales, previous_zero_groups, warm_start,
    codes. For fixed
    recipes the caller owns the frozen scales and passes previous_scales on
    subsequent calls (and previous_zero_groups to retain initialization flags);
    alternatively initialize from explicit warm_start W.
    Initial w is the warm start when neither is provided. Dynamic preparation
    always recomputes. No hidden history/cache mutation occurs. rng is reserved
    for M05: this function does not draw. An externally supplied ordinal
    realization can be shared; generation/parity of stochastic codes is M05.
    Final evaluation may prepare a read-only context without advancing RNG.
    """
    if not isinstance(recipe, dict):
        raise TypeError('Recipe must be a dictionary')
    allowed = {'name', 'rotation', 'backward', 'previous_scales', 'previous_zero_groups', 'warm_start', 'codes'}
    if set(recipe)-allowed:
        raise ValueError(f'Unknown context recipe fields: {sorted(set(recipe)-allowed)}')
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError('Logical step must be a nonnegative integer')
    if w.ndim != 2:
        raise ValueError('Weight context requires an [out,in] matrix')
    name = recipe['name']
    rotation = recipe.get('rotation', 'identity')
    backward = recipe.get('backward', 'trust' if name.startswith('quest') else 'ste')
    grouping = grouping_for(name)
    if backward not in BACKWARD_RULES:
        raise ValueError(f'Unknown backward rule: {backward}')
    u = to_storage(w.detach(), rotation).clone()
    previous = recipe.get('previous_scales')
    warm_start = recipe.get('warm_start')
    previous_zero = recipe.get('previous_zero_groups')
    if previous_zero is not None and (previous is None or 'fixed' not in name):
        raise ValueError('Previous zero flags require previous fixed scales')
    scale_input = u
    if warm_start is not None:
        if 'fixed' not in name or previous is not None:
            raise ValueError('Warm start is only valid for fixed initialization without previous scales')
        if warm_start.shape != w.shape or warm_start.device != w.device:
            raise ValueError('Warm start must match weight shape/device')
        scale_input = to_storage(warm_start.detach(), rotation)
    scales = compute_scales(scale_input, name, previous)
    values, codes = quantize_e2m1(u, scales)
    zero = group_zero_flags(u, name)
    supplied = recipe.get('codes')
    if supplied is not None:
        if supplied.dtype != torch.uint8 or supplied.shape != u.shape or supplied.device != u.device:
            raise ValueError('Supplied codes must be uint8 ordinals with weight shape/device')
        if not bool((supplied <= 14).all()):
            raise ValueError('Supplied codes must be ordinals in [0,14]')
        if bool((zero & (supplied != 7)).any()):
            raise ValueError('Supplied codes must remain canonical zero in an all-zero group')
        codes = supplied.detach().clone()
        values = _preserved_reconstruction(codes.to(torch.int64), scales.view(torch.int32))
    scale_zero = group_zero_flags(scale_input, name) if previous is None else None
    if previous_zero is not None:
        if previous_zero.dtype != torch.bool or previous_zero.shape != scales.shape or previous_zero.device != w.device:
            raise ValueError('Previous zero flags must match fixed scale shape/device and be boolean')
        scale_zero = previous_zero.detach().clone()
    mask = backward_mask(u, values, scales, backward)
    scale_source = ('previous_fixed_scales' if previous is not None else 'explicit_warm_start' if warm_start is not None
                    else 'initial_weight_warm_start' if 'fixed' in name else 'current_logical_step')
    metadata = MappingProxyType({'scale_source': scale_source, 'zero_group_count': int(zero.sum()),
                                 'scale_initialization_zero_group_count': None if scale_zero is None else int(scale_zero.sum()),
                                 'output_coordinates': 'storage_Q(U)', 'rotation_scope': 'full_last_axis',
                                 'forward_realization': 'supplied_ordinals' if supplied is not None else 'nearest_even',
                                 'reference_label': 'project_S14_refit' if name.startswith('quest') else 'project_S14_max',
                                 'trust_threshold': 'physical_scale_s', 'fixed_scale_owner': 'caller',
                                 'stochastic_generation': 'reserved_M05; no_rng_draws'})
    return QuantizationContext(scales, codes.detach().clone(), values.detach().clone(), u,
                               mask.detach().clone(), zero.detach().clone(), scale_zero, rotation, grouping,
                               backward, name, step, metadata)


def quantized_view(w: Tensor, context: QuantizationContext) -> Tensor:
    """Return storage Q(W@R), rebuilding the W-to-storage surrogate graph.

    Cached forward values and masks stay frozen until caller prepares the next
    logical-step context. Fresh graph backpropagates into original W through
    R transpose. No retained graph or optimizer-history mutation is required.
    """
    if w.shape != context.storage_u.shape or w.device != context.storage_u.device:
        raise ValueError('Weight view must match context shape/device')
    u = to_storage(w, context.rotation)
    return surrogate_view(u, context.storage_values, context.gradient_mask)
