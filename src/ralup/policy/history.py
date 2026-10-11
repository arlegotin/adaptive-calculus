"""One immutable cell-history commit per logical update; refresh is a resolution.

Codes are numerical uint8 ordinals. Both prior scale arrays are retained: code
identity alone is never evidence of a reversal when physical endpoints move.
"""
from dataclasses import dataclass, replace, fields
import math
import torch
from torch import Tensor
from ..quantization.formats import quantize_e2m1, _preserved_reconstruction
from ..quantization.fp32 import bits
from ..quantization.rotation import to_storage


@dataclass(frozen=True)
class LossObservation:
    old_ema: float | None
    loss: float

    def __post_init__(self):
        if not math.isfinite(self.loss) or (self.old_ema is not None and not math.isfinite(self.old_ema)):
            raise ValueError('Loss observation must be finite training loss')

    @property
    def prospective_ema(self):
        return self.loss if self.old_ema is None else .95*self.old_ema+.05*self.loss

    @property
    def relative_change(self):
        if self.old_ema is None:return 0.
        return max(-1.,min(1.,(self.prospective_ema-self.old_ema)/max(abs(self.old_ema),1e-6)))


@dataclass(frozen=True)
class MatrixHistory:
    previous_codes: Tensor | None = None
    previous_scales: Tensor | None = None
    two_steps_ago_codes: Tensor | None = None
    two_steps_ago_scales: Tensor | None = None
    cached_codes: Tensor | None = None
    transition_fraction: float = 0.
    reversal_fraction: float = 0.
    disagreement_fraction: float = 0.
    reversal_coverage: float = 0.
    numerical_displacement: float = 0.
    refresh_endpoint_displacement: float = 0.
    reversal_valid: bool = False
    realized_valid: bool = False
    pending: bool = False
    last_step: int = -1
    updates: int = 0
    common_gradient_ema: Tensor | None = None
    last_features: Tensor | None = None

    @property
    def tensor_bytes(self):
        return sum(v.numel()*v.element_size() for f in fields(self) if isinstance(v:=getattr(self,f.name),Tensor))


def _endpoints(codes,scales):
    return _preserved_reconstruction(codes.long(),bits(scales))


@torch.no_grad()
def commit_history(history: MatrixHistory, before: Tensor, after: Tensor, context) -> MatrixHistory:
    """before/after are actual original-W endpoints, never decay-only references."""
    if type(history) is not MatrixHistory:raise ValueError('Registered MatrixHistory required')
    if context.step<=history.last_step or history.pending:
        raise ValueError('History commit requires a new logical step and resolved pending refresh')
    if before.shape!=context.storage_u.shape or after.shape!=before.shape:
        raise ValueError('History endpoints must match context matrix')
    if not torch.equal(to_storage(before.detach(),context.rotation),context.storage_u):
        raise ValueError('History before must be the actual context weights')
    _,cached=quantize_e2m1(to_storage(after.detach(),context.rotation),context.scales)
    current=context.codes
    eligible=torch.zeros_like(current,dtype=torch.bool)
    valid=history.previous_codes is not None
    if valid:
        eligible=torch.broadcast_to(context.scales==history.previous_scales,current.shape)
    count=int(eligible.sum())
    # Completed cached POST endpoint q(t+1) vs actual prepared q(t) and prior
    # prepared q(t-1). At the next feature collection this is previous-update
    # reversal; refresh cannot replace it with next-forward disagreement.
    reversed_mask=(cached==history.previous_codes)&(cached!=current) if valid else eligible
    reversal=float((reversed_mask&eligible).sum())/count if count else 0.
    return replace(history,previous_codes=current.detach().clone(),previous_scales=context.scales.detach().clone(),
                   two_steps_ago_codes=history.previous_codes,two_steps_ago_scales=history.previous_scales,
                   cached_codes=cached,transition_fraction=float((cached!=current).float().mean()),
                   reversal_fraction=reversal,reversal_coverage=count/current.numel(),reversal_valid=valid,
                   numerical_displacement=float((_endpoints(cached,context.scales)-context.storage_values).abs().mean()),
                   pending=True,last_step=context.step,updates=history.updates+1)


@torch.no_grad()
def resolve_realized_history(history: MatrixHistory, next_context) -> MatrixHistory:
    """Next-forward and cached codes refer to the same post-step W.

    Preparation happens before any next update. It resolves pending metadata,
    retaining the two old pre-step histories and leaving the commit count alone.
    """
    if not history.pending:return history
    if next_context.step!=history.last_step+1:raise ValueError('Pending history requires the immediate next forward context')
    if next_context.codes.shape!=history.cached_codes.shape:raise ValueError('Pending history shape mismatch')
    return replace(history,disagreement_fraction=float((history.cached_codes!=next_context.codes).float().mean()),
                   refresh_endpoint_displacement=float((_endpoints(history.cached_codes,history.previous_scales)-next_context.storage_values).abs().mean()),
                   realized_valid=True,pending=False)
