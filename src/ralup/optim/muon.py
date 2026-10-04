"""Clean S23 project Muon equations, explicitly portable FP32 (not upstream BF16).

Project momentum is beta*M+G; optional Nesterov uses G+beta*M. This differs
from KellerJordan/Muon f98f1cacc0263b04290753e32be8d498c1efc806's EMA.
"""
import math
from types import MappingProxyType

import torch
from torch import Tensor

from .base import BaseUpdateRule, Proposal


def orthogonalize_five_step(matrix: Tensor) -> Tensor:
    """Five Jordan polynomial iterations; original tall orientation is restored.

    Arithmetic retains FP32 or CPU FP64 input dtype, with no BF16 conversion.
    This approximates a polar factor: it is not an SVD or exact orthogonality.
    """
    if matrix.ndim != 2 or not all(matrix.shape) or matrix.dtype not in (torch.float32,torch.float64) or matrix.is_sparse:
        raise ValueError('Polynomial requires a nonempty dense FP32/FP64 matrix')
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError('Polynomial matrix must be finite')
    tall = matrix.shape[0] > matrix.shape[1]
    x = matrix.T.clone() if tall else matrix.clone()
    x = x / (torch.linalg.vector_norm(x) + 1e-7)
    for _ in range(5):
        a = x @ x.T
        # Literal S23 associativity, not an algebraically regrouped polynomial.
        x = 3.4445*x - 4.775*(a@x) + 2.0315*((a@a)@x)
    return x.T.contiguous() if tall else x


class MuonUpdateRule(BaseUpdateRule):
    state_schema = 'muon_project_fp32.v1'
    optimizer_state_fields = ('momentum',)
    defaults = {'momentum':.95,'nesterov':True,'auxiliary_lr':.001,
                'auxiliary_betas':(.9,.99),'auxiliary_eps':1e-8}

    def __init__(self, config: dict):
        if not isinstance(config,dict) or set(config)-set(self.defaults):
            raise ValueError('Unknown Muon configuration')
        settings = {**self.defaults,**config}
        beta = settings['momentum']
        if type(beta) not in (int,float) or not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError('Muon momentum must be finite in [0,1)')
        if type(settings['nesterov']) is not bool:
            raise ValueError('Nesterov setting must be boolean')
        lr = settings['auxiliary_lr']
        if type(lr) not in (int,float) or not math.isfinite(lr) or lr < 0:
            raise ValueError('Auxiliary LR must be finite and nonnegative')
        from .adamw import AdamWUpdateRule
        auxiliary = AdamWUpdateRule(betas=tuple(settings['auxiliary_betas']),eps=settings['auxiliary_eps'])
        settings['auxiliary_betas'] = auxiliary.betas
        self.config = MappingProxyType(settings)

    @property
    def configuration(self):
        return dict(self.config)

    @property
    def auxiliary_rule(self):
        from .adamw import AdamWUpdateRule
        return AdamWUpdateRule(betas=self.config['auxiliary_betas'],eps=self.config['auxiliary_eps'])

    def _moment(self, w, grad, state):
        if w.ndim != 2 or not all(w.shape) or w.dtype != torch.float32 or grad.dtype != w.dtype or w.shape != grad.shape or w.device != grad.device or grad.is_sparse:
            raise ValueError('Muon requires matching dense original-W FP32 matrices')
        if not bool(torch.isfinite(w).all()) or not bool(torch.isfinite(grad).all()):
            raise ValueError('Muon W/gradient must be finite')
        if state:
            if set(state) != {'momentum'}:
                raise ValueError('Muon state requires exactly original-W momentum')
            m = state['momentum']
            if m.shape != w.shape or m.dtype != w.dtype or m.device != w.device or not bool(torch.isfinite(m).all()):
                raise ValueError('Muon momentum must match original W')
            return m.detach().clone()
        return torch.zeros_like(w)

    @torch.no_grad()
    def propose(self, w, grad, state, schedule):
        beta = self.config['momentum']
        moment = self._moment(w,grad,state).mul_(beta).add_(grad)
        polar_input = grad + beta*moment if self.config['nesterov'] else moment
        polar = orthogonalize_five_step(polar_input)
        displacement = -schedule.lr*math.sqrt(max(1,w.shape[0]/w.shape[1]))*polar
        reference = w.detach().clone().mul_(1-schedule.lr*schedule.weight_decay)
        return Proposal(reference,displacement,{'momentum':moment})

    @torch.no_grad()
    def commit(self, w, proposal):
        w.copy_(proposal.reference_after_decay)
        w.add_(proposal.gradient_displacement)
