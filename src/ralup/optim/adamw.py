"""Pinned Torch 2.14.1 scalar AdamW primitives in original W coordinates.

Supported path: FP32, Python scalar LR/betas, foreach=False, fused=False,
amsgrad=False, maximize=False, capturable=False, differentiable=False.
The installed AdamW delegates to adam._single_tensor_adam with decoupled decay.
"""
import math
from typing import Mapping

import torch
from torch import Tensor

from .base import BaseUpdateRule, Proposal, SchedulePoint


class AdamWUpdateRule(BaseUpdateRule):
    state_schema = 'adamw.v1'
    optimizer_state_fields = ('step','exp_avg','exp_avg_sq')

    def __init__(self, *, betas: tuple[float,float] = (.9,.99), eps: float = 1e-8):
        if len(betas) != 2 or not all(math.isfinite(beta) and 0 <= beta < 1 for beta in betas):
            raise ValueError('AdamW betas must be finite in [0,1)')
        if not math.isfinite(eps) or eps < 0:
            raise ValueError('AdamW epsilon must be finite and nonnegative')
        self.betas = betas
        self.eps = eps

    @property
    def configuration(self) -> dict:
        return {'betas':self.betas,'eps':self.eps,'foreach':False,'fused':False,
                'amsgrad':False,'maximize':False,'capturable':False,'differentiable':False}

    @torch.no_grad()
    def propose(self, w: Tensor, grad: Tensor, state: Mapping[str,Tensor], schedule: SchedulePoint) -> Proposal:
        if w.dtype != torch.float32 or grad.dtype != torch.float32 or w.shape != grad.shape or w.device != grad.device or grad.is_sparse:
            raise ValueError('AdamW scalar adapter requires matching dense FP32 W and gradient')
        if state and set(state) != set(self.optimizer_state_fields):
            raise ValueError('AdamW state requires exactly step, exp_avg and exp_avg_sq')
        if state:
            step = state['step']
            if step.shape != torch.Size([]) or step.device.type != 'cpu' or step.dtype != torch.float32:
                raise ValueError('Noncapturable AdamW step must be a CPU FP32 scalar')
            for name in ('exp_avg','exp_avg_sq'):
                if state[name].shape != w.shape or state[name].dtype != w.dtype or state[name].device != w.device:
                    raise ValueError('AdamW moments must match original W')
            new_state = {name:value.detach().clone() for name,value in state.items()}
        else:
            new_state = {'step':torch.tensor(0.,dtype=torch.float32,device='cpu'),
                         'exp_avg':torch.zeros_like(w,memory_format=torch.preserve_format),
                         'exp_avg_sq':torch.zeros_like(w,memory_format=torch.preserve_format)}
        beta1,beta2 = self.betas
        new_state['step'].add_(1)
        reference = w.detach().clone()
        if schedule.weight_decay != 0:
            reference.mul_(1-schedule.lr*schedule.weight_decay)
        moment,variance = new_state['exp_avg'],new_state['exp_avg_sq']
        moment.lerp_(grad,1-beta1)
        variance.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
        step = new_state['step'].item()
        step_size = schedule.lr/(1-beta1**step)
        denominator = (variance.sqrt()/((1-beta2**step)**.5)).add_(self.eps)
        # S16 mathematical telemetry; commit uses the upstream operands above.
        displacement = -schedule.lr*(moment/(1-beta1**step))/(torch.sqrt(variance/(1-beta2**step))+self.eps)
        return Proposal(reference,displacement,new_state,{'denominator':denominator,'step_size':step_size})

    @torch.no_grad()
    def commit(self, w: Tensor, proposal: Proposal) -> None:
        w.copy_(proposal.reference_after_decay)
        w.addcdiv_(proposal.new_optimizer_state['exp_avg'],proposal.authoritative_update['denominator'],
                   value=-proposal.authoritative_update['step_size'])
