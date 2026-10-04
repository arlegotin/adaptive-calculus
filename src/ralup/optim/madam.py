"""Independent S23 sign-restricted Madam equations; no author source copied.

Paper arXiv:2006.14560v2; release parameter choices .999, scale 3, clip 10.
Author release c2d23592 is CC BY-NC-SA 4.0 and remains source-copy blocked.
"""
import math
from types import MappingProxyType
import torch
from .base import BaseUpdateRule,Proposal
from .adamw import AdamWUpdateRule


class MadamUpdateRule(BaseUpdateRule):
    state_schema='madam_clean_equations.v1'
    optimizer_state_fields=('step','exp_avg_sq','bound')
    defaults={'beta':.999,'p_scale':3.,'g_bound':10.,'auxiliary_lr':.001,'auxiliary_betas':(.9,.99),'auxiliary_eps':1e-8}
    def __init__(self,config:dict):
        if type(config) is not dict or set(config)-self.defaults.keys():raise ValueError('Unknown Madam configuration')
        settings={**self.defaults,**config}
        for key in ('beta','p_scale','g_bound','auxiliary_lr'):
            v=settings[key]
            if type(v) not in (int,float) or not math.isfinite(v) or (not 0<=v<1 if key=='beta' else v<0 if key=='auxiliary_lr' else v<=0):raise ValueError('Invalid Madam setting')
        aux=AdamWUpdateRule(betas=tuple(settings['auxiliary_betas']),eps=settings['auxiliary_eps']);settings['auxiliary_betas']=aux.betas
        self.config=MappingProxyType(settings)
    @property
    def configuration(self):return dict(self.config)
    @property
    def auxiliary_rule(self):return AdamWUpdateRule(betas=self.config['auxiliary_betas'],eps=self.config['auxiliary_eps'])
    def validate_state(self,state,w):
        if set(state)!=set(self.optimizer_state_fields) or type(state['step']) is not int or state['step']<1:raise ValueError('Incomplete Madam state/counter')
        for key,shape in [('exp_avg_sq',w.shape),('bound',torch.Size([]))]:
            t=state[key]
            if not isinstance(t,torch.Tensor) or t.shape!=shape or t.dtype!=w.dtype or t.device!=w.device or not bool(torch.isfinite(t).all()) or bool((t<0).any()):raise ValueError('Invalid Madam moment/bound geometry')
    @torch.no_grad()
    def propose(self,w,grad,state,schedule):
        if schedule.weight_decay!=0:raise ValueError('Sign-restricted Madam has no weight decay')
        if w.dtype!=torch.float32 or w.layout!=torch.strided or grad.is_sparse or not w.numel() or grad.shape!=w.shape or grad.dtype!=w.dtype or grad.device!=w.device:raise ValueError('Madam requires matching dense FP32 tensors')
        if not bool(torch.isfinite(w).all()) or not bool(torch.isfinite(grad).all()):raise ValueError('Madam W/gradient must be finite')
        if state:self.validate_state(state,w)
        step=state['step']+1 if state else 1
        bound=state['bound'].detach().clone() if state else self.config['p_scale']*w.detach().square().mean().sqrt()
        v=state['exp_avg_sq'].detach().clone() if state else torch.zeros_like(w)
        beta=self.config['beta'];v.mul_(beta).add_(grad.square(),alpha=1-beta)
        normalized=grad/(v/(1-beta**step)).sqrt()
        # No epsilon: define 0/0 as zero, then explicitly clip infinities too.
        normalized=torch.where(torch.isnan(normalized),torch.zeros_like(normalized),normalized).clamp(-self.config['g_bound'],self.config['g_bound'])
        candidate=w.detach().clone().mul_(torch.exp(-schedule.lr*w.sign()*normalized))
        candidate.clamp_(-bound,bound)
        return Proposal(w.detach().clone(),candidate-w.detach(),{'step':step,'exp_avg_sq':v,'bound':bound},{'candidate':candidate})
    @torch.no_grad()
    def commit(self,w,proposal):w.copy_(proposal.authoritative_update['candidate'])
