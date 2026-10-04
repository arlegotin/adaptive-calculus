"""Independent Muon+ v3 equations, serial stabilization in original orientation.

Paper: arXiv:2602.21545v3, Eq.4/Algorithm1/Appendices C,D,E. No author
implementation/software grant verified: this is an equation implementation,
not copied paper pseudocode or a release reproduction. FP32 polar arithmetic,
zero initialization and pre-displacement decoupled decay are project choices.
"""
import math

import torch

from .base import Proposal
from .muon import MuonUpdateRule, orthogonalize_five_step

ORDERS = ('none','row','col','row_col','col_row')


def normalize_polar(matrix, order):
    if order not in ORDERS:
        raise ValueError('Unknown Muon+ normalization order')
    result = matrix.clone()
    for stage in (() if order == 'none' else order.split('_')):
        axis = -1 if stage == 'row' else -2
        result = result / torch.sqrt(result.square().sum(axis,keepdim=True) + 1e-8)
    return result


class MuonPlusUpdateRule(MuonUpdateRule):
    state_schema = 'muon_plus_fp32.v1'
    defaults = {'momentum':.95,'nesterov':False,'normalization':'row',
                'auxiliary_lr':.001,'auxiliary_betas':(.9,.99),'auxiliary_eps':1e-8}

    def __init__(self, config: dict):
        super().__init__(config)
        if self.config['nesterov'] is not False:
            raise ValueError('Muon+ paper comparison uses EMA without Nesterov')
        if self.config['normalization'] not in ORDERS:
            raise ValueError('Unknown Muon+ normalization order')

    @torch.no_grad()
    def propose(self, w, grad, state, schedule):
        beta = self.config['momentum']
        moment = self._moment(w,grad,state).mul_(beta).add_(grad,alpha=1-beta)
        polar = normalize_polar(orthogonalize_five_step(moment),self.config['normalization'])
        displacement = -schedule.lr*math.sqrt(w.shape[0]/w.shape[1])*polar
        reference = w.detach().clone().mul_(1-schedule.lr*schedule.weight_decay)
        return Proposal(reference,displacement,{'momentum':moment})
