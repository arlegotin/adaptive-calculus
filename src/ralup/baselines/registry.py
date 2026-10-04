"""S22 named source/adaptation provenance and honest claim eligibility."""
from dataclasses import dataclass
from typing import Mapping

import torch

from ..optim.base import ParameterGroup
from ..optim.adamw import AdamWUpdateRule


@dataclass(frozen=True)
class MethodRecord:
    name: str
    family: str
    source_revision: str
    adaptation_label: str
    required_operations: tuple[str,...]
    ordered_parameter_groups: tuple[ParameterGroup,...]
    state_schema: Mapping[str,str]
    licenses: tuple[str,...]
    update_rule_settings: Mapping[str,object]
    parity_status: str
    artifact_kind: str
    operations_status: str = 'unverified'
    license_status: str = 'permitted'
    extra_passes: tuple[str,...] = ()
    schedule_fields: tuple[str,...] = ('lr','weight_decay','progress')

    def __post_init__(self):
        for name in ('name','family','source_revision','adaptation_label','required_operations','ordered_parameter_groups','state_schema','licenses','update_rule_settings'):
            if not getattr(self,name):
                raise ValueError(f'Method requires {name}')
        if self.parity_status not in ('unverified','pass','fail','blocked') or self.operations_status not in ('unverified','pass','blocked'):
            raise ValueError('Unknown parity or operations status')
        if self.license_status not in ('permitted','blocked'):
            raise ValueError('Unknown license status')
        if self.artifact_kind not in ('original_fixture','paper_reference','project_adaptation'):
            raise ValueError('Method kind must distinguish fixtures, papers and adaptations')
        required = {'optimizer_states','proposals','adapter_mutable_state'}
        if not required <= set(self.state_schema) or not all(self.state_schema.values()):
            raise ValueError('Method schema must name all optimizer, proposal and adapter mutable state')


class MethodRegistry:
    def __init__(self):
        self._methods: dict[str,MethodRecord] = {}

    def register(self, method: MethodRecord) -> None:
        if method.name in self._methods:
            raise ValueError(f'Method already registered: {method.name}')
        self._methods[method.name] = method

    def require(self, name: str, *, adaptation_label: str | None = None) -> MethodRecord:
        if name not in self._methods:
            raise ValueError(f'Method unregistered: {name}')
        method = self._methods[name]
        if not adaptation_label or adaptation_label != method.adaptation_label:
            raise ValueError(f'Explicit matching adaptation label required: {name}')
        return method

    def claim_eligibility(self, name: str) -> dict:
        if name not in self._methods:
            raise ValueError(f'Method unregistered: {name}')
        method = self._methods[name]
        blockers = []
        if method.artifact_kind == 'original_fixture':
            blockers.append('original fixture parity does not authorize scientific adaptation claims')
        if method.license_status != 'permitted':
            blockers.append('license reuse blocked')
        if method.operations_status != 'pass':
            blockers.append(f'operations {method.operations_status}')
        if method.parity_status != 'pass':
            blockers.append(f'parity {method.parity_status}')
        return {'eligible':not blockers,'blockers':blockers,'blocked_is_defeated_baseline':False}


def default_registry(groups: tuple[ParameterGroup,...], *, rule: AdamWUpdateRule | None = None, parity_status: str = 'unverified', operations_status: str = 'unverified') -> MethodRegistry:
    rule = AdamWUpdateRule() if rule is None else rule
    registry = MethodRegistry()
    registry.register(MethodRecord(
        name='adamw_scalar_fp32',family='AdamW',source_revision=f'torch=={torch.__version__};git={torch.version.git_version}',
        adaptation_label='original_W_scalar_fixture',required_operations=('mul_','lerp_','addcmul_','sqrt','div','add_','addcdiv_'),
        ordered_parameter_groups=groups,state_schema={'optimizer_states':'named.original_W.adamw.v1',
            'proposals':'named.proposal.v1:reference_after_decay,gradient_displacement,new_optimizer_state,authoritative_update',
            'adapter_mutable_state':'none.v1'},licenses=('PyTorch BSD-3-Clause; https://github.com/pytorch/pytorch/blob/main/LICENSE',),
        update_rule_settings=dict(rule.configuration),parity_status=parity_status,operations_status=operations_status,artifact_kind='original_fixture'))
    return registry
