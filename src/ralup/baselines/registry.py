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
    register_matrix_methods(registry,groups)
    register_ewgs_methods(registry,groups)
    register_trs_methods(registry,groups)
    register_routing_methods(registry,groups)
    register_scale_noise_methods(registry,groups)
    return registry


MATRIX_NAMES = ('muon_project_fp32',) + tuple(f'muon_plus_{order}_fp32' for order in ('none','row','col','row_col','col_row'))


def matrix_rule(name, config):
    from ..optim.muon import MuonUpdateRule
    from ..optim.muon_plus import MuonPlusUpdateRule
    from ..optim.soap import SoapUpdateRule
    from ..optim.madam import MadamUpdateRule
    if name == 'soap_pinned_fp32': return SoapUpdateRule(config)
    if name == 'madam_clean_fp32': return MadamUpdateRule(config)
    if name == 'muon_project_fp32': return MuonUpdateRule(config)
    if name not in MATRIX_NAMES: raise ValueError(f'Unregistered matrix method: {name}')
    order = name.removeprefix('muon_plus_').removesuffix('_fp32')
    if 'normalization' in config and config['normalization'] != order:
        raise ValueError('Named Muon+ variant must match normalization')
    return MuonPlusUpdateRule({**config,'normalization':order})


def matrix_record(name, groups, rule, *, parity_status='unverified', operations_status='unverified'):
    if name in SOAP_MADAM_NAMES: return soap_madam_record(name,groups,rule,parity_status=parity_status,operations_status=operations_status)
    plus = name != 'muon_project_fp32'
    order = rule.configuration.get('normalization')
    return MethodRecord(name=name,family='Muon+' if plus else 'Muon',
        source_revision='arXiv:2602.21545v3;html-sha256=10536d1f1921f61f8f427c196990ca1855aef3a414be28a4be594bfacd0a2679' if plus else 'KellerJordan/Muon@f98f1cacc0263b04290753e32be8d498c1efc806;project momentum departure',
        adaptation_label=f'paper_muon_plus_fp32_{order}_original_W' if plus else 'project_muon_fp32_original_W',
        required_operations=('transpose','frobenius_norm','matmul','mul','add','sum','sqrt','div','auxiliary_scalar_adamw'),
        ordered_parameter_groups=groups,
        state_schema={'optimizer_states':f'named.original_W.{rule.state_schema}+auxiliary.adamw.v1',
            'proposals':'named.proposal.v1:reference_after_decay,gradient_displacement,new_optimizer_state,authoritative_update',
            'adapter_mutable_state':'none.v1'},
        licenses=('arXiv distribution grant; author software reuse grant unresolved; clean equation implementation',) if plus else ('MIT; Keller Jordan; independently implemented equations',),
        update_rule_settings={**rule.configuration,'polynomial_dtype':'float32','iterations':5,'coefficients':(3.4445,-4.775,2.0315),
            'norm_stabilizer':1e-7,'transpose_when':'original_rows>original_columns','normalization_epsilon_inside_sqrt':1e-8 if plus else None,
            'rectangular_multiplier':'sqrt(rows/cols)' if plus else 'sqrt(max(1,rows/cols))','initialization':'zero original-W momentum',
            'decay':'decoupled before displacement; auxiliary no decay'},
        parity_status=parity_status,operations_status=operations_status,artifact_kind='project_adaptation',
        license_status='blocked' if plus else 'permitted')


def register_matrix_methods(registry, groups):
    for name in (*MATRIX_NAMES,*SOAP_MADAM_NAMES):
        registry.register(matrix_record(name,groups,matrix_rule(name,{})))


def build_matrix_method(name, parameters, groups, config):
    from ..optim.matrix_adapter import MatrixMethodAdapter
    return MatrixMethodAdapter(parameters,groups,matrix_rule(name,config))


SOAP_MADAM_NAMES = ('soap_pinned_fp32','madam_clean_fp32')


def soap_madam_record(name, groups, rule, *, parity_status='unverified', operations_status='unverified'):
    soap=name=='soap_pinned_fp32'
    return MethodRecord(name=name,family='SOAP' if soap else 'Madam',
        source_revision='nikhilvyas/SOAP@a1e553530fde97d0e6b307d7c82ac6d38b072340;sha256=9e9021335ad02371584622fa256e73fe75f3eb74446a8d8a216e956534148f6b' if soap else 'arXiv:2006.14560v2;independent equations;release settings c2d23592c5e18d6c9e6fcb2a5213ac288a6153cc',
        adaptation_label='pinned_SOAP_CPU_factorization_original_W' if soap else 'clean_Madam_sign_restricted_original_W',
        required_operations=('tensordot','eigh_cpu','qr_cpu','argsort','index_select','cpu_device_transfer','auxiliary_scalar_adamw') if soap else ('square','mean','sqrt','exp','clamp','auxiliary_scalar_adamw'),
        ordered_parameter_groups=groups,state_schema={'optimizer_states':f'named.original_W.{rule.state_schema}+auxiliary.adamw.v1',
            'proposals':'named.proposal.v1:reference_after_decay,gradient_displacement,new_optimizer_state,authoritative_update',
            'adapter_mutable_state':'none.v1'},
        licenses=('MIT; Copyright (c) 2024 Nikhil Vyas; licenses/methods/soap/LICENSE',) if soap else ('Independent mathematical implementation; author source CC BY-NC-SA 4.0 not copied; no legal assurance',),
        update_rule_settings={**rule.configuration,
            'source_copy_status':'permitted_MIT' if soap else 'blocked',
            'release_reproduction_status':'unverified',
            'equation_implementation_status':'pinned_mechanics' if soap else 'independent_local_research',
            'first_observation':'initialize factors/basis only; no decay/moment/update' if soap else 'fixed initial p_scale*RMS(W)',
            'decay':'post-displacement; initialization skips; auxiliary no decay' if soap else 'none; matrix schedule decay disabled; auxiliary no decay',
            'factorization':'explicit CPU eig/QR with timed and byte-counted transfers' if soap else 'none',
            'refresh_screening_candidates':(1,4,10) if soap else (),
            'scope':'bounded mechanics and local research only; no universal replay or scientific eligibility'},
        parity_status=parity_status,operations_status=operations_status,artifact_kind='project_adaptation',license_status='permitted')


EWGS_NAMES=('ewgs_uniform_fixture','ewgs_fp4')


def ewgs_record(name,groups,config,*,parity_status='unverified',operations_status='unverified'):
    from .ewgs import validate_config
    config=validate_config(config)
    if name not in EWGS_NAMES:raise ValueError('Unknown EWGS method')
    original=name=='ewgs_uniform_fixture'
    return MethodRecord(name=name,family='EWGS',source_revision='arXiv:2104.00903v1;cvlab-yonsei/EWGS@56c654cb893d53563eb352dd591d1450c34bdd15;inspection-only',
        adaptation_label='original_uniform_primitive_clean_equations' if original else 'project_FP4_population_std_detached_normalized_leaf',
        required_operations=('sign','mul','clamp')+(() if config['mode']=='fixed' else ('normalized_leaf_gradient','higher_order_autograd','rademacher_hvp')),
        ordered_parameter_groups=groups,state_schema={'optimizer_states':'named.original_W.adamw.v1','proposals':'named.proposal.v1','adapter_mutable_state':'ewgs.normalized_leaf_probes_cadence.v2'},
        licenses=('Independent published equations; source GPL-v3 inspection only; no software grant or release parity inferred',),
        update_rule_settings={**config,'source_copy_status':'blocked_GPL_inspection_only','source_inspection_sha256':{'CIFAR10/custom_modules.py':'8a823e02a8f84c0ec4828fde521e6c4a1d32dfa02b48d233d42dbae9f26e21d0','CIFAR10/utils.py':'16addc01e0e7bff6cec0092fe6b23c07ab04093cfd9cd814bbcac602e2abab21','LICENSE':'3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986'},'equation_implementation_status':'independent_local_research','release_reproduction_status':'unverified',
            'source_primitive':'[0,1];round(x*(L-1))/(L-1)', 'departures':() if original else ('FP4 storage coordinates','population std instead of release sample std','epsilon 1e-12','detached normalized leaf','fixed-count probes; one logical batch instead of three release batches')},
        parity_status=parity_status,operations_status=operations_status,artifact_kind='original_fixture' if original else 'project_adaptation',extra_passes=() if config['mode']=='fixed' else ('extra_loss_and_its_backward','hvp'))


def register_ewgs_methods(registry,groups):
    from .ewgs import FIXTURE_CONFIG
    for name in EWGS_NAMES:registry.register(ewgs_record(name,groups,FIXTURE_CONFIG))


TRS_NAMES=('trs_release_fixture','trs_backbone_adapter')


def trs_record(name,groups,config,*,parity_status='unverified',operations_status='unverified'):
    from .trs import validate_config
    if name not in TRS_NAMES:raise ValueError('Unknown TRS method')
    original=name=='trs_release_fixture';config=validate_config(config)
    return MethodRecord(name=name,family='TRS',
        source_revision='arXiv:2404.19248v4;cvlab-yonsei/TRS@958dedd1baeb60814cb4162ce794bc9f41cdf4a0;inspection-only',
        adaptation_label='released_multiplicative_coupled_Adam_clean_equations' if original else 'project_FP4_logical_step_decoupled_backbone',
        required_operations=('round','clamp','mean','mul','add','normalized_quantized_compare','original_W_backbone'),
        ordered_parameter_groups=groups,state_schema={'optimizer_states':'named.original_W.coupled_adam.v1' if original else 'named.original_W.wrapped_backbone+auxiliary.adamw.v1',
            'proposals':'named.proposal.v1','adapter_mutable_state':'fixed_scale+trs.normalized_codes_scalars.v1' if original else 'trs.normalized_codes_scalars.v1'},
        licenses=('Independent mathematical implementation; GPL-v3 author source inspection only; no copying/software grant or release reproduction assurance',),
        update_rule_settings={**config,'supported_backbone_families':['AdamW','Muon','Muon+'],'unsupported_backbone_families':['SOAP','Madam','arbitrary unregistered BaseUpdateRule'],'compatibility_scope':'explicit original-W families only; no restriction on future B0 selection; unsupported composition needs later owned implementation/evidence', 'release_feedback':'K=mu*K+(1-mu)*k; r=mu*r+(1-mu)*r*R/(K+1e-8); alpha*r',
            'paper_feedback':'additive max(0,U+eta*(R-K)); not implemented as release',
            'source_inspection_sha256':{'custom_optims.py':'a4f4195f4325cf65f84abab3e86960730699aab956531586c1264d9bd6d9e4cb','custom_modules.py':'efe995b17931b4f70e619aa0c51c730f1cc43c07e9a0989806595d1ac3b24413','train_quant.py':'fd5e92d427539e06e0ed58d6a657804671545d9c7a407a936f5f9a7fe1002aa2','LICENSE':'2a2973fb4c7b3129e05fe5f5746af28a8246ea706bf694994b8ceeec245cdb21'},
            'source_copy_status':'blocked_GPL_inspection_only','equation_implementation_status':'independent_local_research','release_reproduction_status':'unverified',
            'initialization':'r=1; K=first target_rate; previous normalized values zero; first rate ignored',
            'scale_contract':'fixed 3*sample_std(W); normalization abs(sW)+1e-6; signed nonbinary round/clamp levels divided by 2**(bits-1)' if original else 'fixed row max initial scale cached by H06; normalized storage_values/scales/6; numerical signed zero merged; storage rotation follows context',
            'decay':'coupled gradient before Adam moments' if original else 'ordinary lr pre-displacement reference decay; feedback applies to gradient displacement only',
            'departures':[] if original else ['FP4 E2M1','fixed row-max scale instead of tensor sample-std scale','storage-coordinate normalized levels divided by 6','original-W selectable AdamW/Muon/Muon+ backbone','decoupled ordinary-LR decay','once per logical batch instead of per training forward'],
            'scope':'bounded independently reproduced released equations; no copied release execution, author grant, pilot relevance or science selection'},
        parity_status=parity_status,operations_status=operations_status,artifact_kind='original_fixture' if original else 'project_adaptation',
        schedule_fields=('lr','weight_decay','progress','target_rate'))


def register_trs_methods(registry,groups):
    from .trs import FIXTURE_CONFIG
    for name in TRS_NAMES:registry.register(trs_record(name,groups,FIXTURE_CONFIG))


ROUTING_NAMES=('qar_uniform_fixture','qar_fp4_adjacent','stochastic_shadow_forward','stochastic_endpoint')


def routing_record(name,groups,config,*,parity_status='unverified',operations_status='unverified'):
    from .qar import validate_config,SOURCE_CLARIFICATION
    if name not in ROUTING_NAMES:raise ValueError('Unknown routing/rounding method')
    rounding=name.startswith('stochastic_');c=validate_config(config,rounding=rounding)
    if rounding and c['mode']!=('shadow_forward' if name=='stochastic_shadow_forward' else 'endpoint'):raise ValueError('Method name must match persistent state mode')
    original=name=='qar_uniform_fixture';shadow=name=='stochastic_shadow_forward'
    return MethodRecord(name=name,family='stochastic_rounding' if rounding else 'QAR',
        source_revision='arXiv:2609.21834v1;2026-09-18;Eq8 and Algorithm1;independent equations' if not rounding else 'S24 adjacent-level expectation equation;independent implementation',
        adaptation_label='source_masked_uniform_endpoint_clean_equations' if original else 'project_FP4_shadow_forward_stochastic' if shadow else 'project_FP4_endpoint_stochastic' if rounding else 'project_FP4_adjacent_endpoint_original_W',
        required_operations=('group_power_normalization','directional_neighbor','uniform_bernoulli','original_W_backbone') if not rounding else ('adjacent_bracket','uniform_draw','original_W_backbone'),
        ordered_parameter_groups=groups,state_schema={'optimizer_states':'named.original_W.wrapped_backbone+auxiliary.adamw.v1','proposals':'named.proposal.v1',
            'adapter_mutable_state':'none; caller owns uniform codes+spacing+moments' if original else 'rounding.explicit_endpoint_or_shadow.v1' if rounding else 'qar.endpoint_codes_scales.v1'},
        licenses=('Independent published equations; author software reuse grant unresolved; arXiv distribution grant is not a software grant; no source copying or legal assurance',),
        update_rule_settings={**c,'persistent_authority':'uniform_codes+spacing' if original else 'original_W_FP32_shadow' if shadow else 'FP4_codes+fixed_scales;W_reconstructed_mirror',
            'moment_coordinates':'original_W','source_clarification':SOURCE_CLARIFICATION,'source_copy_status':'no_author_code_copied;author_software_grant_unresolved',
            'equation_implementation_status':'independent_local_research','release_reproduction_status':'unverified','fp4_uniform_theory_transfer':False,
            'supported_backbone_families':['AdamW','Muon','Muon+'],'unsupported_compositions':['SOAP','Madam','arbitrary unregistered rule'],
            'compatibility_scope':'current explicit compositions only; no restriction on future B0 selection; no blocked comparator defeat',
            'context_choices':['fixed row-max pre-conversion warm grid','last-axis-row normalization','identity','full_hadamard power-of-two width','STE'],
            'rng':'independent named CPU quantizer stream; explicit uniform transfer; one sample per logical batch',
            'primary_evaluation':'deterministic nearest-even, read-only; stochastic expectation secondary only',
            'departures':[] if original else ['nonuniform FP4 physical adjacent gaps' if not rounding else 'FP4 level brackets','fixed row-max scale','original-W selectable backbone and decoupled decay','logical batch context cache','optional full-last-axis Hadamard'],
            'scope':'bounded equation, synthetic-engine and same-device replay engineering only; no universal MPS/model/horizon/science eligibility'},
        parity_status=parity_status,operations_status=operations_status,artifact_kind='original_fixture' if original else 'project_adaptation')


def register_routing_methods(registry,groups):
    from .qar import FIXTURE_CONFIG
    for name in ROUTING_NAMES:
        c={**FIXTURE_CONFIG}
        if name.startswith('stochastic_'):c['mode']='shadow_forward' if name=='stochastic_shadow_forward' else 'endpoint'
        registry.register(routing_record(name,groups,c))


SCALE_NOISE_NAMES=('lsq_uniform_fixture','learned_fp4_scale','damping_fp4','simple_matrix_mask','magma_matrix_mask')


def scale_noise_record(name,groups,config,*,parity_status='unverified',operations_status='unverified'):
    from .learned_scale import validate_config as scale_config,validate_view
    from .noise import validate_config as noise_config
    if name not in SCALE_NOISE_NAMES:raise ValueError('Unknown scale/noise method')
    fixture=name=='lsq_uniform_fixture';learned=name=='learned_fp4_scale';damping=name=='damping_fp4'
    c=validate_view(config) if fixture else scale_config(config) if learned else noise_config(config)
    if fixture and c['format']!='uniform':raise ValueError('Uniform LSQ fixture must name uniform levels')
    if not (fixture or learned) and c['mode']!={'damping_fp4':'damping','simple_matrix_mask':'simple','magma_matrix_mask':'magma'}[name]:raise ValueError('Noise method name/config mode mismatch')
    label={'lsq_uniform_fixture':'original_uniform_LSQ_analytic_surrogate','learned_fp4_scale':'project_FP4_log_scale_original_W',
           'damping_fp4':'project_FP4_stopgrad_damping','simple_matrix_mask':'project_matrix_SkipUpdate_gradient_only','magma_matrix_mask':'project_matrix_practical_Magma_gradient_only'}[name]
    source='arXiv:1902.08153v3;2020-05-07;Eq3,section2.2;clean equations' if fixture or learned else 'S24 independent damping equation' if damping else 'arXiv:2602.15322v1;Algorithm1,section3;html-sha256=ae119060283877a05e1a05b46be45ab916eaa7cf6f3a637764af07ecb9294744'
    return MethodRecord(name=name,family='LSQ' if fixture else 'learned_FP4_scale' if learned else 'damping' if damping else 'masking',
        source_revision=source,adaptation_label=label,required_operations=('nearest_even','exp','analytic_surrogate','sum_to_group','original_W_backbone') if fixture or learned else ('stopgrad','squared_norm','original_W_backbone') if damping else ('current_original_W_cosine','alignment_ema','cpu_bernoulli_half','original_W_backbone'),
        ordered_parameter_groups=groups,state_schema={'optimizer_states':'caller owned' if fixture else 'named.original_W.backbone+auxiliary.adamw.v1',
            'proposals':'named.proposal.v1','adapter_mutable_state':'caller log scale and gradient' if fixture else 'learned_scale.parameters_gradients_adam_normalizer.v1' if learned else 'noise.matrix_alignment_rng.v1'},
        licenses=('Independent local equation implementation; arXiv distribution license is not a software grant; author source not copied; no release or legal assurance',) if fixture or learned else ('Independent S24 project equation; no external implementation copied',) if damping else ('Article CC BY 4.0; attribution arXiv:2602.15322v1; article/legal hashes in docs/harness/m06-scale-noise.md; author software grant unresolved; clean equations',),
        update_rule_settings={**c,'gradient_target':'declared analytic surrogate, never hard-quantizer finite differences' if fixture or learned else 'original-W current gradient/moment',
            'moment_coordinates':'original_W','source_copy_status':'no external implementation copied','release_reproduction_status':'unverified',
            'mask_probability':None if fixture or learned or damping else .5,'practical_Magma_doubling':False,
            'scale_precision':'float32','scale_information':'training_targets_only',
            'departures':[] if fixture else ['FP4 nonuniform levels','positive log scale and independent scale Adam/LR','row or tensor grouping','max/Cmax initialization instead of LSQ mean-absolute initialization','bounded log scale and optional gradient clipping','identity storage with common byte decoder'] if learned else ['S24 stopgrad damping comparator, absent from masking paper','fixed FP4 grid','packing-independent penalty, same backward graph'] if damping else ['matrix projection scope','initial alignment .5','zero-vector cosine zero; scaled norms','dedicated CPU mask RNG','gradient-only masking; separate ordinary-LR decay','original-W AdamW/Muon/Muon+ integration'],
            'supported_backbone_families':[] if fixture else ['AdamW','Muon','Muon+'],'unsupported_backbone_families':[] if fixture else ['SOAP','Madam'],
            'scope':'bounded equations and synthetic same-device replay only; no author release execution, universal device/model/horizon parity, pilot relevance or protected/scientific selection'},
        parity_status=parity_status,operations_status=operations_status,artifact_kind='original_fixture' if fixture else 'project_adaptation',
        schedule_fields=('lr','weight_decay','progress','scale_lr') if learned else ('lr','weight_decay','progress'))


def register_scale_noise_methods(registry,groups):
    from .learned_scale import FIXTURE_CONFIG as SCALE_CONFIG,VIEW_CONTRACT
    from .noise import FIXTURE_CONFIG as NOISE_CONFIG
    for name in SCALE_NOISE_NAMES:
        config={**VIEW_CONTRACT,'format':'uniform','minimum':-8,'maximum':7} if name=='lsq_uniform_fixture' else SCALE_CONFIG if name=='learned_fp4_scale' else {**NOISE_CONFIG,'mode':{'damping_fp4':'damping','simple_matrix_mask':'simple','magma_matrix_mask':'magma'}[name]}
        registry.register(scale_noise_record(name,groups,config))
