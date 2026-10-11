"""M09 deployment: one shared feature scan, original-W moments/base identity.

The adapter holds only registered finite data. No external callback is run.
Auxiliary parameters retain their ordinary rule/LR/no-decay routing.
"""
import hashlib
import time
import torch
from .controllers import ControllerArtifact,ControllerState,Affine45,freeze_controller,evaluate_controller,authenticate,PolicyAction
from .normalizer import FrozenNormalizer
from .features import FeatureVector
from .actions import compose_action,_rms
from ..optim.base import MethodAdapter,SchedulePoint
from ..optim.adamw import AdamWUpdateRule
from ..optim.muon import MuonUpdateRule
from ..optim.muon_plus import MuonPlusUpdateRule
from ..optim.soap import SoapUpdateRule
from ..optim.madam import MadamUpdateRule
from ..quantization.rotation import to_storage,from_storage
from ..config import canonical_bytes

RULES=(AdamWUpdateRule,MuonUpdateRule,MuonPlusUpdateRule,SoapUpdateRule,MadamUpdateRule)
ADAPTER_FIELDS=('optimizer_states','proposals','snapshot_schema','controller_state','normalizer','normalizer_ref','artifact','deployment_sha256','last_actions')


def _normalizer_values(n):
    if type(n) is not FrozenNormalizer:raise ValueError('Registered selected normalizer required')
    # Recheck exposed array owners, not only constructor-time validation.
    if n.mean.shape!=(14,) or n.std.shape!=(14,) or n.mean.dtype!=torch.float32 or n.std.dtype!=torch.float32 or not bool(torch.isfinite(n.mean).all()&torch.isfinite(n.std).all()) or bool((n.std<1e-6).any()):raise ValueError('Frozen normalizer mutated')
    return {'mean':n.mean.cpu().tolist(),'std':n.std.cpu().tolist(),'source_manifest':n.source_manifest}


def deployment_digest(controller,normalizer,normalizer_ref,artifact):
    ControllerState(controller.family,controller.parameters)
    return hashlib.sha256(canonical_bytes({'controller':{'family':controller.family,'parameters':controller.parameters.cpu().tolist()},
        'normalizer':_normalizer_values(normalizer),'normalizer_ref':{'path':str(normalizer_ref.path.resolve()),'sha256':normalizer_ref.sha256},
        'artifact':None if artifact is None else artifact.to_json()})).hexdigest()


def validate_deployment(method,histories=None,configuration=None):
    c=method['controller_state'];n=method['normalizer'];ref=method['normalizer_ref'];artifact=method['artifact']
    if type(c) is not ControllerState:raise ValueError('Registered controller state required')
    data=authenticate(ref)
    if not isinstance(data,dict) or any(data.get(k)!=v for k,v in _normalizer_values(n).items()):raise ValueError('Selected authenticated normalizer differs from frozen deployment')
    if artifact is not None:
        if type(artifact) is not ControllerArtifact or artifact.family!=c.family or not torch.equal(artifact.parameters.cpu(),c.parameters.cpu()) or artifact.normalization!=ref:raise ValueError('Controller artifact/deployment mismatch')
        artifact.to_json()
    if method['deployment_sha256']!=deployment_digest(c,n,ref,artifact):raise ValueError('Frozen deployment bytes changed')
    if configuration is not None:
        cfg=configuration.get('policy_features')
        if type(cfg) is not dict or set(cfg)!={'enabled','full_horizon','trace_cadence'} or cfg.get('enabled') is not True or any(type(cfg.get(k)) is not int or cfg[k]<=0 for k in ('full_horizon','trace_cadence')):raise ValueError('Policy controls require enabled full feature/history mode')
        attached=histories.get('feature_normalizer')
        if type(attached) is not FrozenNormalizer or _normalizer_values(attached)!=_normalizer_values(n):raise ValueError('Selected frozen normalizer must be attached to shared history')


class PolicyAdapter(MethodAdapter):
    def __init__(self,base,controller,normalizer,*,normalizer_ref,family=None):
        if type(base) not in RULES:raise ValueError('Registered authoritative base rule required')
        self.rule=base;self.artifact=controller if type(controller) is ControllerArtifact else None
        if self.artifact is not None:
            if family is not None and family!=controller.family:raise ValueError('Artifact family cannot be replaced')
            self.controller_state=ControllerState(controller.family,controller.parameters)
        else:self.controller_state=freeze_controller(controller,family=family)
        self.normalizer=FrozenNormalizer(normalizer.mean,normalizer.std,normalizer.source_manifest)
        self.normalizer_ref=normalizer_ref
        self.deployment_sha256=deployment_digest(self.controller_state,self.normalizer,normalizer_ref,self.artifact)
        validate_deployment(vars(self))

    def bind(self,parameters,groups):
        if tuple(g.name for g in groups) not in (('internal_projection',),('internal_projection','auxiliary_no_decay')) or groups[0].weight_decay is not None or (len(groups)>1 and groups[1].weight_decay!=0.) or any(parameters[n].ndim!=2 for n in groups[0].parameter_names):raise ValueError('Explicit matrix manifest and no-decay auxiliaries required')
        MethodAdapter.__init__(self,parameters,groups,self.rule)
        if type(self.rule) is not AdamWUpdateRule:self.snapshot_schema['optimizer_states']=f'named.original_W.{self.rule.state_schema}+auxiliary.adamw.v1'
        self.snapshot_schema['adapter_mutable_state']='policy.frozen_controller_normalizer_actions.v1'
        self.last_actions={}
        return self

    def rule_for_parameter(self,name):
        return self.rule if name in self.groups[0].parameter_names or type(self.rule) is AdamWUpdateRule else self.rule.auxiliary_rule

    def schedule_for_group(self,group,schedule):
        if group.name=='auxiliary_no_decay' and type(self.rule) is not AdamWUpdateRule:return SchedulePoint(self.rule.configuration['auxiliary_lr'],0.,schedule.progress)
        if group.name=='internal_projection' and type(self.rule) is MadamUpdateRule:return SchedulePoint(schedule.lr,0.,schedule.progress)
        return super().schedule_for_group(group,schedule)

    @property
    def metadata(self):
        value=super().metadata
        value.update({'controller_family':self.controller_state.family,'controller_parameters':self.controller_state.parameters.numel(),
            'h_range':[.25,32 if self.controller_state.family=='learned_lr49' else 4],
            'source_information':'all14_normalized_causal_scalars_common_history_and_scan_costs',
            'feature_removal':'mask_after_full_scan','privileged_candidate_rms':self.controller_state.family=='norm_matched',
            'normalizer':{'path':str(self.normalizer_ref.path),'sha256':self.normalizer_ref.sha256},'frozen_deployment_sha256':self.deployment_sha256})
        return value

    def prepare_step(self,state,batch):
        super().prepare_step(state,batch)
        validate_deployment(vars(self),state.histories,state.configuration)
        if not state.model._recipe:raise ValueError('Policy requires explicit quantized storage contexts')

    @torch.no_grad()
    def finish_step(self,state,schedule):
        validate_deployment(vars(self),state.histories,state.configuration)
        proposals=self.collect_proposals(schedule,work_recorder=state.record_work)
        features=state.collect_policy_features(proposals,schedule)
        if set(features)!=set(self.groups[0].parameter_names):raise ValueError('Every matrix requires complete common features')
        start=time.perf_counter();actions={};info={};candidates={}
        for name,f in features.items():
            normalized,_=self.normalizer.normalize(f.values)
            action=evaluate_controller(self.controller_state,FeatureVector(normalized))
            actions[name]=action
            if action.identity:
                info[name]={'alpha':0.,'h':1.,'tau':1.,'identity_action':True,'original_W_commit':True,'assistance_bypassed':True}
                continue
            p=proposals[name];ctx=state.model._contexts[name]
            u0=to_storage(p.reference_after_decay,ctx.rotation);da=to_storage(p.gradient_displacement,ctx.rotation)
            candidate,details=compose_action(u0,da,action.alpha,action.h,action.tau,ctx)
            if self.controller_state.family=='norm_matched':
                # Privileged diagnostic: oracle RMS of the complete candidate move.
                oracle_rms=_rms(candidate-u0);base_rms=_rms(da)
                candidate=u0+(oracle_rms/base_rms*da if float(base_rms)>0 else da)
                details.update({'privileged_candidate_rms':True,'candidate_rms':float(oracle_rms),'additive_rms':float(_rms(candidate-u0))})
            candidates[name]=from_storage(candidate,ctx.rotation)
            if not bool(torch.isfinite(candidates[name]).all()):raise ValueError('Nonfinite mapped policy candidate')
            info[name]={'alpha':action.alpha,'h':action.h,'tau':action.tau,'original_W_commit':False,
                        **{k:v for k,v in details.items() if not isinstance(v,torch.Tensor)}}
        if all(a.identity for a in actions.values()):
            # Preserve the authoritative full arithmetic, including aux validation.
            result=self.commit_proposals(proposals)
        else:
            for name,p in proposals.items():
                if name not in candidates:
                    candidate=self.parameters[name].detach().clone();self.rule_for_parameter(name).commit(candidate,p)
                    if not bool(torch.isfinite(candidate).all()):raise ValueError('Nonfinite authoritative candidate')
            for name,p in proposals.items():
                if name in candidates:self.parameters[name].copy_(candidates[name])
                else:self.rule_for_parameter(name).commit(self.parameters[name],p)
            for name,p in proposals.items():self.optimizer_states[name]=p.new_optimizer_state
            self.proposals={};result={'optimizer_updates':len(proposals),'candidate_validation_updates':len(proposals),'extra_forward_passes':0,'extra_backward_passes':0}
        self.last_actions=info
        state._work['policy_seconds']=time.perf_counter()-start
        result['policy_actions']=info
        return result


def trusted_constructor(parameters,groups,rule,method):
    # Registered data-only reconstruction; no artifact-selected class/import.
    obj=PolicyAdapter.__new__(PolicyAdapter);obj.rule=rule
    for key in ('controller_state','normalizer','normalizer_ref','artifact','deployment_sha256'):setattr(obj,key,method[key])
    validate_deployment(vars(obj))
    return obj.bind(parameters,groups)


def validate_owned_state(method,groups,step,histories,configuration):
    validate_deployment(method,histories,configuration)
    actions=method['last_actions']
    if type(actions) is not dict or set(actions)!=(set(groups[0].parameter_names) if step else set()):raise ValueError('Complete policy action telemetry required')
    for info in actions.values():
        if type(info) is not dict or not {'alpha','h','tau','identity_action','original_W_commit','assistance_bypassed'}<=set(info):raise ValueError('Incomplete action telemetry')
        action=PolicyAction(info['alpha'],info['h'],info['tau'])
        if info['identity_action']!=action.identity or info['original_W_commit']!=action.identity or info['assistance_bypassed']!=(action.tau==1.):raise ValueError('Inconsistent policy action telemetry')
