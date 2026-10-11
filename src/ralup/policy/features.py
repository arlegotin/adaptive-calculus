"""S19's fourteen causal scalars; coverage is telemetry, never a policy input."""
from dataclasses import dataclass, field
import math
import time
import torch
from torch import Tensor
from .actions import _rms, _validate
from .history import MatrixHistory, LossObservation
from ..quantization.formats import quantize_e2m1, directional_neighbor, E2M1_THRESHOLDS, _preserved_reconstruction
from ..quantization.fp32 import bits, add_rescaled, multiply_rescaled
from ..quantization.rotation import to_storage


@dataclass(frozen=True)
class FeatureVector:
    values: Tensor
    coverage: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.values.shape!=(14,) or not self.values.is_floating_point() or not bool(torch.isfinite(self.values).all()):
            raise ValueError('FeatureVector requires fourteen finite scalars')
        object.__setattr__(self,'values',self.values.detach().clone())


@torch.no_grad()
def collect_features(u0, da, gradient, first_moment_w, history, context, loss_ema, progress) -> FeatureVector:
    _validate(u0,da)
    if u0.ndim!=2 or u0.shape!=context.storage_u.shape:raise ValueError('Features require the context [out,in] matrix')
    for tensor in (gradient,first_moment_w):
        _validate(u0,tensor)
    if not isinstance(loss_ema,LossObservation):raise ValueError('Explicit prospective training LossObservation required')
    if not math.isfinite(progress):raise ValueError('Full continuation progress must be finite')
    r=_rms(u0);rg=_rms(gradient);rd=_rms(da)
    # Unit-vector dot avoids underflow/overflow in the cosine denominator.
    rm=_rms(first_moment_w)
    cosine=((gradient/rg)*(first_moment_w/rm)).mean() if float(rg)>0 and float(rm)>0 else u0.new_zeros(())
    scales=torch.broadcast_to(context.scales,u0.shape).contiguous()
    tiny=bool(((bits(u0)&0x7fffffff)<0x800000).any()) or bool(((bits(da)&0x7fffffff)<0x800000).any()) or bool((bits(scales)<0x1000000).any())
    add=add_rescaled if tiny else lambda a,b,subtract=False:a-b if subtract else a+b
    mul=multiply_rescaled if tiny else lambda a,b:a*b
    q0,c0=quantize_e2m1(u0,scales);q1,c1=quantize_e2m1(add(u0,da),scales)
    neighbor,exists=directional_neighbor(c0,da)
    qn=_preserved_reconstruction(neighbor.long(),bits(scales))
    thresholds=torch.tensor(E2M1_THRESHOLDS,device=u0.device)
    midpoint=mul(scales,thresholds[torch.minimum(c0,neighbor).long().clamp_max(13)])
    rho=add(midpoint,u0,subtract=True).abs();gap=add(qn,q0,subtract=True).abs()
    upper=_preserved_reconstruction(torch.full_like(c0,14).long(),bits(scales))
    outside=bits(u0.abs())>bits(upper)
    nonzero=(bits(da)&0x7fffffff)>0
    positive=exists&~outside&(bits(rho)>0)
    nz=int(nonzero.sum());eligible=int(positive.sum())
    retention=(c0[nonzero]==c1[nonzero]).float().mean() if nz else u0.new_zeros(())
    reach=(da[positive].abs()/rho[positive].clamp_min(1e-30)).clamp_max(2).mean()/2 if eligible else u0.new_zeros(())
    near=(rho[positive]/gap[positive]<=.1).float().mean() if eligible else u0.new_zeros(())
    values=torch.stack((r.clamp_min(1e-30).log10(),rg.clamp_min(1e-30).log10(),
                        (rd/r.clamp_min(1e-30)).clamp_min(1e-30).log10(),cosine,retention,reach,near,
                        outside.float().mean(),u0.new_tensor(history.transition_fraction if history.updates else 0.),
                        u0.new_tensor(history.reversal_fraction if history.reversal_valid else 0.),
                        u0.new_tensor(history.disagreement_fraction if history.realized_valid else 0.),
                        u0.new_tensor(loss_ema.relative_change),u0.new_tensor(max(0.,min(1.,progress))),
                        u0.new_tensor(math.log2(u0.shape[0]/u0.shape[1]))))
    return FeatureVector(values,{'coordinates':u0.numel(),'nonzero_moves':nz,'positive_distance':eligible,
                                'history_valid':history.updates>0,'reversal_valid':history.reversal_valid,
                                'reversal_compatible_coverage':history.reversal_coverage,'realized_valid':history.realized_valid,
                                'decay_reference_transitions':int((context.codes!=c0).sum()),
                                'base_additive_transitions':int((c0!=c1).sum()),
                                'decay_endpoint_displacement':float((q0-context.storage_values).abs().mean()),
                                'base_additive_endpoint_displacement':float((q1-q0).abs().mean())})


def proposal_first_moment(rule, proposal, gradient, history):
    """Only trusted optimizer-owned conversion supplies the NEW W moment."""
    state=proposal.new_optimizer_state
    if hasattr(rule,'first_moment_in_original_coordinates'):
        return rule.first_moment_in_original_coordinates(state),None,'adapter_inverse_basis_original_W'
    if 'exp_avg' in state:return state['exp_avg'],None,'new_original_W_exp_avg'
    if 'momentum' in state:return state['momentum'],None,'new_original_W_momentum'
    old=history.common_gradient_ema
    moment=gradient.detach().clone() if old is None else .95*old+.05*gradient
    return moment,moment,'common_gradient_ema_095_charged'


def collect_step_features(state, proposals, schedule):
    """Finite explicit common instrumentation shared by policy and all controls.

    Raw baseline deployments leave this off; matched policy/LR/static/ablations
    use it identically. No arbitrary callbacks and no second optimizer proposal.
    Candidates/EMA publish only in train's single history commit.
    """
    cfg=state.configuration.get('policy_features')
    if cfg is None:return {}
    if type(cfg) is not dict or set(cfg)-{'enabled','full_horizon','trace_cadence'} or cfg.get('enabled') is not True:
        raise ValueError('Policy feature mode must be explicit and enabled')
    if any(type(cfg.get(k)) is not int or cfg[k]<=0 for k in ('full_horizon','trace_cadence')):
        raise ValueError('Finite full continuation horizon and trace cadence required')
    if state._work is None:raise ValueError('Feature collection requires an active logical transaction')
    if 'features' in state._work:raise ValueError('Repeated feature scan in one logical step')
    device=next(state.model.parameters()).device
    if device.type=='mps':torch.mps.synchronize()
    start=time.perf_counter();features={};common={};raw={};clipped=total=0
    for name,context in state.model._contexts.items():
        p=proposals[name];history=state.histories.get(name,MatrixHistory());gradient=state.method.parameters[name].grad
        moment,ema,source=proposal_first_moment(state.method.rule_for_parameter(name),p,gradient,history)
        f=collect_features(to_storage(p.reference_after_decay,context.rotation),to_storage(p.gradient_displacement,context.rotation),
                           gradient,moment,history,context,LossObservation(state.histories.get('loss_ema'),state._work['training_loss']),
                           state.schedule['step']/cfg['full_horizon'])
        if ema is not None:common[name]=ema
        features[name]=f
        normalized=f.values;clip={'clipped':0,'total':14}
        if state.histories.get('feature_normalizer') is not None:normalized,clip=state.histories['feature_normalizer'].normalize(f.values)
        clipped+=clip['clipped'];total+=clip['total']
        raw[name]={'values':f.values.cpu().tolist(),'normalized':normalized.cpu().tolist(),'coverage':f.coverage,'moment_source':source}
    if device.type=='mps':torch.mps.synchronize()
    state._work['features']=raw;state._work['feature_tensors']={name:f.values for name,f in features.items()}
    state._work['common_gradient_ema']=common
    state._work['feature_work']={'scan_seconds':time.perf_counter()-start,'matrix_scans':len(features),'clipped':clipped,'scalars':total,
                                 'trace_cadence':cfg['trace_cadence'],'detailed_trace':state.schedule['step']%cfg['trace_cadence']==0}
    return features
