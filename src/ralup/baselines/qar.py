"""Independent QAR Eq8 uniform fixture and named FP4 adjacent-gap adaptation.

Source arXiv:2609.21834v1, §4.2: feasibility masks magnitude BEFORE power
normalization (root clarification 2026-10-04). No author software was copied.
FP4 nonuniform gaps, original-W backbone/decay, rotation and logical caching
are project departures. Uniform-grid theory is not transferred to FP4.
"""
import hashlib,json,math,time
import torch
from torch import Tensor
from ..optim.base import MethodAdapter,SchedulePoint
from ..optim.adamw import AdamWUpdateRule
from ..quantization.context import prepare_context
from ..quantization.formats import _preserved_reconstruction,directional_neighbor
from ..quantization.rotation import to_storage,from_storage
from .trs import backbone_config,construct_backbone,BACKBONES

FIXTURE_CONFIG={'gamma':0.,'scale_recipe':'row_fixed_max','rotation':'identity','backward':'ste',
    'rng_stream':'quantizer','backbone':{'name':'adamw_scalar_fp32','config':{'betas':[.9,.99],'eps':1e-8}}}
ADAPTER_FIELDS=('optimizer_states','proposals','snapshot_schema','config','endpoint_states','observations','scale_initialization')
SOURCE_CLARIFICATION='2026-10-04: arXiv:2609.21834v1 Eq8 uses a=feasible*abs(u) BEFORE group power normalization; supersedes unmasked S24 condensation without rewriting its history'


def _scalar(value,name):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError(f'{name} must be finite nonnegative')


def _uniforms(uniforms,values):
    if uniforms.dtype!=torch.float32 or uniforms.shape!=values.shape or uniforms.device!=values.device or not bool(torch.isfinite(uniforms).all()) or not bool(((uniforms>=0)&(uniforms<1)).all()):
        raise ValueError('Explicit uniforms require matching finite FP32 values in [0,1)')


def _routing(signal,gaps,feasible,lr,gamma,group_axes):
    _scalar(lr,'lr');_scalar(gamma,'gamma')
    if signal.dtype!=torch.float32 or not signal.numel() or not bool(torch.isfinite(signal).all()):raise ValueError('Signal requires nonempty finite FP32')
    if gaps.dtype!=torch.float32 or gaps.shape!=signal.shape or gaps.device!=signal.device or not bool(torch.isfinite(gaps).all()):raise ValueError('Gaps require matching finite FP32')
    if feasible.dtype!=torch.bool or feasible.shape!=signal.shape or feasible.device!=signal.device:raise ValueError('Feasibility requires matching boolean tensor')
    if not bool((gaps[feasible]>0).all()):raise ValueError('Feasible directional gaps must be positive')
    if type(group_axes) is not tuple or not group_axes or any(type(a) is not int or not -signal.ndim<=a<signal.ndim for a in group_axes):raise ValueError('Group axes must be an explicit nonempty valid tuple')
    axes=tuple(a%signal.ndim for a in group_axes)
    if len(set(axes))!=len(axes):raise ValueError('Group axes must be unique')
    a=torch.where(feasible,signal.abs(),0.)
    if gamma==0:b=torch.ones_like(a)
    else:
        maximum=a.amax(dim=axes,keepdim=True)
        power=(a/torch.where(maximum>0,maximum,1.)).pow(gamma)
        mean=power.mean(dim=axes,keepdim=True)
        b=torch.where(mean>0,power/torch.where(mean>0,mean,1.),1.)
    active=(a>0)&feasible
    if lr==0:return torch.zeros_like(a),torch.zeros_like(feasible)
    # Log intensities avoid finite-input overflow and 0*inf. Invalid neighbors
    # have safe operands and are explicitly zero, including gamma=0 zeros.
    log_intensity=math.log(lr)+torch.where(active,a,1.).log()+torch.where(active&(b>0),b,1.).log()-torch.where(feasible,gaps,1.).log()
    active=active&(b>0)
    clipped=active&(log_intensity>0)
    p=torch.where(active,log_intensity.clamp(max=0).exp(),0.)
    return p,clipped


@torch.no_grad()
def routing_probabilities(signal:Tensor,gaps:Tensor,feasible:Tensor,lr:float,gamma:float,group_axes:tuple)->Tensor:
    """Source-masked group powers; actual directional gaps specify the grid."""
    return _routing(signal,gaps,feasible,lr,gamma,group_axes)[0]


@torch.no_grad()
def uniform_qar_step(codes,signal,*,spacing,minimum,maximum,lr,gamma,group_axes,uniforms):
    """Clean source fixture: fixed uniform integer endpoints, one-code moves.

    Persistent caller state is integer codes/shared spacing and original-W
    optimizer moments, never a latent residual. Explicit uniforms couple draws.
    """
    if codes.dtype!=torch.int64 or codes.shape!=signal.shape or codes.device!=signal.device:raise ValueError('Uniform codes require matching int64 tensor')
    if type(minimum) is not int or type(maximum) is not int or minimum>=maximum or not bool(((codes>=minimum)&(codes<=maximum)).all()):raise ValueError('Invalid uniform endpoint range')
    _scalar(spacing,'spacing')
    if spacing==0:raise ValueError('Uniform spacing must be positive')
    _uniforms(uniforms,signal)
    neighbor=codes-signal.sign().to(torch.int64)
    feasible=(signal!=0)&(neighbor>=minimum)&(neighbor<=maximum)
    p=routing_probabilities(signal,torch.full_like(signal,spacing),feasible,lr,gamma,group_axes)
    return torch.where(uniforms<p,neighbor,codes),p


@torch.no_grad()
def reconstruct(codes,scales):
    if codes.dtype!=torch.uint8 or not bool((codes<=14).all()):raise ValueError('Endpoint codes require valid FP4 uint8 ordinals')
    return _preserved_reconstruction(codes.to(torch.int64),scales.contiguous().view(torch.int32))


def validate_config(config,*,rounding=False):
    from ..config import canonical_bytes
    expected=set(FIXTURE_CONFIG)|({'mode'} if rounding else set())
    if type(config) is not dict or set(config)!=expected:raise ValueError('Routing/rounding requires complete explicit configuration')
    try:c=json.loads(canonical_bytes(config))
    except (TypeError,ValueError) as error:raise ValueError('Configuration requires finite strict canonical JSON') from error
    _scalar(c['gamma'],'gamma');construct_backbone(c['backbone'])
    if c['scale_recipe']!='row_fixed_max' or c['backward']!='ste' or c['rotation'] not in ('identity','full_hadamard') or c['rng_stream']!='quantizer':raise ValueError('Unsupported scale/group/context/RNG convention')
    if rounding and (c['mode'] not in ('shadow_forward','endpoint') or c['gamma']!=0):raise ValueError('Rounding requires an explicit state mode and unused gamma zero')
    return c


def _bytes(value):
    if isinstance(value,Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(_bytes(v) for v in value.values())
    return 0


def _digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _sync(device):
    if device.type=='mps':torch.mps.synchronize()


class _QuantizedAdapter(MethodAdapter):
    """Shared finite owner; H06 calls prepare_contexts inside its transaction."""
    rounding=False

    def __init__(self,config):
        self.config=validate_config(config,rounding=self.rounding)
        self.rule=construct_backbone(self.config['backbone'])

    @property
    def endpoint_only(self):return not self.rounding or self.config['mode']=='endpoint'

    def bind(self,parameters,groups):
        if tuple(g.name for g in groups) not in (('internal_projection',),('internal_projection','auxiliary_no_decay')) or groups[0].weight_decay is not None or (len(groups)>1 and groups[1].weight_decay!=0.):raise ValueError('Explicit projection/no-decay auxiliary groups required')
        MethodAdapter.__init__(self,parameters,groups,self.rule)
        self.endpoint_states={};self.observations=0;self.scale_initialization={}
        self.snapshot_schema['optimizer_states']=f'named.original_W.{self.rule.state_schema}+auxiliary.adamw.v1'
        self.snapshot_schema['adapter_mutable_state']='rounding.explicit_endpoint_or_shadow.v1' if self.rounding else 'qar.endpoint_codes_scales.v1'
        return self

    @property
    def metadata(self):
        value=super().metadata
        value['ordered_parameter_groups']=[{'name':g.name,'parameter_names':list(g.parameter_names),'weight_decay':g.weight_decay} for g in self.groups]
        value['update_rule_settings']=json.loads(json.dumps(value['update_rule_settings']))
        value['schedule_fields']=list(value['schedule_fields']);value['extra_passes']=[]
        value.update(routing_rounding_configuration=self.config,persistent_shadow=not self.endpoint_only,
            persistent_authority='codes+fixed_scales; W is reconstructed mirror' if self.endpoint_only else 'original_W FP32 shadow',
            moment_coordinates='original_W',supported_backbone_families=list(BACKBONES),unsupported_compositions=['SOAP','Madam','arbitrary unregistered rule'],
            compatibility_scope='current local composition only; future B0 selection remains unconstrained',source_clarification=SOURCE_CLARIFICATION,
            fp4_uniform_theory_transfer=False,primary_evaluation='deterministic nearest-even; read-only')
        return value

    def rule_for_parameter(self,name):
        if name in self.groups[0].parameter_names or type(self.rule) is AdamWUpdateRule:return self.rule
        return self.rule.auxiliary_rule

    def schedule_for_group(self,group,schedule):
        if group.name=='auxiliary_no_decay' and type(self.rule) is not AdamWUpdateRule:return SchedulePoint(self.rule.configuration['auxiliary_lr'],0.,schedule.progress)
        return super().schedule_for_group(group,schedule)

    def _rng(self,state):
        rng=state.rng_streams.get(self.config['rng_stream'])
        if type(rng) is not torch.Generator or rng.device.type!='cpu':raise ValueError('Requires independent named CPU quantizer RNG')
        return rng

    def _draw(self,state,w):
        # MPS generator support is not assumed; draws use this independently
        # named CPU owner and an explicit measured transfer to the operand device.
        return torch.rand(w.shape,generator=self._rng(state),dtype=torch.float32).to(w.device)

    def _work(self,state,**updates):
        if state._work is None:raise ValueError('Routing/rounding work requires active logical transaction')
        work=state._work.setdefault('routing_rounding_work',{'context_seconds':0.,'probability_seconds':0.,'probability_elements':0,'uniform_draw_elements':0,'uniform_transfer_bytes':0,'endpoint_initialization_elements':0,'persistent_endpoint_bytes':0,'persistent_shadow_bytes':0,'cached_context_bytes':0,'clipped_probability_count':0,'saturated_value_count':0})
        for name,value in updates.items():work[name]+=value

    @torch.no_grad()
    def prepare_contexts(self,state,step):
        self._rng(state)
        if state.model._recipe!={'name':self.config['scale_recipe'],'rotation':self.config['rotation'],'backward':self.config['backward']}:
            # Omitted identity/STE defaults are the same declared context.
            r=state.model._recipe or {}
            if (r.get('name'),r.get('rotation','identity'),r.get('backward','ste'))!=(self.config['scale_recipe'],self.config['rotation'],self.config['backward']):raise ValueError('Method/context configuration mismatch')
        if state.model._context_step==step and set(self.scale_initialization)==set(self.groups[0].parameter_names):return state.model._contexts
        device=next(iter(self.parameters.values())).device;_sync(device);start=time.perf_counter();contexts={}
        for name in self.groups[0].parameter_names:
            w=self.parameters[name]
            # A new arm initializes its frozen grid from the actual common
            # parent W, before conversion, rather than inheriting a stale cache.
            recipe={**state.model._recipe,**(state.model._runtime.get(name,{}) if name in self.scale_initialization else {})}
            if self.endpoint_only and name in self.endpoint_states:
                e=self.endpoint_states[name]
                if not torch.equal(w.detach(),from_storage(reconstruct(e['codes'],e['scales']),self.config['rotation'])):raise ValueError('Original W is not authoritative endpoint reconstruction')
                recipe.update(previous_scales=e['scales'],previous_zero_groups=e['scale_zero_groups'],codes=e['codes'])
            context=prepare_context(w,recipe,step,None)
            if name not in self.scale_initialization:
                self.scale_initialization[name]={'source':'pre_conversion_parent_storage_row_max','logical_step':step,'original_W_sha256':_digest(w),
                    'storage_U_sha256':_digest(context.storage_u),'scales_sha256':_digest(context.scales)}
            if self.endpoint_only and name not in self.endpoint_states:
                self.endpoint_states[name]={'codes':context.codes.clone(),'scales':context.scales.clone(),'scale_zero_groups':context.scale_zero_groups.clone()}
                w.copy_(from_storage(context.storage_values,self.config['rotation']))
                context=prepare_context(w,{**recipe,'previous_scales':context.scales,'previous_zero_groups':context.scale_zero_groups,'codes':context.codes},step,None)
                self._work(state,endpoint_initialization_elements=w.numel())
            elif not self.endpoint_only:
                from .rounding import stochastic_round
                uniforms=self._draw(state,w);values,codes=stochastic_round(context.storage_u,context.scales,uniforms)
                context=prepare_context(w,{**recipe,'codes':codes},step,None)
                self._work(state,probability_elements=w.numel(),uniform_draw_elements=w.numel(),uniform_transfer_bytes=w.numel()*4 if device.type!='cpu' else 0,
                    saturated_value_count=int((to_storage(w,self.config['rotation']).abs()>6*context.scales).sum()))
            contexts[name]=context
        state.model._contexts=contexts;state.model._context_step=step
        _sync(device)
        self._work(state,context_seconds=time.perf_counter()-start,persistent_endpoint_bytes=_bytes(self.endpoint_states),persistent_shadow_bytes=0 if self.endpoint_only else sum(self.parameters[n].numel()*4 for n in contexts),cached_context_bytes=sum(_bytes(dict(zip(range(len(c.numeric_tensors())),c.numeric_tensors()))) for c in contexts.values()))
        return contexts

    def prepare_step(self,state,batch):
        super().prepare_step(state,batch)
        if not state.model.training or set(state.model._contexts)!=set(self.groups[0].parameter_names):raise ValueError('Requires complete training logical contexts')

    @torch.no_grad()
    def finish_step(self,state,schedule):
        if not self.endpoint_only:
            result=super().finish_step(state,schedule);self.observations+=1
            return {**result,'routing_rounding':{'persistent_shadow':True,'clipped_probability_count':0,'method':'stochastic_shadow_forward'}}
        proposals=self.collect_proposals(schedule)
        state.collect_policy_features(proposals,schedule)
        candidates={};next_endpoints={};clipped=0;probabilities={}
        device=next(iter(self.parameters.values())).device;_sync(device);start=time.perf_counter()
        for name,p in proposals.items():
            if name not in self.endpoint_states:
                candidate=self.parameters[name].detach().clone();self.rule_for_parameter(name).commit(candidate,p);candidates[name]=candidate;continue
            endpoint=self.endpoint_states[name];context=state.model._contexts[name];uniforms=self._draw(state,self.parameters[name])
            if self.rounding:
                from .rounding import stochastic_round,upper_probability
                candidate=self.parameters[name].detach().clone();self.rule_for_parameter(name).commit(candidate,p)
                storage=to_storage(candidate,self.config['rotation']);values,codes=stochastic_round(storage,endpoint['scales'],uniforms)
                probability=upper_probability(storage,endpoint['scales'])
                self._work(state,saturated_value_count=int((storage.abs()>6*endpoint['scales']).sum()))
            else:
                # Full original-W proposal includes ordinary-LR decoupled decay;
                # normalize common LR out of group powers by using displacement
                # as signal and eta=1. This equals u=-delta/lr for lr>0.
                displacement=(p.reference_after_decay-self.parameters[name].detach())+p.gradient_displacement
                signal=-to_storage(displacement,self.config['rotation'])
                neighbor,feasible=directional_neighbor(endpoint['codes'],-signal)
                current=reconstruct(endpoint['codes'],endpoint['scales']);target=reconstruct(neighbor,endpoint['scales'])
                gaps=(target-current).abs();probability,cap=_routing(signal,gaps,feasible,1.,self.config['gamma'],(-1,))
                clipped+=int(cap.sum());codes=torch.where(uniforms<probability,neighbor,endpoint['codes'])
                values=reconstruct(codes,endpoint['scales'])
            candidates[name]=from_storage(values,self.config['rotation'])
            next_endpoints[name]={**endpoint,'codes':codes.clone()};probabilities[name]=probability.detach().clone()
            self._work(state,probability_elements=probability.numel(),uniform_draw_elements=uniforms.numel(),uniform_transfer_bytes=uniforms.numel()*4 if device.type!='cpu' else 0)
        if set(next_endpoints)!=set(self.endpoint_states):raise ValueError('Missing endpoint projection gradient')
        if any(not bool(torch.isfinite(v).all()) for v in candidates.values()):raise ValueError('Nonfinite endpoint candidate')
        for name,candidate in candidates.items():self.parameters[name].copy_(candidate)
        for name,p in proposals.items():self.optimizer_states[name]=p.new_optimizer_state
        self.endpoint_states=next_endpoints;self.proposals={};self.observations+=1;_sync(device)
        self._work(state,probability_seconds=time.perf_counter()-start,clipped_probability_count=clipped)
        return {'optimizer_updates':len(candidates),'candidate_validation_updates':len(candidates),'extra_forward_passes':0,'extra_backward_passes':0,
            'routing_rounding':{'persistent_shadow':False,'clipped_probability_count':clipped,'method':'stochastic_endpoint' if self.rounding else 'qar_fp4_adjacent',
                'probability_ranges':{n:[float(p.min()),float(p.max())] for n,p in probabilities.items()}}}


class QarAdjacentAdapter(_QuantizedAdapter):
    """Endpoint-only FP4 routing; no persistent latent/shadow residual."""


def trusted_constructor(parameters,groups,rule):
    return QarAdjacentAdapter({**FIXTURE_CONFIG,'backbone':backbone_config(rule)}).bind(parameters,groups)


def validate_owned_state(method,groups,step,contexts,rule,weights,rng_streams,recipe,*,rounding=False):
    c=validate_config(method['config'],rounding=rounding)
    if c['backbone']!=backbone_config(rule) or type(method['observations']) is not int or method['observations']!=step:raise ValueError('Endpoint rule/observation mismatch')
    if recipe is None or (recipe.get('name'),recipe.get('rotation','identity'),recipe.get('backward','ste'))!=(c['scale_recipe'],c['rotation'],c['backward']):raise ValueError('Endpoint recipe/config mismatch')
    stream=rng_streams.get(c['rng_stream'])
    if type(stream) is not dict or stream.get('__rng_kind__')!='torch_rng' or stream.get('device')!='cpu':raise ValueError('Endpoint requires owned named CPU quantizer RNG')
    shadow=rounding and c['mode']=='shadow_forward';expected=set(groups[0].parameter_names)
    if type(method['endpoint_states']) is not dict or (shadow and method['endpoint_states']) or (not shadow and set(method['endpoint_states'])!=(expected if step or contexts else set())):raise ValueError('Incomplete endpoint state inventory')
    initialization=method['scale_initialization']
    if type(initialization) is not dict or set(initialization)!=(expected if step or contexts else set()):raise ValueError('Incomplete pre-conversion scale initialization inventory')
    for name,entry in initialization.items():
        if type(entry) is not dict or set(entry)!={'source','logical_step','original_W_sha256','storage_U_sha256','scales_sha256'} or entry['source']!='pre_conversion_parent_storage_row_max' or type(entry['logical_step']) is not int or entry['logical_step']!=0:raise ValueError('Invalid pre-conversion parent scale metadata')
        for key in ('original_W_sha256','storage_U_sha256','scales_sha256'):
            if type(entry[key]) is not str or len(entry[key])!=64 or any(ch not in '0123456789abcdef' for ch in entry[key]):raise ValueError('Invalid parent/scale digest')
        if entry['scales_sha256']!=_digest(contexts[name].scales):raise ValueError('Endpoint grid differs from recorded pre-conversion fixed scales')
    for name,e in method['endpoint_states'].items():
        context=contexts[name]
        if type(e) is not dict or set(e)!={'codes','scales','scale_zero_groups'}:raise ValueError('Unknown endpoint state fields')
        codes,scales,zero=e['codes'],e['scales'],e['scale_zero_groups']
        if codes.dtype!=torch.uint8 or codes.shape!=context.codes.shape or not bool((codes<=14).all()) or scales.dtype!=torch.float32 or scales.shape!=context.scales.shape or not bool((scales>0).all()) or zero.dtype!=torch.bool or zero.shape!=scales.shape:raise ValueError('Invalid endpoint state geometry')
        if not torch.equal(scales,context.scales) or not torch.equal(zero,context.scale_zero_groups):raise ValueError('Endpoint fixed scale ownership mismatch')
        if not torch.equal(weights[name],from_storage(reconstruct(codes,scales),c['rotation'])):raise ValueError('W must equal authoritative endpoint reconstruction; no shadow residual')
