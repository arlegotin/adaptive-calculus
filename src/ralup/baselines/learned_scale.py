"""Independent LSQ uniform surrogate and named FP4/log-scale adaptation.

Physical convention y=s*C(w/s); Cmax normalization applies once to dL/ds,
then dL/dlog(s)=s*dL/ds. The target is the declared analytic surrogate,
never a finite difference of the discontinuous nearest-level forward.
"""
from dataclasses import replace
import json
import math
import torch
from torch import Tensor,nn
from ..optim.base import MethodAdapter,SchedulePoint
from ..optim.adamw import AdamWUpdateRule
from ..quantization.formats import quantize_e2m1,E2M1_VALUES
from ..quantization.context import prepare_context
from .noise import NoiseAdapter
from .trs import backbone_config,construct_backbone

VIEW_CONTRACT={'format':'fp4','minimum':-6,'maximum':6,'grouping':'row','weight_gradient':'clipped_ste'}
FIXTURE_CONFIG={'view':VIEW_CONTRACT,'scale_lr':.001,'scale_betas':[.9,.99],'scale_eps':1e-8,
                'scale_clip_norm':1.,'log_scale_range':[-20.,10.],
                'initialization':'max_div_cmax_zero_one','export_precision':'float32',
                'information':'training_targets_only',
                'backbone':{'name':'adamw_scalar_fp32','config':{'betas':[.9,.99],'eps':1e-8}}}
ADAPTER_FIELDS=('optimizer_states','proposals','snapshot_schema','config','log_scales','scale_gradients',
                'scale_optimizer_states','normalizer','logical_target_count','physical_scales','scale_backend','scale_validation_telemetry')


def validate_view(contract):
    from ..config import canonical_bytes
    if type(contract) is not dict or set(contract)!=set(VIEW_CONTRACT):raise ValueError('Complete scale view contract required')
    c=json.loads(canonical_bytes(contract))
    if c['format'] not in ('uniform','fp4') or c['grouping'] not in ('row','tensor') or c['weight_gradient'] not in ('clipped_ste','ste'):raise ValueError('Unknown scale surrogate/grouping')
    low,high=c['minimum'],c['maximum']
    if type(low) is not int or type(high) is not int or not low<0<high:raise ValueError('Explicit negative/positive integer range required')
    if c['format']=='fp4' and (low not in E2M1_VALUES or high not in E2M1_VALUES):raise ValueError('FP4 range endpoints must be E2M1 levels')
    if c['format']=='uniform' and max(abs(low),high)>32767:raise ValueError('Uniform fixture range too large')
    return c


class _LearnedScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx,w,log_scale,contract):
        scale=log_scale.exp();x=w/scale
        if contract['format']=='uniform':code=x.clamp(contract['minimum'],contract['maximum']).round()
        else:
            codes=quantize_e2m1(w.detach(),scale.detach())[1]
            code=torch.tensor(E2M1_VALUES,dtype=w.dtype,device=w.device)[codes.long()]
        code=code.clamp(contract['minimum'],contract['maximum'])
        inside=(x>contract['minimum'])&(x<contract['maximum'])
        local=torch.where(x<=contract['minimum'],float(contract['minimum']),torch.where(x>=contract['maximum'],float(contract['maximum']),code-x))
        n=w.shape[-1] if contract['grouping']=='row' else w.numel()
        factor=1/math.sqrt(n*contract['maximum'])
        ctx.save_for_backward(local,scale,inside);ctx.factor=factor;ctx.weight_gradient=contract['weight_gradient']
        return scale*code

    @staticmethod
    def backward(ctx,gradient):
        local,scale,inside=ctx.saved_tensors
        dw=gradient*inside if ctx.weight_gradient=='clipped_ste' else gradient
        dlog=(gradient*local).sum_to_size(scale.shape)*ctx.factor*scale
        return dw,dlog,None


def learned_scale_view(w:Tensor,log_scale:Tensor,contract:dict)->Tensor:
    c=validate_view(contract)
    shape=(w.shape[0],1) if w.ndim==2 and c['grouping']=='row' else (1,1)
    if w.ndim!=2 or not w.numel() or log_scale.shape!=shape or w.dtype!=torch.float32 or log_scale.dtype!=w.dtype or w.device!=log_scale.device or not bool(torch.isfinite(w).all()&torch.isfinite(log_scale).all()):raise ValueError('Learned scale requires matching finite FP32 matrix/group log scales')
    scales=log_scale.detach().exp()
    if not bool(torch.isfinite(scales).all()&(scales>0).all()):raise ValueError('Scale exponent must be positive finite')
    return _LearnedScale.apply(w,log_scale,c)


def validate_config(config):
    from ..config import canonical_bytes
    if type(config) is not dict or set(config)!=set(FIXTURE_CONFIG):raise ValueError('Complete learned-scale configuration required')
    c=json.loads(canonical_bytes(config));c['view']=validate_view(c['view'])
    if c['view']['format']!='fp4':raise ValueError('Training adapter is the named FP4 adaptation; uniform is an equation fixture')
    if (c['view']['minimum'],c['view']['maximum'])!=(-6,6):raise ValueError('Training/evaluation/export adapter supports the full FP4 range [-6,6] only')
    construct_backbone(c['backbone'])
    AdamWUpdateRule(betas=tuple(c['scale_betas']),eps=c['scale_eps'])
    if type(c['scale_lr']) not in (int,float) or c['scale_lr']<0:raise ValueError('Scale LR must be nonnegative')
    if c['scale_clip_norm'] is not None and (type(c['scale_clip_norm']) not in (int,float) or c['scale_clip_norm']<=0):raise ValueError('Scale clipping must be null or positive')
    if type(c['log_scale_range']) is not list or len(c['log_scale_range'])!=2 or any(type(v) not in (int,float) for v in c['log_scale_range']) or not -80<=c['log_scale_range'][0]<c['log_scale_range'][1]<=80:raise ValueError('Finite ordered log scale range required')
    for k in ('initialization','export_precision','information'):
        if c[k]!=FIXTURE_CONFIG[k]:raise ValueError(f'Unknown scale information/precision recipe: {k}')
    return c


class LearnedScaleAdapter(NoiseAdapter):
    """Adapter-owned actual Parameters; original-W backbone moments stay dense."""
    def __init__(self,config):
        self.config=validate_config(config);self.rule=construct_backbone(self.config['backbone'])

    def bind(self,parameters,groups):
        # Reuse only the explicit original-W group/auxiliary dispatch contract.
        NoiseAdapter.bind(self,parameters,groups)
        del self.mask_states,self.mask_rng_state
        self.log_scales={};self.scale_gradients={};self.scale_optimizer_states={};self.normalizer={};self.logical_target_count=0
        self.physical_scales={};self.scale_backend=None
        self.scale_validation_telemetry={'checks':0,'bytes_checked':0,'seconds':0.}
        self.snapshot_schema['adapter_mutable_state']='learned_scale.parameters_gradients_adam_normalizer.v1'
        return self

    @property
    def recipe(self):return ('row' if self.config['view']['grouping']=='row' else 'tensor')+'_fixed_max'

    @property
    def metadata(self):
        return {'ordered_parameter_groups':[{'name':g.name,'parameter_names':list(g.parameter_names),'weight_decay':g.weight_decay} for g in self.groups],
                'update_rule_settings':self.config,'schedule_fields':['lr','weight_decay','progress','scale_lr'],
                'extra_passes':[],'state_schema':self.snapshot_schema,'scale_recipe':self.scale_contract,
                'rotation':'identity','supported_backbones':['AdamW','Muon','Muon+'],'unsupported_backbones':['SOAP','Madam']}

    @property
    def scale_contract(self):
        return {'physical':'y=s*C(w/s)','grouping':self.config['view']['grouping'],'range':[self.config['view']['minimum'],self.config['view']['maximum']],
                'quantizer_recipe':self.recipe,'codebook':list(E2M1_VALUES),
                'initialization':self.config['initialization'],'information':self.config['information'],'export_precision':self.config['export_precision'],
                'normalizer':'1/sqrt(N*Cmax) once before log chain','scale_lr':self.config['scale_lr'],'scale_clip_norm':self.config['scale_clip_norm'],
                'log_scale_range':list(self.config['log_scale_range'])}

    def export_scales(self):
        self.validate_live_scales()
        return {'contract':self.scale_contract,'scales':{n:s.clone() for n,s in self.physical_scales.items()},'backend':self.scale_backend}

    @torch.no_grad()
    def validate_live_scales(self):
        """Exact on declared producer backend; CPU artifact checks never exp.

        Observed check/copy costs are safe audit telemetry, excluded from the
        scientific hash alongside existing wall/process costs.
        """
        import time
        start=time.perf_counter()
        if not self.log_scales:
            if self.physical_scales or self.scale_backend is not None:raise ValueError('Empty learned owner has a scale/backend cache')
            return
        if self.scale_backend not in ('cpu','mps') or set(self.physical_scales)!=set(self.log_scales):raise ValueError('Invalid complete physical scale/backend owner')
        checked=0
        for n,p in self.log_scales.items():
            cache=self.physical_scales[n]
            if type(cache) is not Tensor or cache.requires_grad or cache.grad_fn is not None or p.device.type!=self.scale_backend or cache.device!=p.device or not torch.equal(p.detach().exp(),cache):raise ValueError('Exact declared-backend detached log/physical scale mismatch')
            checked+=cache.numel()*cache.element_size()
        if self.scale_backend=='mps':torch.mps.synchronize()
        t=self.scale_validation_telemetry;t['checks']+=1;t['bytes_checked']+=checked;t['seconds']+=time.perf_counter()-start

    def transaction_parameters(self):return self.log_scales

    def restore_owned_parameters(self,originals=None):
        values=self.log_scales
        if originals is None:originals={n:nn.Parameter(v.detach().clone()) for n,v in values.items()}
        if set(originals)!=set(values):raise ValueError('Learned scale rollback inventory mismatch')
        with torch.no_grad():
            for n,p in originals.items():p.copy_(values[n]);p.grad=None if self.scale_gradients[n] is None else self.scale_gradients[n].clone()
        self.log_scales=originals
        self.scale_gradients={n:p.grad for n,p in originals.items()}
        for s in self.scale_optimizer_states.values():
            if s:s['step']=s['step'].cpu()

    def prepare_contexts(self,state,step):
        if state.model._recipe!={'name':self.recipe,'rotation':'identity','backward':'ste'}:raise ValueError('Learned scale requires explicit identity/STE/grouped fixed recipe')
        if not self.log_scales:
            for n in self.groups[0].parameter_names:
                w=self.parameters[n].detach()
                maximum=w.abs().amax(dim=-1,keepdim=True) if self.config['view']['grouping']=='row' else w.abs().amax().reshape(1,1)
                s=torch.where(maximum>0,maximum/self.config['view']['maximum'],1.)
                self.log_scales[n]=nn.Parameter(s.log().clamp(*self.config['log_scale_range']))
                self.scale_gradients[n]=None;self.scale_optimizer_states[n]={}
                N=w.shape[-1] if self.config['view']['grouping']=='row' else w.numel()
                self.normalizer[n]=1/math.sqrt(N*self.config['view']['maximum'])
                self.physical_scales[n]=self.log_scales[n].detach().exp().clone()
            self.scale_backend=next(iter(self.log_scales.values())).device.type
        self.validate_live_scales()
        contexts={}
        for n,p in self.log_scales.items():
            w=self.parameters[n];scales=self.physical_scales[n]
            initial_zero=state.model._runtime.get(n,{}).get('previous_zero_groups')
            recipe={'name':self.recipe,'rotation':'identity','backward':'ste','previous_scales':scales}
            if initial_zero is not None:recipe['previous_zero_groups']=initial_zero
            c=prepare_context(w,recipe,step,None)
            if c.scale_zero_groups is None:c=replace(c,scale_zero_groups=c.zero_groups.clone())
            values=learned_scale_view(w,p,self.config['view']).detach()
            codes=quantize_e2m1(values,scales)[1]
            contexts[n]=replace(c,storage_values=values,codes=codes)
        state.model._contexts=contexts;state.model._context_step=step
        return contexts

    def prepare_step(self,state,batch):
        MethodAdapter.prepare_step(self,state,batch);self.logical_target_count=batch.target_count
        for n,p in self.log_scales.items():p.grad=None;self.scale_gradients[n]=None

    def backward_microbatch(self,state,batch):
        replacements={n:learned_scale_view(self.parameters[n],p,self.config['view']) for n,p in self.log_scales.items()}
        loss=state.summed_nll(batch,parameter_overrides=replacements);loss.backward()
        self.scale_gradients={n:p.grad for n,p in self.log_scales.items()}
        return {'summed_nll':float(loss.detach()),'forward_passes':1,'backward_passes':1,'extra_forward_passes':0,'extra_backward_passes':0}

    def finish_step(self,state,schedule):
        import time
        start=time.perf_counter()
        for p in self.log_scales.values():
            if p.grad is None:raise ValueError('Missing learned scale gradient')
            p.grad.div_(self.logical_target_count)
            if not bool(torch.isfinite(p.grad).all()):raise ValueError('Nonfinite normalized scale gradient')
        norm=None
        if self.config['scale_clip_norm'] is not None:norm=float(torch.nn.utils.clip_grad_norm_(list(self.log_scales.values()),self.config['scale_clip_norm'],error_if_nonfinite=True))
        scale_rule=AdamWUpdateRule(betas=tuple(self.config['scale_betas']),eps=self.config['scale_eps'])
        scale_proposals={n:scale_rule.propose(p,p.grad,self.scale_optimizer_states[n],SchedulePoint(self.config['scale_lr'],0.,schedule.progress)) for n,p in self.log_scales.items()}
        for n,p in scale_proposals.items():
            candidate=self.log_scales[n].detach().clone();scale_rule.commit(candidate,p)
            if not bool(torch.isfinite(candidate).all()):raise ValueError('Nonfinite scale candidate')
        result=self.commit_proposals(self.collect_proposals(schedule))
        with torch.no_grad():
            for n,p in scale_proposals.items():
                scale_rule.commit(self.log_scales[n],p);self.log_scales[n].clamp_(*self.config['log_scale_range'])
                self.scale_optimizer_states[n]=p.new_optimizer_state
                self.physical_scales[n]=self.log_scales[n].detach().exp().clone()
                # Runtime is an observational export/evaluation handoff. The
                # trusted learned owner is authoritative, never a reset cache.
                handoff={'previous_scales':self.physical_scales[n].clone(),
                         'previous_zero_groups':state.model._contexts[n].scale_zero_groups.clone()}
                state.model._runtime[n]=handoff;state.quantizer_state[n]={k:v.clone() for k,v in handoff.items()}
        self.scale_gradients={n:p.grad for n,p in self.log_scales.items()};self.logical_target_count=0
        state._work['learned_scale_work']={'scale_parameters':sum(p.numel() for p in self.log_scales.values()),'scale_optimizer_updates':len(scale_proposals),
            'scale_candidate_validation_updates':len(scale_proposals),'scale_seconds':time.perf_counter()-start,'extra_loss_passes':0,'extra_backward_passes':0}
        return {**result,'scale_gradient_norm_before_clip':norm,'learned_scale_contract':self.scale_contract}


def trusted_constructor(parameters,groups,rule):
    return LearnedScaleAdapter({**FIXTURE_CONFIG,'backbone':backbone_config(rule)}).bind(parameters,groups)


def validate_owned_state(method,groups,step,rule,recipe,weights,runtime,quantizer):
    c=validate_config(method['config']);expected=set(groups[0].parameter_names) if step else set(method['log_scales'])
    required_recipe=('row' if c['view']['grouping']=='row' else 'tensor')+'_fixed_max'
    if c['backbone']!=backbone_config(rule) or recipe!={'name':required_recipe,'rotation':'identity','backward':'ste'} or method['logical_target_count']!=0:raise ValueError('Learned-scale rule/recipe/pending target mismatch')
    if expected and expected!=set(groups[0].parameter_names):raise ValueError('Partial learned scale inventory')
    if (not expected and method['scale_backend'] is not None) or (expected and method['scale_backend'] not in ('cpu','mps')):raise ValueError('Unknown finite scale backend')
    telemetry=method['scale_validation_telemetry']
    if type(telemetry) is not dict or set(telemetry)!={'checks','bytes_checked','seconds'} or any(type(telemetry[k]) is not int or telemetry[k]<0 for k in ('checks','bytes_checked')) or type(telemetry['seconds']) not in (int,float) or not math.isfinite(telemetry['seconds']) or telemetry['seconds']<0:raise ValueError('Invalid scale validation cost telemetry')
    for key in ('log_scales','scale_gradients','scale_optimizer_states','normalizer','physical_scales'):
        if set(method[key])!=expected:raise ValueError('Incomplete learned scale owner')
    for n in expected:
        w=weights[n];p=method['log_scales'][n];g=method['scale_gradients'][n]
        shape=(w.shape[0],1) if c['view']['grouping']=='row' else (1,1)
        if p.shape!=shape or p.dtype!=torch.float32 or not bool(((p>=c['log_scale_range'][0])&(p<=c['log_scale_range'][1])).all()):raise ValueError('Invalid scale Parameter range/shape/dtype')
        physical=method['physical_scales'][n]
        if physical.shape!=shape or physical.dtype!=torch.float32 or not bool((physical>0).all()):raise ValueError('Invalid physical scale cache')
        if g is not None and (g.shape!=shape or g.dtype!=torch.float32):raise ValueError('Invalid scale gradient')
        if step and g is None:raise ValueError('Missing completed scale gradient')
        N=w.shape[-1] if c['view']['grouping']=='row' else w.numel()
        if method['normalizer'][n]!=1/math.sqrt(N*c['view']['maximum']):raise ValueError('Scale normalizer mismatch')
        s=method['scale_optimizer_states'][n]
        if step and not s:raise ValueError('Missing scale Adam state')
        if s:
            if set(s)!=set(AdamWUpdateRule.optimizer_state_fields) or s['step'].shape!=torch.Size([]) or s['step'].dtype!=torch.float32 or float(s['step'])!=step:raise ValueError('Scale optimizer inventory/counter mismatch')
            if any(s[k].shape!=shape or s[k].dtype!=torch.float32 for k in ('exp_avg','exp_avg_sq')) or bool((s['exp_avg_sq']<0).any()):raise ValueError('Scale optimizer shape/dtype/variance mismatch')
        if step:
            for owner in (runtime,quantizer):
                if not torch.equal(owner[n]['previous_scales'],physical):raise ValueError('Learned runtime scale differs from authoritative physical cache')
