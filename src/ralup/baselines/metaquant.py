"""Independent MetaQuant mathematics (Chen, Wang & Pan, NeurIPS 2019).

No author software copied: csyhhu/MetaQuant@50694404 has no reuse grant.
The DoReFa equation fixture and physical-unit FP4 LM adaptation are distinct.
No release, performance, legal or universal MPS qualification is implied.
"""
from dataclasses import replace
import json
import math
import torch
from torch import Tensor, nn
from ..optim.base import BaseUpdateRule,MethodAdapter,SchedulePoint
from ..optim.adamw import AdamWUpdateRule
from ..quantization.context import prepare_context
from ..quantization.formats import quantize_e2m1
from ..quantization.rotation import to_storage,from_storage
from ..quantization.scales import group_zero_flags
from .noise import NoiseAdapter
from .trs import backbone_config,construct_backbone

FIXTURE_CONFIG={'variant':'online_target_training','rotation':'identity','scale_recipe':'row_fixed_max',
    'widths':[1,100,100,1],'meta_seed':1010,'meta_lr':.001,'meta_betas':[.9,.999],'meta_eps':1e-8,
    'information':'training_targets_only','normalization':'U_over_cached_physical_row_range',
    'calibration':'physical_identity_ste','decay':'committed_original_W_only',
    'auxiliary':'ordinary_current_weights_no_fast_displacement_no_decay',
    'backbone':{'name':'adamw_scalar_fp32','config':{'betas':[.9,.99],'eps':1e-8}}}
ADAPTER_FIELDS=('optimizer_states','proposals','snapshot_schema','config','theta','theta_gradients','outer_states',
    'gprev','rprev','cached_ranges','fast_contexts','fast_directions','fast_lr','committed_signal','pending_gradients','logical_target_count','observations','theta_updates','reset_count')
FACTOR_SHAPES={'factor0':(100,1),'factor1':(100,100),'factor2':(1,100)}


class FactorizedMetaNetwork(nn.Module):
    """Scalar map with all factors retained; no biases or activations."""
    def __init__(self,widths:tuple=(1,100,100,1)):
        super().__init__()
        if len(widths)!=4 or widths[0]!=1 or widths[-1]!=1 or any(type(w) is not int or w<1 for w in widths):raise ValueError('Three bias-free scalar-input/output factors required')
        self.layers=nn.ModuleList(nn.Linear(a,b,bias=False,dtype=torch.float32) for a,b in zip(widths,widths[1:]))

    def forward(self,x:Tensor)->Tensor:
        product=self.layers[0].weight
        for layer in self.layers[1:]:product=layer.weight@product
        return torch.nn.functional.linear(x,product)


def _matching(u,*values):
    if not u.numel() or not u.is_floating_point() or not bool(torch.isfinite(u).all()) or any(v.shape!=u.shape or v.dtype!=u.dtype or v.device!=u.device or not bool(torch.isfinite(v).all()) for v in values):raise ValueError('Matching finite floating operands required')


def make_fast_weights(u:Tensor,correction:Tensor,previous_direction:Tensor,lr:float)->Tensor:
    _matching(u,correction,previous_direction)
    if type(lr) not in (int,float) or not math.isfinite(lr) or lr<0:raise ValueError('Finite nonnegative fast LR required')
    return u-lr*(correction+(previous_direction.detach()-correction).detach())


def _dorefa_preprocess(w):
    if not w.numel() or not bool(torch.isfinite(w).all()):raise ValueError('Finite DoReFa weights required')
    t=w.detach().tanh();maximum=t.abs().max()
    # Explicit zero-tensor extension of the paper's otherwise undefined ratio.
    maximum=torch.where(maximum>0,maximum,torch.ones_like(maximum))
    return t/(2*maximum)+.5,(1-t.square())/maximum


def uniform_dorefa_view(w:Tensor,bits:int=3)->Tensor:
    if type(bits) is not int or not 1<=bits<=16:raise ValueError('Uniform bits must be in [1,16]')
    a,calibration=_dorefa_preprocess(w);levels=2**bits-1
    numeric=2*(a*levels).round()/levels-1
    # The published calibration convention includes output rescaling by two.
    return numeric+(w-w.detach())*calibration


def dorefa_correction(network:FactorizedMetaNetwork,w:Tensor,previous_gradient:Tensor)->Tensor:
    _matching(w,previous_gradient);a,calibration=_dorefa_preprocess(w)
    return network(a.reshape(-1,1)).reshape_as(w)*previous_gradient.detach()*calibration


def validate_config(config):
    from ..config import canonical_bytes
    if type(config) is not dict or set(config)!=set(FIXTURE_CONFIG):raise ValueError('Complete MetaQuant configuration required')
    c=json.loads(canonical_bytes(config));construct_backbone(c['backbone'])
    if c['variant'] not in ('frozen_theta','online_target_training') or c['rotation'] not in ('identity','full_hadamard'):raise ValueError('Explicit frozen/online and storage rotation required')
    if type(c['meta_seed']) is not int or not 0<=c['meta_seed']<2**32:raise ValueError('Pinned framework meta seed required')
    AdamWUpdateRule(betas=tuple(c['meta_betas']),eps=c['meta_eps'])
    if type(c['meta_lr']) not in (int,float) or c['meta_lr']<0:raise ValueError('Nonnegative meta LR required')
    for key in ('widths','scale_recipe','information','normalization','calibration','decay','auxiliary'):
        if c[key]!=FIXTURE_CONFIG[key]:raise ValueError(f'Unsupported MetaQuant convention: {key}')
    return c


class MetaQuantLMAdapter(NoiseAdapter):
    """Prior-U-gradient correction, late original-W update, explicit theta owner.

    Each microbatch rebuilds its graph. The one current target loss is also the
    meta loss: no extra query, HVP or target exposure is manufactured. Priming
    executes and charges an ordinary full forward/backward, skips matrices and
    still updates auxiliaries. Fast weights have no decay; committed base
    updates apply ordinary schedule decay. Auxiliaries use current weights.
    """
    def __init__(self,base:BaseUpdateRule,config:dict):
        self.config=validate_config(config);self.rule=base
        if backbone_config(base)!=self.config['backbone']:raise ValueError('MetaQuant base/config mismatch')

    def bind(self,parameters,groups):
        NoiseAdapter.bind(self,parameters,groups)
        del self.mask_states,self.mask_rng_state
        device=next(iter(parameters.values())).device
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(self.config['meta_seed']).get_state());network=FactorizedMetaNetwork()
        self.theta={f'factor{i}':nn.Parameter(layer.weight.detach().to(device).clone()) for i,layer in enumerate(network.layers)}
        self.theta_gradients={n:None for n in self.theta};self.outer_states={n:{} for n in self.theta}
        self.gprev={};self.rprev={};self.cached_ranges={};self.committed_signal={};self.pending_gradients={}
        self.fast_contexts={};self.fast_directions={};self.fast_lr=0.
        self.logical_target_count=0;self.observations=0;self.theta_updates=0;self.reset_count=0
        self.snapshot_schema['adapter_mutable_state']='metaquant.factors_gradients_prior_U_outer_counters.v1'
        return self

    @property
    def metadata(self):
        return {'ordered_parameter_groups':self.groups,'update_rule_settings':self.config,'state_schema':self.snapshot_schema,
            'schedule_fields':['lr','weight_decay','progress','meta_lr'],'extra_passes':[],
            'priming':'one charged training loss/backward; selected matrices unchanged',
            'fast_calibration':'physical identity STE; cached row range=6*scale; no decay',
            'auxiliary':self.config['auxiliary'],'source_copy_status':'blocked_missing_grant',
            'common_history':'committed-W endpoint observations; not fast-loss forward codes',
            'loss_context_owner':'adapter.fast_contexts contains actual fast quantized forward values/codes',
            'scope':'independent local paper mechanism and named FP4 adaptation; no release parity'}

    def transaction_parameters(self):return self.theta

    def restore_owned_parameters(self,originals=None):
        values=self.theta
        if originals is None:originals={n:nn.Parameter(v.detach().clone()) for n,v in values.items()}
        if set(originals)!=set(values):raise ValueError('MetaQuant factor rollback inventory mismatch')
        with torch.no_grad():
            for n,p in originals.items():p.copy_(values[n]);p.grad=None if self.theta_gradients[n] is None else self.theta_gradients[n].clone()
        self.theta=originals;self.theta_gradients={n:p.grad for n,p in originals.items()}
        for s in self.outer_states.values():
            if s:s['step']=s['step'].cpu()

    def corrections(self,contexts):
        product=self.theta['factor2']@(self.theta['factor1']@self.theta['factor0'])
        if self.config['variant']=='frozen_theta':product=product.detach()
        return {n:(to_storage(self.parameters[n].detach(),self.config['rotation'])/(6*contexts[n].scales)).mul(product.reshape(())).mul(self.gprev[n].detach()) for n in self.groups[0].parameter_names if n in self.gprev}

    def prepare_contexts(self,state,step):
        expected={'name':'row_fixed_max','rotation':self.config['rotation'],'backward':'ste'}
        if state.model._recipe!=expected:raise ValueError('MetaQuant requires named FP4 cached-row physical identity-STE recipe')
        contexts={};fast_contexts={};fast_directions={}
        point=state.schedule['point']
        if point.lr<=0:raise ValueError('MetaQuant adapter requires positive scheduled LR to recover the refined direction')
        for n in self.groups[0].parameter_names:
            c=prepare_context(self.parameters[n],{**expected,**state.model._runtime.get(n,{})},step,None)
            contexts[n]=c
            fast_directions[n]=self.rprev[n].clone() if self.observations else torch.zeros_like(c.storage_u)
            if self.observations:
                fast=c.storage_u-point.lr*self.rprev[n]
                values,codes=quantize_e2m1(fast,c.scales)
                c=replace(c,storage_u=fast.detach().clone(),storage_values=values.detach(),codes=codes.detach(),
                    zero_groups=group_zero_flags(fast,c.recipe),gradient_mask=torch.ones_like(fast))
            fast_contexts[n]=c
        state.model._contexts=contexts;state.model._context_step=step
        self.fast_contexts=fast_contexts;self.fast_directions=fast_directions;self.fast_lr=point.lr
        self.cached_ranges={n:(6*c.scales).clone() for n,c in contexts.items()}
        return contexts

    def prepare_step(self,state,batch):
        MethodAdapter.prepare_step(self,state,batch)
        self.logical_target_count=batch.target_count
        self.pending_gradients={n:torch.zeros_like(self.parameters[n]) for n in self.groups[0].parameter_names}
        for n,p in self.theta.items():p.grad=None;self.theta_gradients[n]=None
        state._work['metaquant_work']={'variant':self.config['variant'],'priming':not bool(self.observations),
            'priming_selected_matrix_updates':0,'priming_loss_target_bytes':0,
            'priming_backward_target_bytes':0,'fast_loss_target_bytes':0,'meta_backward_target_bytes':0,
            'extra_loss_passes':0,'extra_backward_passes':0,'hvp_passes':0,
            'exposure_identity':f'metaquant-step-{state.schedule["step"]}',
            'classification_overlap':'ordinary/priming/fast/meta are labels on the same executed loss/backward; do not sum',
            'loss_forward_context':'adapter.fast_contexts; gprev=current dL/dU_fast through physical identity STE',
            'common_history_context':'committed-W context observed without extra loss/forward/backward',
            'quantized_context_preparations':len(self.fast_contexts),
            'extra_fast_quantizer_calls':len(self.fast_contexts) if self.observations else 0,
            'functional_microbatches':0,'loss_pass_events':[]}

    def functional_views(self,state,schedule):
        corrections=self.corrections(state.model._contexts) if self.observations else {}
        views={};leaves={}
        for n in self.groups[0].parameter_names:
            u=to_storage(self.parameters[n].detach(),self.config['rotation']).clone().requires_grad_();leaves[n]=u
            fast=make_fast_weights(u,corrections[n],self.rprev[n],schedule.lr) if self.observations else u
            numeric=self.fast_contexts[n].storage_values
            physical=numeric+(fast-fast.detach())
            views[n]=from_storage(physical,self.config['rotation'])
        return views,leaves

    def backward_microbatch(self,state,batch):
        views,leaves=self.functional_views(state,state.schedule['point'])
        work=state._work['metaquant_work'];work['functional_microbatches']+=1
        if self.observations:work['fast_loss_target_bytes']+=batch.target_count
        else:work['priming_loss_target_bytes']+=batch.target_count
        event={'execution_id':f'{work["exposure_identity"]}-microbatch-{work["functional_microbatches"]}',
            'target_bytes':batch.target_count,'forward_launched':True,'forward_completed':False,'backward_launched':False,
            'labels':['ordinary_training_loss','fast_loss' if self.observations else 'priming_loss']+(['meta_loss'] if self.observations and self.config['variant']=='online_target_training' else []),
            'parameter_mapping':'exact H04 projection manifest; detached fast_contexts numerical values, fresh correction graph'}
        work['loss_pass_events'].append(event)
        loss=state.summed_nll(batch,parameter_overrides=views)
        event['forward_completed']=True
        def charge(gradient):
            event['backward_launched']=True
            if not self.observations:work['priming_backward_target_bytes']+=batch.target_count
            elif self.config['variant']=='online_target_training':work['meta_backward_target_bytes']+=batch.target_count
            return gradient
        loss.register_hook(charge)
        loss.backward()
        for n,u in leaves.items():
            if u.grad is None:raise ValueError('Missing MetaQuant current quantizer-input gradient')
            self.pending_gradients[n].add_(u.grad.detach())
        if self.observations:
            # Supply the old-theta correction to H06 before its one shared
            # normalize/clip operation. The current U gradient is independent
            # history for the NEXT step, never the current commit signal.
            for n,c in self.corrections(state.model._contexts).items():
                contribution=from_storage(c.detach(),self.config['rotation'])*batch.target_count
                p=self.parameters[n]
                if p.grad is None:p.grad=contribution.clone()
                else:p.grad.add_(contribution)
        self.theta_gradients={n:p.grad for n,p in self.theta.items()}
        return {'summed_nll':float(loss.detach()),'forward_passes':1,'backward_passes':1,'extra_forward_passes':0,'extra_backward_passes':0}

    def finish_step(self,state,schedule):
        priming=not self.observations
        # Capture numerical correction while theta still has its OLD value.
        signal={} if priming else {n:from_storage(c.detach(),self.config['rotation']).clone() for n,c in self.corrections(state.model._contexts).items()}
        outer={}
        if not priming and self.config['variant']=='online_target_training':
            rule=AdamWUpdateRule(betas=tuple(self.config['meta_betas']),eps=self.config['meta_eps'])
            for n,p in self.theta.items():
                if p.grad is None:raise ValueError('Missing MetaQuant factor gradient')
                p.grad.div_(self.logical_target_count)
                outer[n]=rule.propose(p,p.grad,self.outer_states[n],SchedulePoint(self.config['meta_lr'],0.,schedule.progress))
                candidate=p.detach().clone();rule.commit(candidate,outer[n])
                if not bool(torch.isfinite(candidate).all()):raise ValueError('Nonfinite MetaQuant factor candidate')
            for n,p in self.theta.items():rule.commit(p,outer[n]);self.outer_states[n]=outer[n].new_optimizer_state
            self.theta_updates+=1
        self.committed_signal=signal
        # H06 already normalized/clipped old-theta cW with ordinary auxiliaries.
        # committed_signal is the raw pre-theta/pre-clip audit value; proposals
        # consume the actual shared-clipped original-W gradient owner.
        proposals=self.collect_proposals(schedule)
        state.collect_policy_features(proposals,schedule)
        if not priming:
            self.rprev={n:to_storage(-p.gradient_displacement/schedule.lr,self.config['rotation']).detach().clone() if schedule.lr else torch.zeros_like(self.parameters[n]) for n,p in proposals.items() if n in signal}
        else:self.rprev={n:torch.zeros_like(self.parameters[n]) for n in self.groups[0].parameter_names}
        self.gprev={n:(g/self.logical_target_count).detach().clone() for n,g in self.pending_gradients.items()}
        result=self.commit_proposals(proposals)
        self.pending_gradients={};self.logical_target_count=0;self.observations+=1
        self.theta_gradients={n:p.grad for n,p in self.theta.items()}
        state._work['metaquant_work'].update(theta_optimizer_updates=len(outer),theta_candidate_validation_updates=len(outer),
            selected_matrix_updates=0 if priming else len(signal),factor_parameters=sum(p.numel() for p in self.theta.values()))
        return result


def trusted_constructor(parameters,groups,rule,method=None):
    config={**FIXTURE_CONFIG,'backbone':backbone_config(rule)} if method is None else method['config']
    return MetaQuantLMAdapter(rule,config).bind(parameters,groups)


def validate_owned_state(method,groups,step,rule,recipe,weights,contexts):
    c=validate_config(method['config']);matrix=set(groups[0].parameter_names)
    if c['backbone']!=backbone_config(rule) or recipe!={'name':'row_fixed_max','rotation':c['rotation'],'backward':'ste'}:raise ValueError('MetaQuant rule/recipe mismatch')
    if method['observations']!=step or type(method['observations']) is not int or type(method['logical_target_count']) is not int or method['logical_target_count']!=0 or method['pending_gradients']!={} or type(method['reset_count']) is not int or method['reset_count']!=0:raise ValueError('Invalid completed MetaQuant counters/pending/reset state')
    updates=max(0,step-1) if c['variant']=='online_target_training' else 0
    if type(method['theta_updates']) is not int or method['theta_updates']!=updates:raise ValueError('MetaQuant theta-update counter mismatch')
    for key in ('theta','theta_gradients','outer_states'):
        if set(method[key])!=set(FACTOR_SHAPES):raise ValueError('Complete MetaQuant factor/gradient/outer inventory required')
    def array(v,shape):
        if not isinstance(v,Tensor) or v.dtype!=torch.float32 or v.shape!=shape or not bool(torch.isfinite(v).all()):raise ValueError('MetaQuant finite FP32 shape mismatch')
    for n,shape in FACTOR_SHAPES.items():
        array(method['theta'][n],shape);g=method['theta_gradients'][n]
        if updates:array(g,shape)
        elif g is not None:raise ValueError('Frozen/priming factors cannot have gradients')
        s=method['outer_states'][n]
        if not updates:
            if s!={}:raise ValueError('Untrained factors have outer optimizer state')
        else:
            if set(s)!=set(AdamWUpdateRule.optimizer_state_fields) or s['step'].dtype!=torch.float32 or s['step'].shape!=torch.Size([]) or float(s['step'])!=updates:raise ValueError('MetaQuant outer Adam counter/inventory mismatch')
            for k in ('exp_avg','exp_avg_sq'):array(s[k],shape)
            if bool((s['exp_avg_sq']<0).any()):raise ValueError('Negative MetaQuant outer variance')
    for key in ('gprev','rprev','cached_ranges'):
        if set(method[key])!=(matrix if step else set()):raise ValueError('Complete MetaQuant prior/range inventory required')
    if set(method['committed_signal'])!=(matrix if step>1 else set()):raise ValueError('Complete pre-theta committed signal required')
    for key in ('fast_contexts','fast_directions'):
        if set(method[key])!=(matrix if step else set()):raise ValueError('Complete fast-context/direction inventory required')
    if type(method['fast_lr']) not in (int,float) or not math.isfinite(method['fast_lr']) or (method['fast_lr']<=0 if step else method['fast_lr']!=0.):raise ValueError('Invalid completed fast LR')
    for n in method['gprev']:
        for key in ('gprev','rprev'):array(method[key][n],weights[n].shape)
        array(method['cached_ranges'][n],(weights[n].shape[0],1))
        if not torch.equal(method['cached_ranges'][n],6*contexts[n].scales):raise ValueError('MetaQuant cached physical row range mismatch')
        if step==1 and bool(method['rprev'][n].any()):raise ValueError('Priming refined direction must be zero')
        if step>1:array(method['committed_signal'][n],weights[n].shape)
        from ..quantization.context import QuantizationContext
        f=method['fast_contexts'][n];base=contexts[n]
        if type(f) is not QuantizationContext or (f.recipe,f.rotation,f.backward,f.step)!=(base.recipe,base.rotation,'ste',step-1):raise ValueError('Fast context recipe/step mismatch')
        array(method['fast_directions'][n],weights[n].shape)
        expected_fast=base.storage_u-method['fast_lr']*method['fast_directions'][n]
        if not torch.equal(f.storage_u,expected_fast) or not torch.equal(f.scales,base.scales):raise ValueError('Fast numerical displacement/range mismatch')
        values,codes=quantize_e2m1(f.storage_u,f.scales)
        if not torch.equal(f.codes,codes) or not torch.equal(f.storage_values,values) or not torch.equal(f.gradient_mask,torch.ones_like(values)) or not torch.equal(f.zero_groups,group_zero_flags(f.storage_u,f.recipe)):raise ValueError('Fast quantized context mismatch')
        s=method['optimizer_states'][n]
        if step==1 and s:raise ValueError('Priming selected matrices have optimizer moments')
        if step>1 and type(rule) is AdamWUpdateRule and float(s['step'])!=step-1:raise ValueError('Selected matrix counter includes priming')


def original_mechanism_fixture(device='cpu'):
    """Two target-loss steps, paper DoReFa calibration and late SGD commit.

    Independent arithmetic fixture: squared scalar prediction, not author
    CIFAR/ImageNet execution, release software parity or test-best selection.
    """
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(1010).get_state());net=FactorizedMetaNetwork().to(device)
    w=torch.tensor([[-.8,.3,1.2]],device=device,requires_grad=True)
    x=torch.tensor([[.2,-.4,.8]],device=device);target=torch.tensor(.3,device=device);lr=.03
    # First selected batch only observes the quantized-output gradient.
    q=uniform_dorefa_view(w,3);q.retain_grad();loss=(q.mul(x).sum()-target).square()/2;loss.backward()
    previous=q.grad.detach().clone();before=w.detach().clone();rprev=torch.zeros_like(w)
    assert torch.equal(w.detach(),before)
    correction=dorefa_correction(net,w,previous)
    pre=correction.detach().clone();fast=make_fast_weights(w,correction,rprev,lr)
    torch.testing.assert_close(fast,w-lr*rprev,atol=2e-6,rtol=2e-5)
    qfast=uniform_dorefa_view(fast,3);qfast.retain_grad()
    current_loss=(qfast.mul(x).sum()-target).square()/2
    actual=torch.autograd.grad(current_loss,tuple(net.parameters()),retain_graph=True)
    # Independent theta derivative: fixed numerical quantized output and the
    # paper's calibration at the fast numerical W multiply -eta*dc/dtheta.
    _,calibration=_dorefa_preprocess(fast)
    upstream=(qfast.detach().mul(x).sum()-target)*x*calibration
    independent=torch.autograd.grad((-lr*upstream.detach()*correction).sum(),tuple(net.parameters()),retain_graph=True)
    for a,e in zip(actual,independent):torch.testing.assert_close(a,e,atol=2e-6,rtol=2e-5)
    current_loss.backward()
    meta=AdamWUpdateRule(betas=(.9,.999))
    for p in net.parameters():
        proposal=meta.propose(p,p.grad,{},SchedulePoint(.001,0.,0.));meta.commit(p,proposal)
    # SGD refinement is identity; late update uses correction computed before
    # meta Adam, not a new correction evaluated with updated factors.
    with torch.no_grad():w.sub_(lr*pre)
    torch.testing.assert_close(w,before-lr*pre)
    changed=dorefa_correction(net,before,previous).detach()
    assert not torch.equal(changed,pre)
    wrong=before.clone().requires_grad_()-lr*rprev
    erased=torch.autograd.grad(wrong.square().sum(),tuple(net.parameters()),allow_unused=True)
    assert all(g is None for g in erased)
    return {'status':'pass','quantizer':'DoReFa uniform 3-bit','calibration':'(1-tanh(W)^2)/max(abs(tanh(W))); zero denominator extension one',
        'factor_parameters':10200,'factor_gradient_parity':'pass','theta_derivative':'independent_chain_rule_pass',
        'fast_value':'U-lr*rprev','simplified_negative_control':'meta_gradient_erased',
        'priming_selected_matrix_updates':0,'late_pretheta_commit':'pass','training_losses':2,'backward_passes':2,
        'additional_gradient_only_executions':2,'executed_pass_identity':'two loss backwards plus actual/independent factor-gradient audits; latter are separate gradient-only executions',
        'task':'synthetic scalar squared prediction','original_release_parity_claimed':False}


def check_metaquant(device,output):
    """Frozen risk-based scope: actual first-order functional H04 route + replay.

    Only 1-layer width8/head2, two 4-byte blocks, microbatch1, three steps,
    row-fixed FP4, identity/full-Hadamard and frozen/online are admitted here.
    This is no accepted Cartesian rerun, no higher-order operation assertion.
    """
    import hashlib,os
    from pathlib import Path
    from ..config import canonical_bytes
    from ..records import ArtifactRef,sha256_file
    from ..models.transformer import ByteTransformer,ModelConfig
    from ..optim.base import default_parameter_groups
    from ..data.articles import Article
    from ..data.blocks import tile_article
    from ..data.batches import LogicalBatch
    from ..train import TrainingState,train_step,TrainingFailure
    from ..state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    from ..evaluate import score_blocks,summed_nll
    root=Path(__file__).resolve().parents[3]
    report={'suite':'metaquant','device':device,'status':'pass','blockers':[],'scientific_gate':None,
        'source_copy_status':'blocked_missing_grant','release_reproduction_status':'unverified','blocked_is_defeated_baseline':False,
        'scope':'Independent paper fixture and named FP4 H04 first-order functional learning mechanism. Three steps incl priming, synthetic 1x8x2 model, two4-byte blocks/microbatch1, identity and full-Hadamard, frozen and online. No universal MPS, high-order, production-model/horizon or comparative performance claim.',
        'predeclared_bounds':{'factor_value_atol':2e-6,'factor_value_rtol':2e-5,'factor_gradient_atol':2e-6,'factor_gradient_rtol':2e-5,
            'replay':'same-device exact complete scientific state','fallback':'0 required on actual MPS'},
        'source_hashes':{str(p.relative_to(root)):sha256_file(p) for p in sorted(root.glob('src/ralup/**/*.py'))},'variants':[]}
    if device not in ('cpu','mps'):raise ValueError('Unknown MetaQuant device')
    if device=='mps' and (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or not torch.backends.mps.is_available()):
        report.update(status='blocked',blockers=['Actual MPS unavailable or fallback not zero before startup'])
    else:
        try:
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator().manual_seed(1010).get_state());net=FactorizedMetaNetwork().to(device)
            x=torch.tensor([[-.8],[.3],[.9]],device=device);up=torch.tensor([[1.],[2.],[-3.]],device=device)
            unfused=x
            for layer in net.layers:unfused=layer(unfused)
            torch.testing.assert_close(net(x),unfused,atol=2e-6,rtol=2e-5)
            a=torch.autograd.grad((net(x)*up).sum(),tuple(net.parameters()))
            e=torch.autograd.grad((unfused*up).sum(),tuple(net.parameters()))
            for actual,expected in zip(a,e):torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
            report['original_mechanism']=original_mechanism_fixture(device)
            blocks=[b for raw in (b'abcd',b'efgh') for b in tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))]
            batch=LogicalBatch(blocks,8)
            for variant in ('frozen_theta','online_target_training'):
                for rotation in ('identity','full_hadamard'):
                    record={'variant':variant,'rotation':rotation,'status':'unverified','operation_status':'unverified','completed_steps':0,'work':[]}
                    report['variants'].append(record)
                    with torch.random.fork_rng(devices=[]):
                        torch.set_rng_state(torch.Generator().manual_seed(1010).get_state());model=ByteTransformer(ModelConfig(1,8,2)).to(device)
                    model.configure_quantization({'name':'row_fixed_max','rotation':rotation,'backward':'ste'})
                    method=MetaQuantLMAdapter(AdamWUpdateRule(),{**FIXTURE_CONFIG,'variant':variant,'rotation':rotation}).bind(dict(model.named_parameters()),default_parameter_groups(model))
                    state=TrainingState(model,method,microbatch=1,schedule={'step':0,'point':SchedulePoint(.002,.1,0.)})
                    original={n:p.detach().clone() for n,p in method.parameters.items()}
                    try:
                        first=train_step(state,batch);record['work'].append(first.work);record['completed_steps']=1
                        for n in method.gprev:assert torch.equal(original[n],method.parameters[n])
                        before=exact_cpu_state_hash(state);views,_=method.functional_views(state,state.schedule['point'])
                        value=summed_nll(model,batch,parameter_overrides=views)
                        assert value.requires_grad and exact_cpu_state_hash(state)==before
                        score_blocks(model,blocks,evaluation_context={'partition':'fixture'})
                        assert exact_cpu_state_hash(state)==before
                        resumed=restore_snapshot(capture_snapshot(state),device)
                        for step in (2,3):
                            result=train_step(state,batch);record['work'].append(result.work);record['completed_steps']=step
                            train_step(resumed,batch)
                            assert exact_cpu_state_hash(state)==exact_cpu_state_hash(resumed)
                        record.update(status='pass',operation_status='pass',same_device_complete_state_replay=True,
                            evaluation_read_only=True,functional_mapping_read_only=True,original_W_moments=True,
                            factor_parameters=10200,theta_updates=method.theta_updates,committed_signal_pretheta=True,
                            executed_logical_steps_including_replay=5,additional_readonly_functional_loss_passes=1,
                            additional_scoring_loss_passes=1,loss_target_exposures=56,
                            replay_work=[e for e in resumed.ledger.events if e['step']>=1])
                    except TrainingFailure as error:
                        record.update(status='blocked',operation_status='blocked',failure=str(error.__cause__),failed_work=state.ledger.events[-1])
                        if device=='mps' and isinstance(error.__cause__,RuntimeError) and any(s in str(error.__cause__).lower() for s in ('not implemented','not supported','does not support')):
                            report['status']='blocked';report['blockers'].append(str(error.__cause__))
                        else:raise
        except (AssertionError,ValueError,RuntimeError) as error:
            report.update(status='fail',blockers=[f'{type(error).__name__}: {error}'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(canonical_bytes(report))
    return ArtifactRef(output,sha256_file(output))
