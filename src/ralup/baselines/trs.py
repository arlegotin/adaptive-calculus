"""Independent TRS release equations and a named original-W FP4 adaptation.

GPL source was inspected, not copied. Release multiplicative feedback differs
from paper Eq.11's additive feedback. Release fixture uses fixed sample-std scale
and coupled Adam decay; backbone adapter uses FP4 normalized storage values,
fixed row scales, logical boundary observations and separate ordinary-LR decay.
"""
from dataclasses import dataclass,replace
import math,json
import torch
from torch import Tensor
from ..optim.base import MethodAdapter,BaseUpdateRule,Proposal,SchedulePoint
from ..optim.adamw import AdamWUpdateRule
from ..optim.muon import MuonUpdateRule
from ..optim.muon_plus import MuonPlusUpdateRule

BACKBONES={'adamw_scalar_fp32':AdamWUpdateRule,'muon_project_fp32':MuonUpdateRule,'muon_plus_fp32':MuonPlusUpdateRule}
FIXTURE_CONFIG={'mu':.99,'target_schedule':{'kind':'constant','start':.1,'end':.1},
    'normalization':'storage_values_div_scale_div_6','scale_recipe':'row_fixed_max',
    'decay':'separate_ordinary_lr_pre_displacement',
    'backbone':{'name':'adamw_scalar_fp32','config':{'betas':[.9,.99],'eps':1e-8}}}
ADAPTER_FIELDS=('optimizer_states','proposals','snapshot_schema','config','trs_states','pending_codes')


@dataclass
class TrsState:
    r: float
    K: float
    previous_codes: Tensor
    observations: int

    def __post_init__(self):
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (self.r,self.K)):
            raise ValueError('TRS r/K must be finite nonnegative scalars')
        if type(self.observations) is not int or self.observations<0:
            raise ValueError('TRS observations must be a nonnegative integer')
        if not isinstance(self.previous_codes,Tensor) or not self.previous_codes.numel() or not self.previous_codes.is_floating_point() or not bool(torch.isfinite(self.previous_codes).all()):
            raise ValueError('TRS previous codes require finite normalized tensor values')


def trs_effective_lr(state: TrsState, observed_rate: float, target_rate: float, alpha: float, mu: float) -> tuple[float,TrsState]:
    """Pure released feedback; first measured rate is ignored.

    A caller publishes new normalized codes once, at its logical commit boundary.
    Source's mu=0 disabled branch uses ordinary group LR (the target schedule).
    """
    state.__post_init__()
    for value in (observed_rate,target_rate,alpha,mu):
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('TRS inputs must be finite and nonnegative')
    if observed_rate>1 or target_rate>1 or mu>=1:raise ValueError('TRS rates require [0,1], mu [0,1)')
    r,K=float(state.r),float(state.K)
    if mu==0:lr=target_rate
    elif state.observations==0:lr=alpha
    else:
        K=mu*K+(1-mu)*observed_rate
        r=mu*r+(1-mu)*r*target_rate/(K+1e-8)
        lr=alpha*r
    if not math.isfinite(lr):raise ValueError('Nonfinite TRS feedback')
    return lr,TrsState(r,K,state.previous_codes.detach().clone(),state.observations+1)


def release_normalized_codes(w,scale,bits=4,symmetric=False):
    """Released nonbinary normalized levels; physical scale never enters history."""
    if type(bits) is not int or not 2<=bits<=16 or type(symmetric) is not bool:raise ValueError('Fixture requires declared nonbinary bit/symmetry convention')
    if not bool(torch.isfinite(w).all()) or not bool(torch.isfinite(scale).all()):raise ValueError('Nonfinite release quantizer operands')
    power=2**(bits-1);low=-(power-1) if symmetric else -power
    return torch.round((w.detach()/(scale.detach().abs()+1e-6)*power).clamp(low,power-1))/power


class TrsReleaseFixture:
    """Bounded clean Adam/nonbinary release mechanics, not copied CNN software."""
    def __init__(self,w,*,target_rate,alpha,mu=.99,betas=(.9,.999),eps=1e-8,bits=4,symmetric=False):
        self.scale=(3*w.detach().std(correction=1)).clone()
        self.state=TrsState(1.,target_rate,torch.zeros_like(w),0)
        self.alpha,self.mu,self.betas,self.eps=alpha,mu,tuple(betas),eps
        self.bits,self.symmetric=bits,symmetric
        self.optimizer_state={};self.last_lr=None
        AdamWUpdateRule(betas=self.betas,eps=eps)
        trs_effective_lr(self.state,0.,target_rate,alpha,mu)
        release_normalized_codes(w,self.scale,bits,symmetric)

    @torch.no_grad()
    def step(self,w,grad,*,target_rate,weight_decay=0.):
        SchedulePoint(self.alpha,weight_decay,0.)
        if w.dtype!=torch.float32 or grad.dtype!=w.dtype or grad.shape!=w.shape or grad.device!=w.device or not bool(torch.isfinite(grad).all()):raise ValueError('Release fixture requires matching finite FP32 W/gradient')
        codes=release_normalized_codes(w,self.scale,self.bits,self.symmetric)
        observed=float((codes!=self.state.previous_codes).float().mean())
        lr,new=trs_effective_lr(self.state,observed,target_rate,self.alpha,self.mu)
        new.previous_codes=codes.clone()
        # Independent scalar Adam: source add-based moment arithmetic, coupled decay.
        old=self.optimizer_state
        step=(int(old['step'].item()) if old else 0)+1
        g=grad.detach()+weight_decay*w.detach()
        b1,b2=self.betas
        m=(old['exp_avg'].clone() if old else torch.zeros_like(w)).mul_(b1).add_(g,alpha=1-b1)
        v=(old['exp_avg_sq'].clone() if old else torch.zeros_like(w)).mul_(b2).addcmul_(g,g,value=1-b2)
        denom=(v.sqrt()/math.sqrt(1-b2**step)).add_(self.eps)
        candidate=w.detach().clone().addcdiv_(m,denom,value=-lr/(1-b1**step))
        if not bool(torch.isfinite(candidate).all()):raise ValueError('Nonfinite release update')
        w.copy_(candidate);self.optimizer_state={'step':torch.tensor(float(step)),'exp_avg':m,'exp_avg_sq':v}
        self.state=new;self.last_lr=lr
        return {'effective_lr':lr,'observed_rate':observed}


def backbone_config(base):
    if type(base) is AdamWUpdateRule:return {'name':'adamw_scalar_fp32','config':{'betas':list(base.betas),'eps':base.eps}}
    for name,kind in BACKBONES.items():
        if type(base) is kind:return {'name':name,'config':json.loads(json.dumps(dict(base.configuration)))}
    raise ValueError('TRS requires a trusted original-W AdamW or Muon backbone')


def construct_backbone(envelope):
    if type(envelope) is not dict or set(envelope)!={'name','config'} or envelope['name'] not in BACKBONES or type(envelope['config']) is not dict:raise ValueError('Unknown TRS backbone envelope')
    kind=BACKBONES[envelope['name']]
    if kind is AdamWUpdateRule:
        if set(envelope['config'])!={'betas','eps'} or type(envelope['config']['betas']) is not list:raise ValueError('TRS AdamW config requires public list betas and epsilon')
        rule=kind(betas=tuple(envelope['config']['betas']),eps=envelope['config']['eps'])
    else:
        if set(envelope['config'])!=set(kind.defaults):raise ValueError('TRS matrix backbone requires complete configuration')
        rule=kind(envelope['config'])
    if backbone_config(rule)!=envelope:raise ValueError('TRS backbone configuration mismatch')
    return rule


def validate_config(config):
    if type(config) is not dict or set(config)!=set(FIXTURE_CONFIG):raise ValueError('TRS requires complete explicit configuration')
    from ..config import canonical_bytes
    try:c=json.loads(canonical_bytes(config))
    except (TypeError,ValueError) as error:raise ValueError('TRS configuration requires strict finite canonical JSON') from error
    if type(c['target_schedule']) is not dict:raise ValueError('TRS target schedule must be an object')
    construct_backbone(c['backbone'])
    if type(c['mu']) not in (int,float) or not math.isfinite(c['mu']) or not 0<=c['mu']<1:raise ValueError('Invalid TRS mu')
    for key in ('normalization','scale_recipe','decay'):
        if c[key]!=FIXTURE_CONFIG[key]:raise ValueError(f'Unsupported TRS convention: {key}')
    s=c['target_schedule']
    if set(s)!={'kind','start','end'} or s['kind'] not in ('constant','linear'):raise ValueError('Explicit registered TRS target schedule required')
    if any(type(s[k]) not in (int,float) or not math.isfinite(s[k]) or not 0<=s[k]<=1 for k in ('start','end')):raise ValueError('Invalid TRS target schedule rates')
    if s['kind']=='constant' and s['start']!=s['end']:raise ValueError('Constant target schedule endpoints must agree')
    c['backbone']=backbone_config(construct_backbone(c['backbone']))
    return c


def target_rate(config,progress):
    if not math.isfinite(progress) or not 0<=progress<=1:raise ValueError('Invalid target schedule progress')
    s=config['target_schedule']
    if s['kind']=='constant' or progress==0:return s['start']
    if progress==1:return s['end']
    return s['start']+(s['end']-s['start'])*progress


def normalized_codes(context):
    """Compare numerical FP4 levels, not encoding IDs or physical weights.

    storage_values=s*E2M1; divide by detached cached s and max level 6.
    Duplicate signed-zero IDs both compare as numerical zero. Fixed row scales
    are mandatory for this named adaptation; row and Hadamard coordinates follow
    the context's declared storage geometry, never original-W optimizer moments.
    """
    return (context.storage_values.detach()/context.scales.detach()/6).clone()


class TrsBackboneAdapter(MethodAdapter):
    def __init__(self,base:BaseUpdateRule,config:dict):
        self.config=validate_config(config)
        if backbone_config(base)!=self.config['backbone']:raise ValueError('TRS config must identify the actual wrapped backbone')
        self.rule=base

    def bind(self,parameters,groups):
        if tuple(g.name for g in groups) not in (('internal_projection',),('internal_projection','auxiliary_no_decay')) or groups[0].weight_decay is not None or (len(groups)>1 and groups[1].weight_decay!=0.):raise ValueError('TRS requires explicit projection/no-decay auxiliary groups')
        MethodAdapter.__init__(self,parameters,groups,self.rule)
        self.trs_states={};self.pending_codes={}
        self.snapshot_schema['optimizer_states']=f'named.original_W.{self.rule.state_schema}+auxiliary.adamw.v1'
        self.snapshot_schema['adapter_mutable_state']='trs.normalized_codes_scalars.v1'
        return self

    @property
    def metadata(self):
        # Public metadata must already be canonical JSON-compatible, including
        # scalar-backbone betas and every ordered group/sequence.
        value=super().metadata
        value['ordered_parameter_groups']=[{'name':g.name,'parameter_names':list(g.parameter_names),'weight_decay':g.weight_decay} for g in self.groups]
        value['update_rule_settings']=json.loads(json.dumps(value['update_rule_settings']))
        value['extra_passes']=list(value['extra_passes'])
        value.update(schedule_fields=['lr','weight_decay','progress','target_rate'],trs_configuration=validate_config(self.config),supported_backbone_families=list(BACKBONES),
                     unsupported_backbone_families=['SOAP','Madam','arbitrary_unregistered_BaseUpdateRule'],
                     decay='ordinary lr decay before displacement; feedback changes gradient displacement only')
        return value

    def rule_for_parameter(self,name):
        if name in self.groups[0].parameter_names or type(self.rule) is AdamWUpdateRule:return self.rule
        return self.rule.auxiliary_rule

    def schedule_for_group(self,group,schedule):
        if group.name=='auxiliary_no_decay' and type(self.rule) is not AdamWUpdateRule:return SchedulePoint(self.rule.configuration['auxiliary_lr'],0.,schedule.progress)
        return super().schedule_for_group(group,schedule)

    def prepare_step(self,training_state,logical_batch):
        super().prepare_step(training_state,logical_batch)
        if self.pending_codes:raise ValueError('Uncommitted TRS codes remain')
        contexts=training_state.model._contexts
        if set(contexts)!=set(self.groups[0].parameter_names) or not training_state.model.training:raise ValueError('TRS requires training projection contexts')
        for name,c in contexts.items():
            if c.recipe!=self.config['scale_recipe']:raise ValueError('TRS requires declared fixed row scale recipe')
            self.pending_codes[name]=normalized_codes(c)

    def _feedback(self,name,schedule):
        if name not in self.pending_codes:raise ValueError('TRS proposal requires one prepared logical context')
        codes=self.pending_codes[name];rate=target_rate(self.config,schedule.progress)
        old=self.trs_states.get(name,TrsState(1.,rate,torch.zeros_like(codes),0))
        observed=float((codes!=old.previous_codes).float().mean())
        lr,new=trs_effective_lr(old,observed,rate,schedule.lr,self.config['mu'])
        new.previous_codes=codes.clone()
        return lr,new,{'effective_lr':lr,'target_rate':rate,'observed_rate':observed,'r':new.r,'K':new.K,'observations':new.observations}

    def collect_proposals(self,schedule,*,work_recorder=None):
        if self.proposals:raise ValueError('Uncommitted proposals remain')
        proposals={}
        for group in self.groups:
            ordinary=self.schedule_for_group(group,schedule)
            for name in group.parameter_names:
                w=self.parameters[name]
                if w.grad is None:continue
                if not bool(torch.isfinite(w.grad).all()):raise ValueError('TRS requires finite original-W gradients')
                lr=self._feedback(name,ordinary)[0] if group.name=='internal_projection' else ordinary.lr
                # Remove backbone decay entirely, then apply separate ordinary LR
                # decay to reference. r affects only the gradient displacement.
                p=self.rule_for_parameter(name).propose(w,w.grad,self.optimizer_states[name],SchedulePoint(lr,0.,ordinary.progress))
                proposals[name]=replace(p,reference_after_decay=w.detach().clone().mul_(1-ordinary.lr*ordinary.weight_decay))
        self.proposals=proposals
        return proposals

    def finish_step(self,training_state,schedule):
        proposals=self.collect_proposals(schedule)
        training_state.collect_policy_features(proposals,schedule)
        next_states={};telemetry={}
        for name in self.groups[0].parameter_names:
            if name not in proposals:raise ValueError('Missing TRS projection gradient')
            _,next_states[name],telemetry[name]=self._feedback(name,schedule)
        result=self.commit_proposals(proposals)
        self.trs_states=next_states;self.pending_codes={}
        return {**result,'trs':telemetry}


def trusted_constructor(parameters,groups,rule):
    return TrsBackboneAdapter(rule,{**FIXTURE_CONFIG,'backbone':backbone_config(rule)}).bind(parameters,groups)


def validate_owned_state(method,groups,step,contexts,rule):
    c=validate_config(method['config'])
    if backbone_config(rule)!=c['backbone'] or method['pending_codes']:raise ValueError('TRS wrapped rule or pending context mismatch')
    expected=set(groups[0].parameter_names)
    if set(method['trs_states'])!=(expected if step else set()):raise ValueError('Incomplete TRS feedback inventory')
    for name,s in method['trs_states'].items():
        if type(s) is not TrsState:raise ValueError('Unknown TRS state owner')
        s.__post_init__();context=contexts[name]
        if s.observations!=step or s.previous_codes.dtype!=torch.float32 or s.previous_codes.shape!=context.codes.shape or context.recipe!=c['scale_recipe']:raise ValueError('TRS state step/geometry/recipe mismatch')
        if not torch.equal(s.previous_codes,normalized_codes(context)):raise ValueError('TRS normalized code history differs from committed context')


def check_trs(device,output):
    """Small equation/accumulation gates with declared scope; no source execution."""
    import hashlib,json,os
    from dataclasses import asdict
    from pathlib import Path
    from ..config import canonical_bytes
    from ..records import ArtifactRef,sha256_file
    from ..optim.base import default_parameter_groups,ParameterGroup
    from ..data.articles import Article
    from ..data.blocks import tile_article
    from ..data.batches import LogicalBatch
    from ..models.transformer import ByteTransformer,ModelConfig
    from ..train import TrainingState,train_step
    from ..state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    from ..evaluate import score_blocks
    from .registry import trs_record
    root=Path(__file__).resolve().parents[3]
    report={'suite':'trs','device':device,'status':'pass','blockers':[],
        'scope':'clean released-equation fixture plus 1x8x2 synthetic 16-target logical accumulation and same-device complete-state replay; no author release execution, large-model/horizon gate, pilot relevance, tuning or science eligibility',
        'source_copy_status':'blocked_GPL_inspection_only','source_hashes':{str(p.relative_to(root)):sha256_file(p) for p in sorted(root.glob('src/ralup/**/*.py'))},
        'ordinary_lr_schedule':['lr','weight_decay','progress'],'target_rate_schedule':FIXTURE_CONFIG['target_schedule'],'scientific_gate':None,'variants':[]}
    if device not in ('cpu','mps'):raise ValueError('Unknown TRS device')
    if device=='mps' and (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or not torch.backends.mps.is_available()):
        report.update(status='blocked',blockers=['Actual MPS unavailable or fallback not zero before startup'])
    else:
        try:
            w=torch.tensor([[.3,-.2,.1,-.4]],device=device)
            fixture=TrsReleaseFixture(w,target_rate=.1,alpha=.01,mu=.99,bits=3)
            first=fixture.step(w,torch.tensor([[.1,-.1,.3,-.2]],device=device),target_rate=.1,weight_decay=.1)
            torch.testing.assert_close(w,torch.tensor([[.29000002,-.19,.09,-.39000002]],device=device),atol=2e-8,rtol=1e-7)
            torch.testing.assert_close(fixture.optimizer_state['exp_avg'],torch.tensor([[.013,-.012,.031,-.024]],device=device),atol=1e-8,rtol=1e-7)
            second=fixture.step(w,torch.tensor([[-.3,.2,-.1,.4]],device=device),target_rate=.1,weight_decay=.1)
            torch.testing.assert_close(w,torch.tensor([[.29381345,-.19250204,.08566745,-.39248970]],device=device),atol=5e-7,rtol=1e-6)
            s=TrsState(1.,.1,torch.zeros(4,device=device),0)
            lrs=[]
            for k,R in ((.75,.1),(.5,.1),(0.,.2)):
                lr,s=trs_effective_lr(s,k,R,.01,.99);lrs.append(lr)
            assert math.isclose(lrs[0],.01,rel_tol=0,abs_tol=0)
            assert math.isclose(lrs[1],.009996153836908285,rel_tol=1e-12)
            assert math.isclose(lrs[2],.010090367762137316,rel_tol=1e-12)
            report['release_fixture']={'status':'pass','first_rate_ignored':True,'effective_lrs':lrs,'coupled_decay':True,'fixed_scale':float(fixture.scale),'first':first,'second':second,'final_weight':w.cpu().tolist(),
                'bounds':{'weight_atol':5e-7,'weight_rtol':1e-6,'scalar_feedback_rtol':1e-12}}
            blocks=[b for raw in (b'abcd',b'efgh',b'ijkl',b'mnop') for b in tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))]
            batch=LogicalBatch(blocks,16)
            def create(packing):
                with torch.random.fork_rng():
                    torch.manual_seed(404);model=ByteTransformer(ModelConfig(1,8,2)).to(device)
                model.configure_quantization({'name':'row_fixed_max'})
                method=TrsBackboneAdapter(AdamWUpdateRule(),FIXTURE_CONFIG).bind(dict(model.named_parameters()),default_parameter_groups(model))
                return TrainingState(model,method,microbatch=packing,schedule={'step':0,'point':SchedulePoint(.001,.1,0.)})
            one,four=create(1),create(4);maximum_error=0.
            for step in range(3):
                a=train_step(one,batch);b=train_step(four,batch)
                assert one.histories['loss_updates']==four.histories['loss_updates']==step+1
                for name,s in one.method.trs_states.items():
                    peer=four.method.trs_states[name]
                    assert s.observations==peer.observations==step+1 and s.r==peer.r and s.K==peer.K
                    assert torch.equal(s.previous_codes,peer.previous_codes)
                    assert a.telemetry['trs'][name]['effective_lr']==b.telemetry['trs'][name]['effective_lr']
                for name,p in one.model.named_parameters():
                    other=dict(four.model.named_parameters())[name]
                    torch.testing.assert_close(p,other,atol=3e-7,rtol=3e-5)
                    maximum_error=max(maximum_error,float((p-other).detach().abs().max()))
            before=exact_cpu_state_hash(one)
            score_blocks(one.model,blocks,evaluation_context={'partition':'fixture'})
            assert exact_cpu_state_hash(one)==before
            resumed=restore_snapshot(capture_snapshot(one),device)
            train_step(one,batch);train_step(resumed,batch)
            assert exact_cpu_state_hash(one)==exact_cpu_state_hash(resumed)
            report['accumulation']={'status':'pass','logical_steps':3,'targets_per_step':16,'packing':[1,4],'lr_feedback_exact':True,'parameter_max_error':maximum_error,'bounds':{'atol':3e-7,'rtol':3e-5},'same_device_replay_exact':True,'evaluation_read_only':True,
                'work':{'original_branch':{'stream':one.ledger.stream,'passes':one.ledger.passes,'elapsed':one.ledger.elapsed},'packed_branch':{'stream':four.ledger.stream,'passes':four.ledger.passes,'elapsed':four.ledger.elapsed},'resumed_branch':{'stream':resumed.ledger.stream,'passes':resumed.ledger.passes,'elapsed':resumed.ledger.elapsed}},'CPU_MPS_bitwise_parity_claimed':False}
        except AssertionError as error:report.update(status='fail',blockers=[str(error)])
        except (RuntimeError,ValueError) as error:report.update(status='blocked',blockers=[str(error)])
        for name in ('trs_release_fixture','trs_backbone_adapter'):
            method=asdict(trs_record(name,(ParameterGroup('internal_projection',('fixture',)),),FIXTURE_CONFIG,parity_status=report['status'],operations_status='pass' if report['status']=='pass' else 'blocked'))
            report['variants'].append({'name':name,'method':json.loads(json.dumps(method))})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(canonical_bytes(report)+b'\n')
    return ArtifactRef(output,sha256_file(output))
