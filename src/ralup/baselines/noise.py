"""Clean equations: damping adaptation and practical SkipUpdate/Magma.

Magma: arXiv:2602.15322v1 Algorithm 1/section 3, article CC BY 4.0.
No author software was copied. Initial .5, zero-cosine convention, matrix
projection scope and separately applied decay are explicit project choices.
"""
import json
import math
import torch
from torch import Tensor
from ..optim.base import MethodAdapter, SchedulePoint
from ..optim.adamw import AdamWUpdateRule
from .trs import backbone_config, construct_backbone

MASK_CONFIG={'mode':'magma','temperature':2.,'probability':.5,'initial_alignment':.5,
             'ema':.9,'zero_cosine':'zero','scope':'matrix','decay':'separate_gradient_only'}
FIXTURE_CONFIG={**MASK_CONFIG,'coefficient':0.,'scale_recipe':'row_fixed_max',
                'backbone':{'name':'adamw_scalar_fp32','config':{'betas':[.9,.99],'eps':1e-8}}}
ADAPTER_FIELDS=('optimizer_states','proposals','snapshot_schema','config','mask_states','mask_rng_state')


def _number(value,minimum=0):
    return type(value) in (int,float) and math.isfinite(value) and value>=minimum


def damping_penalty(w:Tensor,quantized:Tensor,coefficient:float)->Tensor:
    """lambda/2 ||W-stopgrad(Q(W))||²; Q has no damping gradient."""
    if not _number(coefficient) or w.shape!=quantized.shape or w.device!=quantized.device or w.dtype!=torch.float32 or quantized.dtype!=w.dtype or not bool(torch.isfinite(w).all()&torch.isfinite(quantized).all()):
        raise ValueError('Damping requires finite matching FP32 operands and nonnegative coefficient')
    return coefficient/2*(w-quantized.detach()).square().sum()


def validate_mask(config):
    if type(config) is not dict or set(config)!=set(MASK_CONFIG):raise ValueError('Complete mask configuration required')
    c=json.loads(json.dumps(config,allow_nan=False))
    if c['mode'] not in ('simple','magma'):raise ValueError('Unknown mask mode')
    if not _number(c['temperature']) or c['temperature']==0:raise ValueError('Temperature must be positive')
    for k in ('probability','initial_alignment','ema','zero_cosine','scope','decay'):
        if c[k]!=MASK_CONFIG[k]:raise ValueError(f'Unsupported practical mask convention: {k}')
    return c


@torch.no_grad()
def mask_displacement(displacement:Tensor,moment:Tensor,gradient:Tensor,state:dict,uniform:float,config:dict)->tuple[Tensor,dict]:
    """Pure matrix decision; current original-W moment, dense EMA on rejection."""
    c=validate_mask(config)
    if not _number(uniform) or uniform>=1:raise ValueError('Uniform must be in [0,1)')
    if any(v.shape!=displacement.shape or v.device!=displacement.device or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in (displacement,moment,gradient)) or not displacement.numel():raise ValueError('Mask requires matching finite FP32 tensors')
    validate_mask_state(state,allow_empty=True)
    alignment=state.get('alignment',.5);observations=state.get('observations',0)
    # Scale each operand first to avoid overflow/underflow of norm products.
    a=moment.abs().max();b=gradient.abs().max()
    if float(a)==0 or float(b)==0:cosine=0.
    else:
        m=moment/a;g=gradient/b
        cosine=float((m*g).sum()/(m.square().sum().sqrt()*g.square().sum().sqrt()))
        cosine=max(-1.,min(1.,cosine))
    if c['mode']=='magma':alignment=.9*alignment+.1*float(torch.sigmoid(torch.tensor(cosine/c['temperature'],dtype=torch.float64)))
    accepted=uniform<.5
    multiplier=(2. if c['mode']=='simple' else alignment) if accepted else 0.
    return displacement*multiplier,{'alignment':alignment,'observations':observations+1,'cosine':cosine,'probability':.5,'accepted':accepted,'multiplier':multiplier}


def validate_mask_state(state,allow_empty=False):
    keys={'alignment','observations','cosine','probability','accepted','multiplier'}
    if allow_empty and state=={}:return
    if type(state) is not dict or set(state)!=keys or not _number(state['alignment']) or state['alignment']>1 or type(state['observations']) is not int or state['observations']<1 or not _number(state['cosine'],-1) or state['cosine']>1 or state['probability']!=.5 or type(state['accepted']) is not bool or not _number(state['multiplier']):raise ValueError('Invalid complete mask state')


def validate_config(config):
    from ..config import canonical_bytes
    if type(config) is not dict or set(config)!=set(FIXTURE_CONFIG):raise ValueError('Complete noise configuration required')
    c=json.loads(canonical_bytes(config));construct_backbone(c['backbone'])
    if c['mode'] not in ('damping','simple','magma') or not _number(c['coefficient']):raise ValueError('Unknown noise mode/coefficient')
    if c['mode']!='damping' and c['coefficient']!=0:raise ValueError('Masking and damping are separate controls')
    validate_mask({k:('magma' if k=='mode' and c[k]=='damping' else c[k]) for k in MASK_CONFIG})
    if c['scale_recipe'] not in ('row_fixed_max','tensor_fixed_max'):raise ValueError('Noise requires explicit fixed scale grouping')
    return c


class NoiseAdapter(MethodAdapter):
    def __init__(self,config):
        self.config=validate_config(config);self.rule=construct_backbone(self.config['backbone'])

    def bind(self,parameters,groups):
        if tuple(g.name for g in groups) not in (('internal_projection',),('internal_projection','auxiliary_no_decay')) or groups[0].weight_decay is not None or (len(groups)>1 and groups[1].weight_decay!=0.):raise ValueError('Explicit projection and no-decay auxiliary groups required')
        MethodAdapter.__init__(self,parameters,groups,self.rule)
        self.mask_states={};self.mask_rng_state=torch.Generator().manual_seed(606).get_state()
        self.snapshot_schema['optimizer_states']=f'named.original_W.{self.rule.state_schema}+auxiliary.adamw.v1'
        self.snapshot_schema['adapter_mutable_state']='noise.matrix_alignment_rng.v1'
        return self

    def rule_for_parameter(self,name):
        if name in self.groups[0].parameter_names or type(self.rule) is AdamWUpdateRule:return self.rule
        return self.rule.auxiliary_rule

    def schedule_for_group(self,group,schedule):
        if group.name=='auxiliary_no_decay' and type(self.rule) is not AdamWUpdateRule:return SchedulePoint(self.rule.configuration['auxiliary_lr'],0.,schedule.progress)
        return super().schedule_for_group(group,schedule)

    @property
    def metadata(self):
        return {'ordered_parameter_groups':[{'name':g.name,'parameter_names':list(g.parameter_names),'weight_decay':g.weight_decay} for g in self.groups],
                'update_rule_settings':self.config,'schedule_fields':['lr','weight_decay','progress'],'extra_passes':[],
                'state_schema':self.snapshot_schema,'supported_backbones':['AdamW','Muon','Muon+'],'unsupported_backbones':['SOAP','Madam'],
                'scale_contract':{'quantizer_recipe':self.config['scale_recipe'],'physical':'y=s*C(w/s)','range':[-6,6],
                                  'grouping':'tensor' if self.config['scale_recipe'].startswith('tensor') else 'row','initialization':'max_div_cmax_zero_one',
                                  'information':'training_targets_only','export_precision':'float32'},
                'decay':'separate ordinary-LR decay; mask only gradient displacement'}

    def prepare_step(self,state,batch):
        super().prepare_step(state,batch)
        if state.model._recipe!={'name':self.config['scale_recipe'],'rotation':'identity','backward':'ste'}:raise ValueError('Noise requires declared identity/STE/fixed-scale recipe')

    def finish_step(self,state,schedule):
        import time
        start=time.perf_counter();proposals=self.collect_proposals(schedule)
        if self.config['mode']=='damping':
            return self.commit_proposals(proposals)
        rng=torch.Generator();rng.set_state(self.mask_rng_state.cpu())
        candidates={};next_masks={}
        for name,p in proposals.items():
            if name in self.groups[0].parameter_names:
                moment=p.new_optimizer_state.get('exp_avg',p.new_optimizer_state.get('momentum'))
                if moment is None:raise ValueError('Mask requires current original-W first moment')
                displacement,s=mask_displacement(p.gradient_displacement,moment,self.parameters[name].grad,self.mask_states.get(name,{}),float(torch.rand((),generator=rng)),{k:self.config[k] for k in MASK_CONFIG})
                candidates[name]=p.reference_after_decay+displacement;next_masks[name]=s
            else:
                candidate=self.parameters[name].detach().clone();self.rule_for_parameter(name).commit(candidate,p);candidates[name]=candidate
            if not bool(torch.isfinite(candidates[name]).all()):raise ValueError('Nonfinite masked candidate')
        if set(next_masks)!=set(self.groups[0].parameter_names):raise ValueError('Missing matrix gradient')
        with torch.no_grad():
            for name,w in candidates.items():self.parameters[name].copy_(w)
        for name,p in proposals.items():self.optimizer_states[name]=p.new_optimizer_state
        self.mask_states=next_masks;self.mask_rng_state=rng.get_state();self.proposals={}
        state._work['noise_work']={'matrix_mask_draws':len(next_masks),'dense_moment_updates':len(proposals),'mask_seconds':time.perf_counter()-start,
                                  'extra_loss_passes':0,'extra_backward_passes':0}
        return {'optimizer_updates':len(proposals),'candidate_validation_updates':len(proposals),'noise':next_masks}

    def backward_microbatch(self,state,batch):
        if self.config['mode']!='damping':return super().backward_microbatch(state,batch)
        loss=state.summed_nll(batch);nll=float(loss.detach())
        # One complete penalty per logical update, independent of packing and
        # shared H06 division by target count. No extra data pass is launched.
        penalty=sum(damping_penalty(self.parameters[n],state.model._contexts[n].storage_values,self.config['coefficient']) for n in self.groups[0].parameter_names)
        (loss+penalty*batch.target_count).backward()
        work=state._work.setdefault('damping_work',{'penalty_evaluations':0,'matrix_elements':0,'extra_loss_passes':0,'extra_backward_passes':0})
        work['penalty_evaluations']+=1;work['matrix_elements']+=sum(self.parameters[n].numel() for n in self.groups[0].parameter_names)
        return {'summed_nll':nll,'forward_passes':1,'backward_passes':1,'extra_forward_passes':0,'extra_backward_passes':0}


def trusted_constructor(parameters,groups,rule):
    return NoiseAdapter({**FIXTURE_CONFIG,'backbone':backbone_config(rule)}).bind(parameters,groups)


def validate_owned_state(method,groups,step,rule,recipe):
    c=validate_config(method['config'])
    if c['backbone']!=backbone_config(rule) or recipe!={'name':c['scale_recipe'],'rotation':'identity','backward':'ste'}:raise ValueError('Noise rule/recipe mismatch')
    if set(method['mask_states'])!=(set(groups[0].parameter_names) if step and c['mode']!='damping' else set()):raise ValueError('Incomplete mask inventory')
    for s in method['mask_states'].values():
        validate_mask_state(s)
        if s['observations']!=step or s['multiplier']!=((2. if c['mode']=='simple' else s['alignment']) if s['accepted'] else 0.):raise ValueError('Mask count/multiplier mismatch')
    torch.Generator().set_state(method['mask_rng_state'].cpu())


def check_scale_noise(device,output):
    """Prospective bounded engineering gate, analytic targets and safe replay."""
    import hashlib,os
    from pathlib import Path
    from ..config import canonical_bytes
    from ..records import ArtifactRef,sha256_file
    from ..optim.base import default_parameter_groups
    from ..models.transformer import ByteTransformer,ModelConfig
    from ..data.articles import Article
    from ..data.blocks import tile_article
    from ..data.batches import LogicalBatch
    from ..train import TrainingState,train_step
    from ..state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    from ..evaluate import score_blocks,evaluation_contexts
    from .learned_scale import LearnedScaleAdapter,learned_scale_view,VIEW_CONTRACT,FIXTURE_CONFIG as SCALE_CONFIG
    from .registry import scale_noise_record
    root=Path(__file__).resolve().parents[3]
    report={'suite':'scale-noise','device':device,'status':'pass','blockers':[],
        'scope':'analytic LSQ and FP4 surrogate targets; practical matrix masks; 1x8x2 synthetic 8-target two-step learned row/tensor and damping/simple/Magma same-device complete-state replay. No author software execution, universal device/model/horizon/packing parity, pilot relevance or science selection',
        'source_hashes':{str(p.relative_to(root)):sha256_file(p) for p in sorted(root.glob('src/ralup/**/*.py'))},
        'scientific_gate':None,'CPU_MPS_bitwise_parity_claimed':False,
        'predeclared_bounds':{'analytic_gradient_atol':2e-6,'analytic_gradient_rtol':2e-5,'same_device_resume':'exact complete scientific state'},
        'variants':[]}
    if device not in ('cpu','mps'):raise ValueError('Unknown scale/noise device')
    if device=='mps' and (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or not torch.backends.mps.is_available()):
        report.update(status='blocked',blockers=['Actual MPS unavailable or fallback not zero before startup'])
    else:
        try:
            analytics=[]
            for format,low,high in (('uniform',-8,7),('fp4',-6,6)):
                w=torch.tensor([[-20.,-1.8,.3,1.8,20.]],device=device,requires_grad=True)
                log=torch.tensor([[math.log(2.)]],device=device,requires_grad=True)
                c={**VIEW_CONTRACT,'format':format,'minimum':low,'maximum':high,'grouping':'tensor'}
                upstream=torch.tensor([[1.,2.,3.,4.,5.]],device=device)
                y=learned_scale_view(w,log,c);(upstream*y).sum().backward()
                x=w.detach()/2;codes=y.detach()/2
                local=torch.where(x<=low,float(low),torch.where(x>=high,float(high),codes-x))
                target=(local*upstream).sum().reshape(1,1)*2/math.sqrt(5*high)
                torch.testing.assert_close(log.grad,target,atol=2e-6,rtol=2e-5)
                analytics.append({'format':format,'log_scale_gradient':log.grad.cpu().tolist(),'analytic_target':target.cpu().tolist(),'normalizer':1/math.sqrt(5*high),'hard_finite_difference_target':False})
            report['analytic_surrogates']=analytics
            d=torch.tensor([[-.1,.2]],device=device);g=torch.tensor([[1.,-2.]],device=device)
            decisions={}
            for mode in ('simple','magma'):
                accepted,s=mask_displacement(d,g,g,{},.49,{**MASK_CONFIG,'mode':mode})
                rejected,r=mask_displacement(d,g,g,{},.51,{**MASK_CONFIG,'mode':mode})
                assert torch.count_nonzero(rejected)==0 and s['probability']==r['probability']==.5
                target=d*(2 if mode=='simple' else .9*.5+.1/(1+math.exp(-.5)))
                torch.testing.assert_close(accepted,target,atol=2e-6,rtol=2e-5)
                decisions[mode]={'accepted':s,'rejected':r}
            report['mask_decisions']=decisions
            blocks=[b for raw in (b'abcd',b'efgh') for b in tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))]
            batch=LogicalBatch(blocks,8)
            for name,grouping in (('learned_fp4_scale','row'),('learned_fp4_scale','tensor'),('damping_fp4','row'),('simple_matrix_mask','row'),('magma_matrix_mask','row')):
                with torch.random.fork_rng():
                    torch.manual_seed(606);model=ByteTransformer(ModelConfig(1,8,2)).to(device)
                if name=='learned_fp4_scale':
                    config={**SCALE_CONFIG,'view':{**VIEW_CONTRACT,'grouping':grouping}}
                    method=LearnedScaleAdapter(config).bind(dict(model.named_parameters()),default_parameter_groups(model));recipe=method.recipe
                else:
                    config={**FIXTURE_CONFIG,'mode':{'damping_fp4':'damping','simple_matrix_mask':'simple','magma_matrix_mask':'magma'}[name],
                            'coefficient':.1 if name=='damping_fp4' else 0.}
                    method=NoiseAdapter(config).bind(dict(model.named_parameters()),default_parameter_groups(model));recipe=config['scale_recipe']
                model.configure_quantization({'name':recipe,'rotation':'identity','backward':'ste'})
                state=TrainingState(model,method,microbatch=1,schedule={'step':0,'point':SchedulePoint(.001,.1,0.)})
                first=train_step(state,batch);before=exact_cpu_state_hash(state)
                score_blocks(model,blocks,evaluation_context={'partition':'fixture'})
                assert exact_cpu_state_hash(state)==before
                if name=='learned_fp4_scale':
                    contexts=evaluation_contexts(model,{'partition':'fixture'})
                    for n,s in method.export_scales()['scales'].items():assert torch.equal(s,contexts[n].scales)
                resumed=restore_snapshot(capture_snapshot(state),device)
                second=train_step(state,batch);train_step(resumed,batch)
                assert exact_cpu_state_hash(state)==exact_cpu_state_hash(resumed)
                assert all(v['step'].item()==2 for v in method.optimizer_states.values())
                def tensor_bytes(value):
                    if isinstance(value,Tensor):return value.numel()*value.element_size()
                    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
                    return 0
                record=scale_noise_record(name,method.groups,config,parity_status='pass',operations_status='pass')
                owned={k:v for k,v in vars(method).items() if k not in ('parameters','groups','rule')}
                report['variants'].append({'name':name,'grouping':grouping,'adaptation_label':record.adaptation_label,
                    'same_device_resume_exact':True,'evaluation_read_only':True,'original_W_moments':True,'state_tensor_bytes_including_explicit_gradient_owners':tensor_bytes(owned),
                    'first_work':first.work,'second_work':second.work,'export_scale_contract':method.scale_contract if name=='learned_fp4_scale' else {'initialization':'max_div_6_zero_one','grouping':grouping,'information':'training_targets_only','export_precision':'float32'},
                    'source_copy_status':'independent local equations; no author software copied','scientific_eligible':False})
                if name=='learned_fp4_scale':
                    report['variants'][-1]['physical_cache']={'backend':method.scale_backend,'bytes':tensor_bytes(method.physical_scales),
                        'validation_telemetry':dict(method.scale_validation_telemetry),'validation':'CPU exact cache/runtime/quantizer ownership; exact producing-backend exp check at capture and restore; cross-backend restore unsupported'}
        except (AssertionError,ValueError,RuntimeError) as error:report.update(status='fail',blockers=[f'{type(error).__name__}: {error}'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(canonical_bytes(report))
    return ArtifactRef(output,sha256_file(output))
