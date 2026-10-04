"""Explicit-uniform FP4 stochastic rounding and separate state controls."""
import torch
from torch import Tensor
from ..quantization.formats import E2M1_VALUES,_fp32_pair,_preserved_ratio,_preserved_reconstruction
from .qar import _QuantizedAdapter,_uniforms,FIXTURE_CONFIG,backbone_config


@torch.no_grad()
def _bracket(values,scales):
    xbits,sbits=_fp32_pair(values,scales)
    if values.numel()==0:raise ValueError('Stochastic rounding requires nonempty values')
    exceptional=bool((((xbits&0x7fffffff)>0)&((xbits&0x7fffffff)<0x800000)).any()) or bool((sbits<0x800000).any())
    x=(_preserved_ratio(xbits,sbits) if exceptional else values/scales).clamp(-6,6)
    lower=torch.zeros_like(x,dtype=torch.int64)
    for i,level in enumerate(E2M1_VALUES):lower=torch.where(x>=level,i,lower)
    upper=(lower+1).clamp(max=14);levels=torch.tensor(E2M1_VALUES,dtype=torch.float32,device=values.device)
    lo,hi=levels[lower],levels[upper];gap=hi-lo
    probability=torch.where(gap>0,(x-lo)/torch.where(gap>0,gap,1.),0.).clamp(0,1)
    return lower,upper,probability


@torch.no_grad()
def upper_probability(values:Tensor,scales:Tensor)->Tensor:
    """Secondary expectation probability; exact levels/endpoints are deterministic."""
    return _bracket(values,scales)[2]


@torch.no_grad()
def stochastic_round(values:Tensor,scales:Tensor,uniforms:Tensor)->tuple[Tensor,Tensor]:
    """P(upper)=(x-l)/(u-l), saturation; caller supplies every uniform draw.

    Uniforms are in [0,1); zero probability never rounds upward, including a
    uniform exactly zero. Returns reconstructed FP32 values and uint8 ordinals.
    """
    _uniforms(uniforms,values);lower,upper,p=_bracket(values,scales)
    codes=torch.where(uniforms<p,upper,lower)
    return _preserved_reconstruction(codes,scales.contiguous().view(torch.int32)),codes.to(torch.uint8)


class RoundingAdapter(_QuantizedAdapter):
    """mode=shadow_forward retains W; mode=endpoint retains codes/scales only."""
    rounding=True


def trusted_constructor(parameters,groups,rule):
    return RoundingAdapter({**FIXTURE_CONFIG,'backbone':backbone_config(rule),'mode':'endpoint'}).bind(parameters,groups)


def check_routing_rounding(device,output):
    """Owned small engineering fixtures; no uniform-to-FP4 theory claim."""
    import hashlib,json,os
    from pathlib import Path
    from ..config import canonical_bytes
    from ..records import ArtifactRef,sha256_file
    from ..optim.base import default_parameter_groups,SchedulePoint
    from ..models.transformer import ByteTransformer,ModelConfig
    from ..data.articles import Article
    from ..data.blocks import tile_article
    from ..data.batches import LogicalBatch
    from ..train import TrainingState,train_step
    from ..state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    from ..evaluate import score_blocks
    from .qar import QarAdjacentAdapter,uniform_qar_step,SOURCE_CLARIFICATION
    from .registry import routing_record
    root=Path(__file__).resolve().parents[3]
    report={'suite':'routing-rounding','device':device,'status':'pass','blockers':[],
        'scope':'source-masked uniform equations; FP4 analytic/enumerated expectation; 1x8x2 synthetic 16-target two-step cached contexts and same-device complete-state replay. No author release execution, universal model/horizon/packing parity, pilot/scientific selection or restricted-software copying',
        'source_clarification':SOURCE_CLARIFICATION,'fp4_uniform_theory_transfer':False,'scientific_gate':None,
        'source_hashes':{str(p.relative_to(root)):sha256_file(p) for p in sorted(root.glob('src/ralup/**/*.py'))},'variants':[]}
    if device not in ('cpu','mps'):raise ValueError('Unknown routing/rounding device')
    if device=='mps' and (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or not torch.backends.mps.is_available()):
        report.update(status='blocked',blockers=['Actual MPS unavailable or fallback not zero before startup'])
    else:
        try:
            codes,p=uniform_qar_step(torch.tensor([[0,2]],dtype=torch.int64,device=device),torch.tensor([[100.,1.]],device=device),spacing=1.,minimum=0,maximum=4,lr=.1,gamma=1.,group_axes=(-1,),uniforms=torch.zeros((1,2),device=device))
            torch.testing.assert_close(p,torch.tensor([[0.,.2]],device=device),atol=1e-7,rtol=1e-6)
            assert codes.tolist()==[[0,1]]
            report['uniform_fixture']={'name':'qar_uniform_fixture','state':'integer codes+shared uniform spacing+original-W moments; no shadow residual','status':'pass','probabilities':p.cpu().tolist(),'codes':codes.cpu().tolist(),'artifact_kind':'original_fixture'}
            x=torch.full((100,1),1.04,device=device);uniforms=(torch.arange(100,dtype=torch.float32,device=device)[:,None]+.5)/100
            values,codes=stochastic_round(x,torch.ones((1,1),device=device),uniforms)
            assert int((codes==10).sum())==8
            torch.testing.assert_close(values.mean(),torch.tensor(1.04,device=device),atol=2e-7,rtol=0)
            report['expectation']={'values':[1.,1.5],'upper_probability':.08,'analytical_expectation':1.04,'enumerated_uniforms':100,'upper_count':8,'enumerated_mean':float(values.mean()),'atol':2e-7,'scope':'bounded secondary expectation fixture; no stochastic primary inference'}
            blocks=[b for raw in (b'abcd',b'efgh',b'ijkl',b'mnop') for b in tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))]
            batch=LogicalBatch(blocks,16)
            for name in ('qar_fp4_adjacent','stochastic_shadow_forward','stochastic_endpoint'):
                with torch.random.fork_rng():
                    torch.manual_seed(505);model=ByteTransformer(ModelConfig(1,8,2)).to(device)
                config={**FIXTURE_CONFIG}
                if name.startswith('stochastic_'):config['mode']='shadow_forward' if name=='stochastic_shadow_forward' else 'endpoint'
                method=(QarAdjacentAdapter(config) if name=='qar_fp4_adjacent' else RoundingAdapter(config)).bind(dict(model.named_parameters()),default_parameter_groups(model))
                model.configure_quantization({'name':'row_fixed_max','rotation':'identity','backward':'ste'})
                state=TrainingState(model,method,microbatch=1,rng_streams={'quantizer':torch.Generator().manual_seed(777)},schedule={'step':0,'point':SchedulePoint(.001,.1,0.)})
                first=train_step(state,batch);before=exact_cpu_state_hash(state)
                score=score_blocks(model,batch.block_slices,evaluation_context={'partition':'fixture'})
                assert before==exact_cpu_state_hash(state)
                resumed=restore_snapshot(capture_snapshot(state),device);second=train_step(state,batch);train_step(resumed,batch)
                assert exact_cpu_state_hash(state)==exact_cpu_state_hash(resumed)
                record=routing_record(name,method.groups,config,parity_status='pass',operations_status='pass')
                report['variants'].append({'name':name,'adaptation_label':record.adaptation_label,'persistent_state':record.update_rule_settings['persistent_authority'],'original_W_moments':True,
                    'seed_replay':'pass','deterministic_evaluation_read_only':True,'primary_evaluation':'deterministic nearest-even','evaluation_bpb':score.bpb,
                    'first_work':first.work['routing_rounding_work'],'second_work':second.work['routing_rounding_work'],'second_telemetry':second.telemetry['routing_rounding'],
                    'scale_initialization':method.scale_initialization,'state_schema':method.snapshot_schema,'scope':'enumerated synthetic geometry/horizon only'})
        except (AssertionError,ValueError,RuntimeError) as error:
            report.update(status='fail',blockers=[f'{type(error).__name__}: {error}'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(canonical_bytes(report))
    return ArtifactRef(output,sha256_file(output))
