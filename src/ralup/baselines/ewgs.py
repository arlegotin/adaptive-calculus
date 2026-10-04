"""Independent S24 EWGS equations; no GPL source copied.

Uniform primitive is [0,1]. FP4 uses storage-coordinate dimensionless residuals.
Adaptive delta uses one whole projection tensor (not each scale row), one
logical-mean loss on the same training batch, population std plus 1e-12, and
fixed-count independent Rademacher probes. These are named release departures.
"""
from dataclasses import dataclass,replace
import math,time
from types import MappingProxyType
from typing import Callable
import torch
from torch import Tensor
from ..optim.base import MethodAdapter
from ..optim.adamw import AdamWUpdateRule

FIXTURE_CONFIG={'mode':'fixed','delta':.25,'normalization':'storage_U_div_6_cached_scale',
    'estimator_scope':'whole_projection_tensor','loss_reduction':'logical_mean',
    'std_correction':0,'epsilon':1e-12,'probes':2,'cadence':2,'probe_seed':303,
    'clipping':'representable_range_inclusive','pilot_relevance':False,'operation_support':'unverified'}
ADAPTER_FIELDS=('optimizer_states','proposals','snapshot_schema','config','deltas','probe_rng_state','last_refresh_step','refresh_telemetry')


def validate_config(config):
    if type(config) is not dict or set(config)!=set(FIXTURE_CONFIG):
        raise ValueError('EWGS requires complete explicit configuration')
    c=dict(config)
    for key in ('normalization','estimator_scope','loss_reduction','std_correction','epsilon','clipping'):
        if c[key]!=FIXTURE_CONFIG[key] or type(c[key]) is not type(FIXTURE_CONFIG[key]):raise ValueError(f'Unsupported EWGS convention: {key}')
    if c['mode'] not in ('fixed','adaptive'):raise ValueError('Unknown EWGS mode')
    if type(c['delta']) not in (int,float) or not math.isfinite(c['delta']) or c['delta']<0:raise ValueError('Delta must be finite and nonnegative')
    for key in ('probes','cadence','probe_seed'):
        if type(c[key]) is not int or c[key]<(0 if key=='probe_seed' else 1):raise ValueError(f'Invalid {key}')
    if c['probe_seed']>2**64-1:raise ValueError('Probe seed outside uint64')
    if type(c['pilot_relevance']) is not bool or c['operation_support'] not in ('unverified','pass','blocked'):raise ValueError('Explicit relevance and operation support required')
    if c['mode']=='adaptive' and (not c['pilot_relevance'] or c['operation_support']!='pass'):
        raise ValueError('Adaptive EWGS requires pilot relevance and operation support')
    return c


def ewgs_backward(upstream: Tensor, normalized_shadow: Tensor, normalized_code: Tensor, delta: float, clipping_mask: Tensor) -> Tensor:
    if type(delta) not in (int,float) or not math.isfinite(delta) or delta<0:raise ValueError('Delta must be finite and nonnegative')
    if any(t.shape!=upstream.shape or t.device!=upstream.device for t in (normalized_shadow,normalized_code,clipping_mask)):
        raise ValueError('EWGS operands must share geometry/device')
    return upstream*(1+delta*upstream.sign()*(normalized_shadow-normalized_code))*clipping_mask


def estimate_delta(loss_fn: Callable, quantized_leaf: Tensor, probes: Tensor, *, work_recorder=None) -> dict:
    """Detach/own normalized leaf; H and g share these exact coordinates.

    loss_fn is responsible for a declared physical 6*s map and mean reduction.
    The optional launch recorder charges real attempts before each derivative.
    Constant graph-free gradients infer a zero Hessian without HVP launches;
    requested probes remain distinct from actual backward counts.
    """
    if probes.ndim!=quantized_leaf.ndim+1 or probes.shape[1:]!=quantized_leaf.shape or not probes.shape[0] or probes.device!=quantized_leaf.device:
        raise ValueError('Probes must be nonempty [P,*leaf.shape] on leaf device')
    if not bool(((probes==1)|(probes==-1)).all()):raise ValueError('Rademacher probes must be +/-1')
    leaf=quantized_leaf.detach().clone().requires_grad_(True)
    start=time.perf_counter()
    if work_recorder:work_recorder('extra_loss')
    loss=loss_fn(leaf)
    if loss.ndim!=0 or not bool(torch.isfinite(loss)):raise ValueError('Estimator needs finite scalar loss')
    if work_recorder:work_recorder('extra_loss_backward')
    g=torch.autograd.grad(loss,leaf,create_graph=True)[0]
    traces=[];hvp_backward_passes=0
    zero_hessian_inferred=not g.requires_grad
    for probe in probes:
        if zero_hessian_inferred:
            hv=torch.zeros_like(leaf)
        else:
            if work_recorder:work_recorder('hvp')
            hvp_backward_passes+=1
            hv=torch.autograd.grad(g,leaf,probe,retain_graph=True)[0]
        traces.append((probe*hv).sum().detach())
    trace=torch.stack(traces).mean();std=g.detach().std(correction=0)
    delta=(trace/leaf.numel()/(3*std+1e-12)).clamp_min(0)
    if not bool(torch.isfinite(delta)):raise ValueError('Nonfinite adaptive EWGS delta')
    if leaf.device.type=='mps':torch.mps.synchronize()
    return {'delta':float(delta),'estimated_trace':float(trace),'population_std':float(std),'elements':leaf.numel(),
        'probes':probes.shape[0],'coordinates':'detached_normalized_quantized_leaf',
        'extra_forward_passes':1,'extra_backward_passes':1+hvp_backward_passes,
        'hvp_backward_passes':hvp_backward_passes,'zero_hessian_inferred':zero_hessian_inferred,
        'leaf_bytes':leaf.numel()*leaf.element_size(),'probe_bytes':probes.numel()*probes.element_size(),
        'elapsed_seconds':time.perf_counter()-start}


def with_delta(context,delta):
    return replace(context,metadata=MappingProxyType({**context.metadata,'ewgs_delta':float(delta)}))


@dataclass(frozen=True)
class NormalizedLeafView:
    """Ephemeral functional loss override; never stored in model or snapshot."""
    storage_values: Tensor
    rotation: str


class EwgsAdapter(MethodAdapter):
    extra_passes=('conditional_logical_mean_loss','normalized_leaf_gradient','rademacher_hvp')
    def __init__(self,config: dict):
        self.config=validate_config(config)
        self.deltas={}
        self.probe_rng_state=torch.Generator(device='cpu').manual_seed(self.config['probe_seed']).get_state()
        self.last_refresh_step=-1
        self.refresh_telemetry={}

    def bind(self,parameters,groups,rule=None):
        rule=AdamWUpdateRule() if rule is None else rule
        if type(rule) is not AdamWUpdateRule:raise ValueError('EWGS owns original-W AdamW only')
        MethodAdapter.__init__(self,parameters,groups,rule)
        self.snapshot_schema['adapter_mutable_state']='ewgs.normalized_leaf_probes_cadence.v2'
        return self

    @property
    def metadata(self):
        return {**super().metadata,'ewgs':dict(self.config),'probe_stream':'adapter_owned_cpu_generator; independent of data RNG'}

    def prepare_step(self,state,batch):
        super().prepare_step(state,batch)
        if state.model._recipe.get('backward')!='ewgs':raise ValueError('EWGS requires explicit ewgs backward recipe')
        contexts={name:with_delta(c,self.deltas.get(name,self.config['delta'])) for name,c in state.model.quantization_contexts.items()}
        step=state.schedule['step']
        if self.config['mode']=='adaptive' and step%self.config['cadence']==0:
            from ..evaluate import _target_losses
            generator=torch.Generator(device='cpu');generator.set_state(self.probe_rng_state.cpu())
            for name,c in contexts.items():
                start=time.perf_counter()
                q=c.storage_values/c.scales/6
                probes=(torch.randint(0,2,(self.config['probes'],*q.shape),generator=generator)*2-1).to(q)
                self.probe_rng_state=generator.get_state()
                def loss_fn(leaf):
                    overrides={**contexts,name:NormalizedLeafView(leaf*6*c.scales.detach(),c.rotation)}
                    return _target_losses(state.model,batch,contexts=overrides).sum()/batch.target_count
                counter=0
                def record(kind):
                    nonlocal counter
                    counter+=1
                    state.record_work(additional_loss=batch.target_count if kind=='extra_loss' else 0,
                        loss_targets=batch.target_count if kind=='extra_loss' else 0,
                        backward=batch.target_count if kind!='extra_loss' else 0,
                        forward_context=sum(len(b.inputs) for b in batch.block_slices) if kind=='extra_loss' else 0,
                        extra_pass_event={'event_id':f'ewgs:{step}:{name}:{counter}','execution_id':f'ewgs:{step}:{name}:loss' if kind!='hvp' else f'ewgs:{step}:{name}:hvp:{counter-2}','kind':kind,'target_bytes':batch.target_count})
                try:
                    estimate=estimate_delta(loss_fn,q,probes,work_recorder=record)
                    self.deltas[name]=estimate['delta'];self.refresh_telemetry[name]=estimate
                finally:
                    if q.device.type=='mps':torch.mps.synchronize()
                    state.record_work(ewgs_seconds=time.perf_counter()-start)
            self.last_refresh_step=step
        state.model._contexts={name:with_delta(c,self.deltas.get(name,self.config['delta'])) for name,c in contexts.items()}


def trusted_constructor(parameters,groups,rule):
    return EwgsAdapter(dict(FIXTURE_CONFIG)).bind(parameters,groups,rule)


def validate_owned_state(method,groups,step,contexts):
    """Finite trusted config/RNG/delta/cadence owners validated before transfer."""
    validate_config(method['config'])
    names=set(groups[0].parameter_names)
    if set(method['deltas'])-names or set(method['refresh_telemetry'])!=set(method['deltas']):raise ValueError('Unknown EWGS projection state')
    rng=method['probe_rng_state']
    if not isinstance(rng,Tensor) or rng.dtype!=torch.uint8 or rng.ndim!=1:raise ValueError('Invalid EWGS RNG owner')
    torch.Generator(device='cpu').set_state(rng.cpu())
    last=method['last_refresh_step'];config=method['config']
    expected=-1 if config['mode']=='fixed' or not step else ((step-1)//config['cadence'])*config['cadence']
    if type(last) is not int or last!=expected:raise ValueError('EWGS refresh cadence mismatch')
    if last>=0 and set(method['deltas'])!=names:raise ValueError('Missing adaptive EWGS projection')
    if config['mode']=='fixed' and method['deltas']:raise ValueError('Fixed EWGS has adaptive state')
    for name,delta in method['deltas'].items():
        if type(delta) not in (float,int) or not math.isfinite(delta) or delta<0:raise ValueError('Invalid EWGS delta')
        estimate=method['refresh_telemetry'][name]
        required={'delta','estimated_trace','population_std','elements','probes','coordinates','extra_forward_passes','extra_backward_passes','hvp_backward_passes','zero_hessian_inferred','leaf_bytes','probe_bytes','elapsed_seconds'}
        if type(estimate) is not dict or set(estimate)!=required or estimate['delta']!=delta or estimate['probes']!=config['probes'] or estimate['coordinates']!='detached_normalized_quantized_leaf':raise ValueError('Invalid EWGS estimator telemetry')
        inferred=estimate['zero_hessian_inferred']
        if type(inferred) is not bool:raise ValueError('Invalid EWGS zero-Hessian inference flag')
        hvp_passes=0 if inferred else config['probes']
        if inferred and (estimate['estimated_trace']!=0 or delta!=0):raise ValueError('Inferred zero Hessian requires zero trace/delta')
        elements=contexts[name].storage_values.numel()
        for key,expected_value in {'elements':elements,'probes':config['probes'],'extra_forward_passes':1,'extra_backward_passes':1+hvp_passes,'hvp_backward_passes':hvp_passes,'leaf_bytes':elements*4,'probe_bytes':elements*4*config['probes']}.items():
            if type(estimate[key]) is not int or estimate[key]!=expected_value:raise ValueError('EWGS estimator geometry/pass mismatch')
        for key in ('estimated_trace','population_std','elapsed_seconds'):
            if type(estimate[key]) not in (int,float) or not math.isfinite(estimate[key]) or (key!='estimated_trace' and estimate[key]<0):raise ValueError('Invalid EWGS measurement')
        expected_delta=max(0.,estimate['estimated_trace']/elements/(3*estimate['population_std']+1e-12))
        if not math.isclose(delta,expected_delta,rel_tol=2e-6,abs_tol=1e-8):raise ValueError('EWGS delta/estimator mismatch')
    for name,c in contexts.items():
        if c.backward!='ewgs' or c.metadata.get('ewgs_delta')!=method['deltas'].get(name,config['delta']):raise ValueError('EWGS context delta owner mismatch')


def check_ewgs(device,output):
    """Bounded independent equations and actual-device higher-order gate."""
    import itertools,json,os
    from pathlib import Path
    from dataclasses import asdict
    from ..config import canonical_bytes
    from ..records import ArtifactRef,sha256_file
    from .registry import ewgs_record
    from ..optim.base import ParameterGroup
    root=Path(__file__).resolve().parents[3]
    report={'suite':'ewgs','device':device,'status':'pass','blockers':[],
        'scope':'independent primitive and normalized-leaf quadratic; actual neural gate separate; no pilot relevance or science eligibility',
        'conventions':dict(FIXTURE_CONFIG),'cpu_analytic_evidence_is_actual_mps_support':False,
        'source_hashes':{str(p.relative_to(root)):sha256_file(p) for p in sorted(root.glob('src/ralup/**/*.py'))},
        'source_copy_status':'blocked_GPL_inspection_only','scientific_gate':None,'variants':[]}
    if device not in ('cpu','mps'):raise ValueError('Unsupported EWGS device')
    if device=='mps' and (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or not torch.backends.mps.is_available()):
        report.update(status='blocked',blockers=['Actual MPS unavailable or fallback not zero before startup'])
    else:
        for name in ('ewgs_uniform_fixture','ewgs_fp4'):
            variant={'name':name,'operation_status':'pass','parity_status':'pass'}
            try:
                x=torch.tensor([.1,.2,.6,.9],device=device);q=torch.round(x*3)/3;g=torch.tensor([2.,-3.,0.,-4.],device=device);mask=torch.tensor([1.,1.,1.,0.],device=device)
                torch.testing.assert_close(ewgs_backward(g,x,q,.3,mask),g*(1+.3*g.sign()*(x-q))*mask)
                variant['original_rule_parity']='pass'
                if name=='ewgs_fp4':
                    from ..quantization.context import prepare_context,quantized_view
                    w=torch.tensor([[4.,-.2,.3,-4.]],device=device,requires_grad=True)
                    context=with_delta(prepare_context(w,{'name':'row_fixed_max','backward':'ewgs','previous_scales':torch.tensor([[.5]],device=device)},0,None),.5)
                    physical=torch.tensor([[2.,-3.,4.,-5.]],device=device)
                    quantized_view(w,context).backward(physical)
                    z=(w.detach()/context.scales/6).clamp(-1,1);normalized=context.storage_values/context.scales/6
                    torch.testing.assert_close(w.grad,physical*(1+.5*physical.sign()*(z-normalized))*context.gradient_mask)
                    variant['fp4_dimensionless_residual_clipping']='pass'
                    h=torch.tensor([[3.,1.],[1.,5.]],device=device)
                    probes=torch.tensor(list(itertools.product([-1.,1.],repeat=2)),device=device)
                    launches=[]
                    estimate=estimate_delta(lambda leaf:.5*leaf@h@leaf,torch.tensor([1/3,2/3],device=device),probes,work_recorder=launches.append)
                    if abs(estimate['estimated_trace']-8)>1e-6:raise AssertionError('Analytic Hessian trace disagreement')
                    variant.update(analytic_hessian_agreement='pass',estimate=estimate,
                        adaptive_pass_cost={'events':launches,'extra_loss_passes':1,'loss_backward_passes':1,'gradient_only_passes':0,'hvp_passes':4,'seconds':estimate['elapsed_seconds'],
                            'scope':'two-dimensional quadratic; first gradient shares extra-loss exposure; target-byte neural passes measured separately'},
                        adaptive_neural_execution='conditional on measured pilot relevance and method/model operation support; M07 owns relevance')
            except AssertionError as error:
                variant.update(parity_status='fail',error=str(error));report['status']='fail';report['blockers'].append(str(error))
            except (RuntimeError,ValueError) as error:
                variant.update(operation_status='blocked',parity_status='blocked',error=str(error))
                if report['status']!='fail':report['status']='blocked'
                report['blockers'].append(str(error))
            variant['method']=json.loads(json.dumps(asdict(ewgs_record(name,(ParameterGroup('internal_projection',('fixture',)),),FIXTURE_CONFIG,
                parity_status=variant['parity_status'],operations_status=variant['operation_status']))))
            report['variants'].append(variant)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(canonical_bytes(report)+b'\n')
    return ArtifactRef(output,sha256_file(output))
