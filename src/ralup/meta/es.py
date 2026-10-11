"""S21 antithetic estimator and explicit resumable, decay-free outer Adam."""
import math
import torch


@torch.no_grad()
def estimate_es(phi, perturbations, plus, minus, sigma):
    if (not isinstance(sigma,(int,float)) or not math.isfinite(sigma) or sigma<=0
        or phi.ndim!=1 or perturbations.ndim!=2 or perturbations.shape[1:]!=phi.shape
        or perturbations.shape[0]==0 or plus.shape!=minus.shape
        or plus.shape!=(perturbations.shape[0],)
        or any(not x.is_floating_point() or not bool(torch.isfinite(x).all()) for x in (phi,perturbations,plus,minus))):
        raise ValueError('Finite compatible ES vectors and positive sigma required')
    return ((plus-minus)[:,None]*perturbations).sum(0)/(2*len(plus)*sigma)


class OuterAdam:
    """Outer state is finite data, never an unrolled autograd graph."""
    def __init__(self,phi,lr=.01,betas=(.9,.999),eps=1e-8):
        if phi.ndim!=1 or phi.dtype!=torch.float32 or not bool(torch.isfinite(phi).all()) or not math.isfinite(lr) or lr<=0 or len(betas)!=2 or any(not 0<=b<1 for b in betas) or not math.isfinite(eps) or eps<=0:
            raise ValueError('Invalid outer Adam configuration')
        self.phi=phi.detach().clone();self.m=torch.zeros_like(phi);self.v=torch.zeros_like(phi)
        self.lr=lr;self.betas=tuple(betas);self.eps=eps;self.step_count=0

    @torch.no_grad()
    def step(self,gradient):
        if gradient.shape!=self.phi.shape or not bool(torch.isfinite(gradient).all()):raise ValueError('Invalid outer gradient')
        b1,b2=self.betas;step=self.step_count+1
        m=self.m.clone().mul_(b1).add_(gradient,alpha=1-b1)
        v=self.v.clone().mul_(b2).addcmul_(gradient,gradient,value=1-b2)
        phi=self.phi.clone().addcdiv_(m/(1-b1**step),(v/(1-b2**step)).sqrt()+self.eps,value=-self.lr)
        if not all(bool(torch.isfinite(value).all()) for value in (m,v,phi)):
            raise ValueError('Outer Adam update must remain finite')
        self.m,self.v,self.phi,self.step_count=m,v,phi,step
        return self.phi.clone()

    def state_dict(self):
        return {'phi':self.phi.tolist(),'m':self.m.tolist(),'v':self.v.tolist(),'lr':self.lr,'betas':list(self.betas),'eps':self.eps,'step_count':self.step_count}

    @classmethod
    def from_state_dict(cls,value):
        if set(value)!={'phi','m','v','lr','betas','eps','step_count'}:raise ValueError('Unknown outer state')
        obj=cls(torch.tensor(value['phi'],dtype=torch.float32),value['lr'],value['betas'],value['eps'])
        obj.m=torch.tensor(value['m'],dtype=torch.float32);obj.v=torch.tensor(value['v'],dtype=torch.float32)
        if obj.m.shape!=obj.phi.shape or obj.v.shape!=obj.phi.shape or not bool(torch.isfinite(obj.m).all()&torch.isfinite(obj.v).all()) or bool((obj.v<0).any()) or type(value['step_count']) is not int or value['step_count']<0:raise ValueError('Invalid outer moments')
        obj.step_count=value['step_count'];return obj


def train_generation(task,family,outer,perturbations,horizon=16,sigma=.05):
    """One matched antithetic generation; inactive final states are disposable."""
    from dataclasses import asdict
    from .tasks import rollout
    from ..policy.controllers import ControllerState
    if task.provenance.get('final_snapshot_directory'):raise ValueError('Antithetic finals must use declared disposable retention')
    # Validate before spending any rollout work.
    estimate_es(outer.phi,perturbations,torch.zeros(len(perturbations)),torch.zeros(len(perturbations)),sigma)
    phi=outer.phi.clone();pairs=[];plus=[];minus=[]
    for epsilon in perturbations:
        a=rollout(task,ControllerState(family,phi+sigma*epsilon),horizon)
        b=rollout(task,ControllerState(family,phi-sigma*epsilon),horizon)
        if a.work['initial_state_hash']!=b.work['initial_state_hash'] or a.work['input_sha256']!=b.work['input_sha256'] or a.initial_query!=b.initial_query:
            raise ValueError('Antithetic replay/input mismatch')
        pairs.append({'plus':asdict(a),'minus':asdict(b)})
        plus.append(a.objective);minus.append(b.objective)
    gradient=estimate_es(phi,perturbations,torch.tensor(plus),torch.tensor(minus),sigma)
    outer.step(gradient)
    return {'family':family,'outer_step':outer.step_count,'phi_before':phi.tolist(),'phi_after':outer.phi.tolist(),
        'perturbations':perturbations.tolist(),'sigma':sigma,'horizon':horizon,'gradient':gradient.tolist(),'pairs':pairs,
        'optimizer_steps':2*len(pairs)*horizon,'query_targets':sum(p[side]['work']['query_targets'] for p in pairs for side in ('plus','minus'))}


def run_campaign(path):
    """Execute an immutable finite campaign, with complete outer/RNG recovery.

    A resumed interrupted campaign retains its scientific configuration. An
    explicitly frozen subsequent campaign may import an outer checkpoint while
    changing its pool under an authenticated refresh record. Neither route
    imports old-source model state or makes development outcomes policy inputs.
    """
    import hashlib
    import json
    import os
    import random
    import time
    from dataclasses import replace
    from pathlib import Path
    from ..config import canonical_bytes
    from ..records import ArtifactRef,sha256_file
    from ..search import source_identity
    from ..state import load_snapshot,restore_snapshot
    from ..data.batches import next_logical_batch
    from ..data.blocks import TargetBlock
    from ..policy.controllers import ControllerState
    from ..policy.checks import read_exploratory_backbone
    from ..train import _rng_state,_restore_rng
    from .tasks import MetaTask,sample_meta_task,authenticate_queries

    def authenticated(ref):
        artifact=ArtifactRef(**ref)
        if sha256_file(artifact.path)!=artifact.sha256:
            raise ValueError('Frozen campaign artifact changed')
        return json.loads(artifact.path.read_bytes())

    started=time.perf_counter()
    value=json.loads(Path(path).read_bytes())
    if value.get('schema')!='ralup.meta_campaign.v1':
        raise ValueError('meta-train unresolved: freeze current pool, normalizer, calibration and finite campaign first; B0 remains unresolved')
    source=source_identity()
    if value.get('source_identity')!=source:
        raise ValueError('Frozen current-source campaign required')
    for key in ('generations','horizon','pairs','seed','maximum_optimizer_steps','logical_batch_targets'):
        if type(value.get(key)) is not int or value[key]<=0:
            raise ValueError('Finite positive campaign budget required')
    updates=value['generations']*2*value['pairs']*value['horizon']
    seconds=value.get('maximum_seconds')
    if updates>value['maximum_optimizer_steps'] or type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:
        raise ValueError('Campaign exceeds frozen finite budget')
    deadline=started+seconds
    synthetic=value.get('synthetic') is True
    if synthetic:
        if value.get('purpose')!='synthetic-smoke' or value.get('device')!='cpu':
            raise ValueError('Synthetic campaign must be an explicit CPU smoke')
    else:
        if value.get('synthetic') is not False or value.get('purpose')!='exploratory-meta-training' or value['logical_batch_targets']!=8192:
            raise ValueError('Genuine exploratory protocol and8192-target logical batches required')
        if not value.get('backbone'):
            raise ValueError('Genuine exploratory protocol required')
        authenticated(value['backbone'])
        read_exploratory_backbone(Path(value['backbone']['path']))
        authenticated(value['calibration'])
        if any(entry.get('provenance',{}).get('synthetic') is not False for entry in value['pool']):
            raise ValueError('Genuine campaign cannot import synthetic tasks')
    controller=ControllerState(value['family'],torch.tensor(value['parameters'],dtype=torch.float32))
    outer=OuterAdam(controller.parameters,lr=value['outer_learning_rate'])
    # This also rejects invalid sigma before creating output or executing work.
    estimate_es(outer.phi,torch.zeros(value['pairs'],outer.phi.numel()),torch.zeros(value['pairs']),torch.zeros(value['pairs']),value['sigma'])
    rng=random.Random(value['seed'])
    generator=torch.Generator().manual_seed(value['seed'])
    identity=hashlib.sha256(canonical_bytes({k:v for k,v in value.items() if k not in ('output','resume')})).hexdigest()
    completed=0
    imported=value.get('resume') or value.get('initial_checkpoint')
    if imported:
        saved=authenticated(imported)
        if saved['source_identity']!=source or saved['family']!=value['family']:
            raise ValueError('Outer checkpoint source/family mismatch')
        if value.get('resume'):
            if saved['campaign_identity']!=identity:
                raise ValueError('Resume cannot change the frozen scientific campaign')
            completed=saved['completed_generations']
            if type(completed) is not int or not 0<=completed<value['generations']:
                raise ValueError('No unfinished campaign generations to resume')
        else:
            refresh=authenticated(value['pool_refresh'])
            if refresh['source_identity']!=source:
                raise ValueError('Refreshed pool must have current source')
        outer=OuterAdam.from_state_dict(saved['outer'])
        if outer.lr!=value['outer_learning_rate'] or outer.phi.numel()!=controller.parameters.numel():
            raise ValueError('Outer checkpoint hyperparameters/capacity changed')
        if not value.get('resume') and not torch.equal(outer.phi,controller.parameters):
            raise ValueError('Frozen refreshed-campaign parameters differ from outer checkpoint')
        sampling=saved['sampling_rng']
        rng.setstate((sampling[0],tuple(sampling[1]),sampling[2]))
        generator.set_state(torch.tensor(saved['perturbation_rng'],dtype=torch.uint8))
    schedule=value.get('task_schedule')
    if schedule is not None and (type(schedule) is not list or len(schedule)!=value['generations'] or any(type(i) is not int or not 0<=i<len(value['pool']) for i in schedule)):
        raise ValueError('Invalid prospectively frozen task schedule')
    # Authenticate every input before opening a scientific output directory.
    pool=[]
    for entry in value['pool']:
        if time.perf_counter()>deadline:
            raise RuntimeError('Frozen campaign deadline exhausted during input preparation')
        snapshot=load_snapshot(ArtifactRef(**entry['snapshot']))
        outside_rng=_rng_state()
        try:
            state=restore_snapshot(snapshot,value['device'])
            q=authenticated(entry['queries'])
            blocks=[TargetBlock(**{k:tuple(v) if isinstance(v,list) else v for k,v in b.items()}) for b in q['blocks']]
            if not synthetic:
                authenticate_queries(state.cursor,blocks)
            batches=[next_logical_batch(state.cursor,value['logical_batch_targets']) for _ in range(value['horizon'])]
        finally:
            _restore_rng(outside_rng)
        task=MetaTask(snapshot,batches,blocks,{'query_seed':entry.get('query_seed',value['seed'])},
                      {**entry['provenance'],'device':value['device']})
        task.validate(value['horizon'])
        if task.provenance.get('synthetic') is not synthetic:
            raise ValueError('Campaign/task synthetic provenance differs')
        pool.append(task)
        del state
    # Validate mixture without advancing the actual sampling stream.
    sample_meta_task(pool,random.Random(value['seed']),{'weights':value['weights'],'horizon':value['horizon']})
    output=Path(value['output'])
    output.mkdir(parents=True,exist_ok=False)
    def write(name,obj):
        temporary=output/(name+'.tmp')
        with temporary.open('xb') as handle:
            handle.write(canonical_bytes(obj)+b'\n')
            handle.flush()
            os.fsync(handle.fileno())
        if (output/name).exists():
            raise FileExistsError(output/name)
        os.rename(temporary,output/name)
        descriptor=os.open(output,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    write('resolved.json',value)
    write('progress-0.json',{'gate':'WP06','status':'running','outer':outer.state_dict(),'completed_generations':completed,
                           'preparation_seconds':time.perf_counter()-started})
    try:
        for generation in range(completed,value['generations']):
            if time.perf_counter()>deadline:
                raise RuntimeError('Frozen campaign deadline exhausted')
            selected=pool[schedule[generation]] if schedule is not None else sample_meta_task(pool,rng,{'weights':value['weights'],'horizon':value['horizon']})
            task=replace(selected,provenance={**selected.provenance,'journal':str(output/f'generation-{generation+1}-journal.jsonl'),'deadline_monotonic':deadline})
            perturbations=torch.randn(value['pairs'],outer.phi.numel(),generator=generator)
            task_index=next(i for i,t in enumerate(pool) if t is selected)
            write(f'generation-{generation+1}-inputs.json',{'family':value['family'],'phi_before':outer.phi.tolist(),
                'perturbations':perturbations.tolist(),'task_index':task_index,'task':value['pool'][task_index],
                'horizon':value['horizon'],'sigma':value['sigma'],'source_identity':source})
            record=train_generation(task,value['family'],outer,perturbations,value['horizon'],value['sigma'])
            record['task_index']=task_index
            write(f'generation-{generation+1}.json',record)
            completed=generation+1
            sampling=rng.getstate()
            write(f'outer-{completed}.json',{'outer':outer.state_dict(),'sampling_rng':[sampling[0],list(sampling[1]),sampling[2]],
                'perturbation_rng':generator.get_state().tolist(),'completed_generations':completed,
                'campaign_identity':identity,'source_identity':source,'family':value['family']})
        result={'status':'synthetic_smoke_completed' if synthetic else 'trained_exploratory_pending_two_stage_selection',
                'family':value['family'],'parameters':outer.phi.tolist(),'generations':value['generations'],
                'elapsed_seconds':time.perf_counter()-started,'confirmation_eligible':False}
        write('result.json',result)
        return result
    except BaseException as error:
        write('failure.json',{'status':'failed','error':str(error),'completed_generations':completed,
                             'elapsed_seconds':time.perf_counter()-started,'unknown_partial_work':True,
                             'recovery':'Only completed outer checkpoints may resume; retain interrupted journals and charge their work.'})
        raise
