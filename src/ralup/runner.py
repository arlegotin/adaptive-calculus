"""Locked synthetic paired execution and one-way scoring.

The actual scientific producer has not supplied a confirmation manifest. This
engine executes explicitly declared synthetic unigram byte SGD fixtures only;
it never trains or scores WikiText, restores historical state, or supplies n.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid

from .config import ConfirmationManifest, canonical_bytes
from .records import (ArtifactRef, IntegrityError, IncompleteConfirmation,
                      ProtectedAccessError, read_artifact, sha256_file, write_immutable_json)


def _ref(reference):
    return {'path':str(reference.path),'sha256':reference.sha256}


def _plain(value):
    return json.loads(canonical_bytes(value))


def load_confirmation_manifest(path: Path) -> ConfirmationManifest:
    path=Path(path)
    if not path.exists():
        from .evaluate import _refuse_design
        root=Path(__file__).resolve().parents[2]
        index=json.loads((root/'docs/harness/e02-evidence.json').read_bytes())
        attempt=next(row for row in index['attempts'] if row['attempt']=='plan-confirmation-v1')
        finish_ref=ArtifactRef(**attempt['files']['finish.json']);finish=read_artifact(finish_ref)
        design=next(row for row in finish['completed_objects'] if row['path']=='confirmation-plan.json')
        design_ref=ArtifactRef(finish_ref.path.parent/'confirmation-plan.json',design['sha256'])
        try:_refuse_design(design_ref)
        except IncompleteConfirmation as error:
            raise IncompleteConfirmation('No frozen scientific manifest; C/LR, fresh paired pilot and locked n unresolved. '
                'WP08 not attempted. Authenticated E02 producer refusal: '+str(error)) from error
        raise IntegrityError('Actual design producer unexpectedly advanced; integrate genuine production before execution')
    value=json.loads(path.read_bytes());manifest=ConfirmationManifest(value)
    if sha256_file(path)!=manifest.sha256:raise IntegrityError('Exact canonical manifest bytes required')
    _,_,reference=_contract(manifest)
    if reference.path.resolve()!=path.resolve():raise IntegrityError('Frozen manifest location mismatch')
    return manifest


def _contract(manifest):
    from .evaluate import _validate_contract, confirmation_access_events
    payload=_plain(manifest.payload);_validate_contract(payload)
    analysis=read_artifact(ArtifactRef(**payload['analysis']))
    if analysis.get('engine')!='synthetic-byte-unigram-v1':
        raise IncompleteConfirmation('Only explicit numerical synthetic engine supported; real authority absent')
    reference=ArtifactRef(analysis['manifest_path'],manifest.sha256)
    if read_artifact(reference)!=payload:raise IntegrityError('Frozen manifest identity mismatch')
    confirmation_access_events(reference)
    rng=random.Random(analysis['timing_seed']);orders=[]
    for seed in payload['seeds']:
        arms=payload['arms'].copy();rng.shuffle(arms);orders.append({'seed':seed,'arms':arms})
    if orders!=payload['timing_order']:raise IntegrityError('Arm order must be frozen randomized timing stream output')
    directory=Path(analysis['output_directory'])
    if directory.resolve()!=reference.path.parent.resolve()/'runs':
        raise IntegrityError('Synthetic runner output must be task-owned manifest sibling runs directory')
    configs={arm:read_artifact(ArtifactRef(**ref)) for arm,ref in payload['resolved_configs'].items()}
    fields={'synthetic','arm','endpoint','engine','training_text','warm_start_bytes','learning_rate','momentum','quantization','diverge_seeds'}
    for arm,config in configs.items():
        if set(config)!=fields or config['engine']!=analysis['engine'] or config['quantization']!='uniform4-logit-v1':
            raise IntegrityError('Exact synthetic numerical engine config required')
        if not isinstance(config['training_text'],str) or not config['training_text']:
            raise IntegrityError('Explicit synthetic training bytes required')
        if type(config['warm_start_bytes']) is not int or not 0<config['warm_start_bytes']<=1024:
            raise IntegrityError('Finite synthetic warm-start budget required')
        for name in ('learning_rate','momentum'):
            if type(config[name]) not in (int,float) or not math.isfinite(config[name]) or config[name]<0:
                raise IntegrityError('Finite nonnegative numerical update settings required')
        if config['momentum']>=1:raise IntegrityError('Momentum must be below one')
        if not isinstance(config['diverge_seeds'],list) or not set(config['diverge_seeds'])<=set(payload['seeds']):
            raise IntegrityError('Synthetic predeclared divergence seeds required')
    common={(c['training_text'],c['warm_start_bytes'],c['quantization']) for c in configs.values()}
    if len(common)!=1:raise IntegrityError('All arms must share byte schedule and warm start')
    if payload['endpoint']['target_bytes']>1024*1024 or payload['n']>256 or payload['endpoint']['continuation_steps']>1024:
        raise IntegrityError('Synthetic engineering bounds exceeded; no scientific budget fallback')
    return payload,analysis,reference


@contextmanager
def _exclusive(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'runner.lock').open('a+b') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def _artifact(path):
    return ArtifactRef(path,sha256_file(path))


def _put(path,value):
    if path.exists():
        reference=_artifact(path)
        if read_artifact(reference)!=value:raise IntegrityError('Immutable resume artifact conflict')
        return reference
    return write_immutable_json(path,value)


def _named_seed(seed,name):
    return int.from_bytes(hashlib.sha256(f'ralup.synthetic.e04:{seed}:{name}'.encode()).digest()[:8],'big')


def _quantize(weights):
    if not all(math.isfinite(w) for w in weights):raise ValueError('Synthetic logits must be finite before quantization')
    return [max(-8,min(7,round(max(-.8,min(.7,w))/.1)))*.1 for w in weights]


def _advance(state,config,count):
    """One pure finite synthetic full-byte SUM-gradient SGD step.

    Quantized forward probabilities and a straight-through logit derivative;
    the declared mean normalization is applied exactly once before SGD.
    No protected targets or hidden/global random state are consumed.
    """
    next_state=_plain(state);weights=state['weights'];logits=_quantize(weights)
    maximum=max(logits);exp=[math.exp(x-maximum) for x in logits];denominator=math.fsum(exp)
    probabilities=[x/denominator for x in exp];data=config['training_text'].encode('utf-8')
    histogram=[0]*256
    for index in range(count):histogram[data[(state['data_offset']+state['cursor']+index)%len(data)]]+=1
    gradients=[p-h/count for p,h in zip(probabilities,histogram)]
    momentum=[config['momentum']*old+g for old,g in zip(state['momentum'],gradients)]
    updated=[w-config['learning_rate']*g for w,g in zip(weights,momentum)]
    if not all(math.isfinite(w) for w in updated):raise FloatingPointError('Nonfinite synthetic finite-budget update')
    next_state.update(weights=updated,momentum=momentum,step=state['step']+1,cursor=state['cursor']+count,
                      schedule_step=state['schedule_step']+1)
    return next_state


def _event(identity,count,stream,classification,offset):
    return {'event_id':identity,'kind':'loss','targets':count,'stream_targets':stream,
        'classifications':classification,'source_intervals':None}


def _training_event(identity,count,classification,config,state,offset):
    from .analysis.cost import source_intervals
    event=_event(identity,count,count,classification,offset)
    data=config['training_text'].encode();document=hashlib.sha256(data).hexdigest()
    event['source_intervals']=source_intervals('synthetic-e04-training',[
        {'document_id':document,'offsets':[(state['data_offset']+offset+i)%len(data) for i in range(count)]}])
    return event


def _wall(start):
    return {'clock':'host-epoch-wall','start':start,'end':time.time()}


def _cost_run(identity,status,events,walls,parents=None,unknown=False):
    return {'run_id':identity,'status':status,'parents':parents or [],'events':events,
        'wall_intervals':walls,'unknown_partial_work':unknown,'common_engineering_seconds':None}


def _warm_start(root,manifest,seed,config):
    started=time.time()
    directory=root/f'seed-{seed}';directory.mkdir(exist_ok=True)
    receipt_path=directory/'warm-cost.json';execution_path=directory/'warm-execution.json'
    completed=None
    try:
        retained=[read_artifact(_artifact(p)) for p in sorted(directory.glob('interrupted-*.json'))]
        if execution_path.exists():completed=read_artifact(_artifact(execution_path))
        elif retained:
            candidates=[r.get('completed_warm') for r in retained]
            if any(c is None for c in candidates):
                raise IntegrityError('Unknown prior warm execution: refuse recomputation; retain partial evidence')
            if any(c!=candidates[0] for c in candidates):raise IntegrityError('Conflicting completed warm evidence')
            completed=candidates[0]
        elif (directory/'warm-state.json').exists() or receipt_path.exists():
            raise IntegrityError('Warm persistence has no authenticated completed execution; refuse recomputation')
        if completed is None:
            rng=random.Random(_named_seed(seed,'initialization'))
            streams={name:_named_seed(seed,name) for name in ('initialization','data','quantizer','timing')}
            state={'schema':'ralup.synthetic_full_state.v1','synthetic':True,'manifest_sha256':manifest.sha256,
                'seed':seed,'weights':[rng.uniform(-.01,.01) for _ in range(256)],'momentum':[0.]*256,
                'step':0,'cursor':0,'data_offset':random.Random(streams['data']).randrange(len(config['training_text'].encode())),
                'named_streams':streams,'warm_updates':0,
                'quantizer_state':{'format':'uniform4-logit-v1','fixed_scale':.1},'histories':{},'schedule_step':0}
            state=_advance(state,{**config,'learning_rate':.1,'momentum':0.},config['warm_start_bytes'])
            state.update(step=0,cursor=0,warm_updates=1,schedule_step=0)
            completed={'synthetic':True,'manifest_sha256':manifest.sha256,'seed':seed,'state':state,
                'state_sha256':hashlib.sha256(canonical_bytes(state)).hexdigest(),
                'run':_cost_run(f'{manifest.sha256}:seed-{seed}:warm','completed',
                    [_training_event(f'{seed}:warm',config['warm_start_bytes'],['warm_start','backward'],config,state,0)],[_wall(started)])}
        state=completed['state']
        if (completed['manifest_sha256']!=manifest.sha256 or completed['seed']!=seed
                or state['manifest_sha256']!=manifest.sha256 or state['seed']!=seed
                or state['warm_updates']!=1 or completed['state_sha256']!=hashlib.sha256(canonical_bytes(state)).hexdigest()):
            raise IntegrityError('Completed warm state/source/physical execution binding mismatch')
        state_ref=_put(directory/'warm-state.json',state)
        execution_ref=_put(execution_path,completed)
        if receipt_path.exists():
            receipt_ref=_artifact(receipt_path);receipt=read_artifact(receipt_ref)
            if receipt['state']!=_ref(state_ref) or receipt['execution']!=_ref(execution_ref):
                raise IntegrityError('Warm cost publication conflicts with completed execution')
        else:
            receipt={'synthetic':True,'state':_ref(state_ref),'execution':_ref(execution_ref),'run':completed['run'],
                'reset':'none; identical full synthetic W/momentum/cursor/named streams per arm'}
            receipt_ref=_put(receipt_path,receipt)
        publication=_publication_receipt(directory/'warm-publication.json',manifest,
            {'cost':_ref(receipt_ref),'state':_ref(state_ref),'execution':_ref(execution_ref)},
            [receipt['run']]+[r['run'] for r in retained],started)
        return state,state_ref,receipt_ref,{**receipt,'run':publication['run']}
    except BaseException as error:
        _retained_attempt(directory,manifest,seed,'warm',started,error,completed_warm=completed)
        raise


def _retained_attempt(directory,manifest,seed,arm,started,error,*,completed_warm=None):
    status='interrupted' if isinstance(error,KeyboardInterrupt) else 'implementation_failure'
    return write_immutable_json(directory/f'{status}-{uuid.uuid4().hex}.json',{
        'synthetic':True,'manifest_sha256':manifest.sha256,'seed':seed,'arm':arm,'status':status,
        'utc':datetime.now(timezone.utc).isoformat(),'local_time':datetime.now().astimezone().isoformat(),
        'reason':str(error),'unknown_partial_work':completed_warm is None,'completed_warm':completed_warm,
        'unknown_persistence_elapsed':True,
        'run':_cost_run(f'{manifest.sha256}:retained-{uuid.uuid4().hex}','failed',
            [] if completed_warm is None else completed_warm['run']['events'],[_wall(started)],unknown=completed_warm is None)})


def _publication_receipt(path,manifest,bindings,parents,started):
    """Retain measured required-publication intervals, excluding this footer's I/O."""
    if path.exists():
        value=read_artifact(_artifact(path))
        if value['manifest_sha256']!=manifest.sha256 or value['bindings']!=bindings:
            raise IntegrityError('Publication receipt identity mismatch')
        return value
    value={'synthetic':True,'manifest_sha256':manifest.sha256,'bindings':bindings,
        'run':_cost_run(f'{manifest.sha256}:{path.relative_to(path.parents[2])}:publication','completed',[],
                        None if started is None else [_wall(started)],parents=parents),
        'unmeasured_components':['publication-cost footer persistence'],
        'cost_status':'known lower bound; complete elapsed unavailable'}
    _put(path,value)
    return value


def _cost_lower_bound(roots):
    from .analysis.cost import _totals
    value=_totals(roots)
    value.update(seconds=None,complete_seconds_available=False,
        unmeasured_components=['publication-cost footer persistence; interrupted persistence if present'])
    return value


def run_confirmation(manifest: ConfirmationManifest) -> list[ArtifactRef]:
    """Execute the fixed complete inventory; resume immutable full-state steps.

    Diverged seeds retain costs and no final checkpoint. Infrastructure failure
    amends/invalidate this manifest; a documented new version must rerun affected
    units. Keyboard interruption resumes with original and uncertain work kept.
    """
    from .evaluate import confirmation_access_events,record_confirmation_amendment,validate_confirmation
    payload,analysis,reference=_contract(manifest);root=Path(analysis['output_directory'])
    refs=[]
    with _exclusive(root):
        events=confirmation_access_events(reference)
        if any(row['kind']=='amendment' for row in events):raise ProtectedAccessError('Implementation-invalidated manifest; retain originals and versioned rerun required')
        configs={arm:read_artifact(ArtifactRef(**value)) for arm,value in payload['resolved_configs'].items()}
        for order in payload['timing_order']:
            seed=order['seed']
            try:warm,warm_ref,warm_cost_ref,warm_cost=_warm_start(root,manifest,seed,configs['C'])
            except BaseException as error:
                if not isinstance(error,KeyboardInterrupt):
                    record_confirmation_amendment(reference,{'implementation_invalidation':str(error),
                        'affected_seeds':payload['seeds'],'affected_arms':payload['arms'],
                        'disposition':'warm-start failure; all bug-affected units invalid; retain originals and versioned rerun required'})
                raise
            for arm in order['arms']:
                directory=root/f'seed-{seed}'/arm;directory.mkdir(exist_ok=True)
                terminal_path=directory/'terminal.json'
                if terminal_path.exists():
                    terminal_ref=_artifact(terminal_path);terminal=read_artifact(terminal_ref)
                    cost=read_artifact(ArtifactRef(**terminal['costs']))
                    _publication_receipt(directory/'terminal-publication.json',manifest,
                        {'terminal':_ref(terminal_ref),'cost':terminal['costs']},[cost['run']],None)
                    refs.append(terminal_ref);continue
                if any(row['kind']=='opened' for row in events):raise ProtectedAccessError('Training cannot resume after protected inspection')
                started=time.time();config=configs[arm];state=_plain(warm);state_ref=warm_ref;steps=[]
                try:
                    step_count=payload['endpoint']['continuation_steps'];budget=payload['endpoint']['target_bytes']
                    if budget<step_count:raise IntegrityError('Each declared step requires at least one target byte')
                    for step in range(1,step_count+1):
                        path=directory/f'step-{step:04d}.json'
                        count=budget*step//step_count-budget*(step-1)//step_count
                        if path.exists():
                            saved=read_artifact(_artifact(path))
                            if (saved['parent_state']!=_ref(state_ref) or saved['config_sha256']!=payload['resolved_configs'][arm]['sha256']
                                    or saved['state']['manifest_sha256']!=manifest.sha256 or saved['state']['step']!=step
                                    or saved['state']['cursor']!=budget*step//step_count):
                                raise IntegrityError('Full-state resume ancestry/config/endpoint mismatch')
                            state=saved['state'];state_ref=_artifact(path);steps.append(saved);continue
                        step_start=time.time()
                        if seed in config['diverge_seeds']:
                            raise FloatingPointError('Predeclared synthetic adversarial finite-budget divergence')
                        state=_advance(state,config,count)
                        saved={'synthetic':True,'state':state,'parent_state':_ref(state_ref),
                            'config_sha256':payload['resolved_configs'][arm]['sha256'],
                            'event':_training_event(f'{seed}:{arm}:step-{step}',count,['optimizer_stream','backward'],config,state,budget*(step-1)//step_count),
                            'wall':_wall(step_start)}
                        state_ref=_put(path,saved);steps.append(saved)
                    status='completed';failure=None
                except FloatingPointError as error:
                    status='diverged';failure={'disposition':'substantive_failure','reason':str(error),'replacement':False}
                except BaseException as error:
                    retained=_retained_attempt(directory,manifest,seed,arm,started,error)
                    if not isinstance(error,KeyboardInterrupt):
                        record_confirmation_amendment(reference,{'implementation_invalidation':str(error),
                            'affected_seeds':payload['seeds'],'affected_arms':payload['arms'],'retained_attempt':_ref(retained),
                            'disposition':'all bug-affected arms invalid; originals retained; documented versioned rerun required'})
                    raise
                retained=[read_artifact(_artifact(p)) for p in sorted(directory.glob('interrupted-*.json'))]
                checkpoint=None;export=None;checkpoint_started=time.time()
                if status=='completed':
                    checkpoint=_ref(_put(directory/'final-checkpoint.json',{
                        'schema':'ralup.synthetic_final_checkpoint.v1','synthetic':True,'seed':seed,'arm':arm,
                        'endpoint':payload['endpoint'],'config_sha256':payload['resolved_configs'][arm]['sha256'],
                        'source_identity':payload['source_identity'],'runtime_identity':payload['runtime_identity'],'final':True}))
                    export=_put(directory/'quantized-logits.json',{'synthetic':True,'manifest_sha256':manifest.sha256,
                        'seed':seed,'arm':arm,'full_state':_ref(state_ref),'quantization':config['quantization'],
                        'quantized_logits':_quantize(state['weights'])})
                checkpoint_seconds=time.time()-checkpoint_started
                run=_cost_run(f'{manifest.sha256}:seed-{seed}:{arm}',status,[r['event'] for r in steps],
                    [r['wall'] for r in steps]+[_wall(started)],parents=[warm_cost['run']]+[r['run'] for r in retained],
                    unknown=bool(retained) or status=='diverged' and seed not in config['diverge_seeds'])
                costs={'synthetic':True,'manifest_sha256':manifest.sha256,'seed':seed,'arm':arm,
                    'initial_state_sha256':warm_ref.sha256,'parent':_ref(warm_cost_ref),'final_state':_ref(state_ref),
                    'stream_targets':state['cursor'],'run':run,'unknown_partial_work':run['unknown_partial_work'],
                    'optimizer_updates':len(steps),'query_passes':0,'gradient_only_passes':0,'hvp_passes':0,
                    'checkpoint_io_seconds':checkpoint_seconds,
                    'feature_scans':0,'quantizer_passes':len(steps),'transfer_bytes':0,
                    'export':None if export is None else _ref(export),
                    'export_bytes':None if export is None else export.path.stat().st_size,
                    'full_state_bytes':state_ref.path.stat().st_size,'rss_bytes':None,'mps_allocated_bytes':None,
                    'mps_driver_bytes':None,'synchronized_throughput':None,'time_to_development_target':None,
                    'cost_scope':'actual synthetic execution only; offline tuning/meta and enclosing engineering cost unknown'}
                costs_path=directory/'costs.json'
                if costs_path.exists():
                    costs_ref=_artifact(costs_path);existing=read_artifact(costs_ref)
                    if existing['final_state']!=_ref(state_ref) or existing['initial_state_sha256']!=warm_ref.sha256:
                        raise IntegrityError('Retained terminal cost/state identity conflict')
                else:costs_ref=_put(costs_path,costs)
                terminal={'schema':'ralup.synthetic_terminal.v1','manifest_sha256':manifest.sha256,
                    'seed':seed,'arm':arm,'status':status,'endpoint':payload['endpoint'],
                    'config_sha256':payload['resolved_configs'][arm]['sha256'],'source_identity':payload['source_identity'],
                    'runtime_identity':payload['runtime_identity'],
                    'lineage':{'unit':f'fresh-{manifest.sha256}-{seed}','shared_parent':warm_ref.sha256},
                    'checkpoint':checkpoint,'failure':failure,'costs':_ref(costs_ref)}
                terminal_ref=_put(terminal_path,terminal)
                _publication_receipt(directory/'terminal-publication.json',manifest,
                    {'terminal':_ref(terminal_ref),'cost':_ref(costs_ref)},[run],started)
                refs.append(terminal_ref)
        validate_confirmation(reference,terminal_inventory(refs))
    return refs


def terminal_inventory(references: list[ArtifactRef]) -> list[dict]:
    return [{'seed':row['seed'],'arm':row['arm'],'terminal':_ref(reference)}
            for reference in references for row in [read_artifact(reference)]]


def invalidate_confirmation(manifest,terminals,reason,*,affected_seeds):
    """Invalidate every arm of documented bug-affected units without replacement."""
    from .evaluate import record_confirmation_amendment
    payload,analysis,reference=_contract(manifest);root=Path(analysis['output_directory'])
    if not isinstance(reason,str) or not reason.strip() or not affected_seeds or not set(affected_seeds)<=set(payload['seeds']):
        raise ValueError('Documented reason and planned affected seeds required')
    with _exclusive(root):
        affected=[_ref(r) for r in terminals if read_artifact(r)['seed'] in affected_seeds]
        identities={(read_artifact(r)['seed'],read_artifact(r)['arm']) for r in terminals if read_artifact(r)['seed'] in affected_seeds}
        if identities!={(s,a) for s in affected_seeds for a in payload['arms']}:raise IntegrityError('Invalidate every bug-affected paired arm')
        receipt=write_immutable_json(root/f'invalidation-{uuid.uuid4().hex}.json',{'synthetic':True,
            'manifest_sha256':manifest.sha256,'reason':reason,'affected_seeds':affected_seeds,
            'affected_terminals':affected,'disposition':'invalidated; originals retained; documented versioned rerun required'})
        record_confirmation_amendment(reference,{'implementation_invalidation':_ref(receipt)})
        return receipt


def _score_model(state):
    import torch
    class SyntheticByteModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.logits=torch.nn.Parameter(torch.tensor(_quantize(state['weights']),dtype=torch.float32),requires_grad=False)
        def forward(self,inputs,positions,padding,contexts=None):
            return self.logits.expand(*inputs.shape,256)
    return SyntheticByteModel()


def _scoring_state(terminal,manifest):
    costs=read_artifact(ArtifactRef(**terminal['costs']));reference=ArtifactRef(**costs['final_state'])
    saved=read_artifact(reference);state=saved['state'] if 'state' in saved else saved
    if (costs['manifest_sha256']!=manifest.sha256 or state['manifest_sha256']!=manifest.sha256
            or (costs['seed'],costs['arm'])!=(terminal['seed'],terminal['arm'])
            or saved.get('config_sha256')!=terminal['config_sha256']
            or state['seed']!=terminal['seed'] or state['step']!=manifest.payload['endpoint']['continuation_steps']
            or state['cursor']!=manifest.payload['endpoint']['target_bytes']):
        raise IntegrityError('Exact full-state final endpoint required before protected payload opening')
    exported=read_artifact(ArtifactRef(**costs['export']))
    if (exported['full_state']!=costs['final_state'] or exported['manifest_sha256']!=manifest.sha256
            or (exported['seed'],exported['arm'])!=(terminal['seed'],terminal['arm'])
            or exported['quantized_logits']!=_quantize(state['weights'])):
        raise IntegrityError('Exact final quantized export/full-state binding required')
    return costs,state


def _terminal_cost_run(terminal,terminal_ref,manifest):
    cost=read_artifact(ArtifactRef(**terminal['costs']))
    path=terminal_ref.path.parent/'terminal-publication.json'
    if not path.exists():raise IntegrityError('Missing retained terminal publication receipt; resume inventory first')
    value=read_artifact(_artifact(path))
    if (value['manifest_sha256']!=manifest.sha256
            or value['bindings']!={'terminal':_ref(terminal_ref),'cost':terminal['costs']}):
        raise IntegrityError('Exact terminal/cost publication binding required')
    return value['run']


def published_confirmation_results(reference):
    """Verified result readback with attributable post-publication lower bounds.

    Result bytes remain immutable. The sidecar binds those exact bytes and the
    successful E03 audit event; its own persistence remains unmeasured explicitly.
    An absent sidecar leaves the original conservative lower-bound costs intact.
    """
    from .evaluate import existing_protected_result,confirmation_access_events
    value=read_artifact(reference)
    manifest_ref=ArtifactRef(**value['manifest_reference'])
    if existing_protected_result(manifest_ref)!=reference:raise ProtectedAccessError('No verified locked result readback')
    path=reference.path.with_name('scoring-publication.json')
    if not path.exists():return value
    publication=read_artifact(_artifact(path))
    events=confirmation_access_events(manifest_ref)
    result_event=next(row for row in events if row['kind']=='result')
    if (publication['manifest_sha256']!=value['manifest_sha256']
            or publication['bindings']!={'result':_ref(reference),'audit_event_id':result_event['event_id']}):
        raise IntegrityError('Post-publication result/audit identity mismatch')
    runs=[]
    for row,run in zip(value['outcomes'],value['execution_runs']):
        complete=_cost_run(run['run_id']+':publication-attributed',run['status'],[],[],parents=[run,publication['run']])
        row['deployment_costs']=_cost_lower_bound([complete]);runs.append(complete)
    value['project_costs']=_cost_lower_bound(runs)
    for curve,row in zip(value['quality_cost_curves'],value['outcomes']):curve['deployment_costs']=row['deployment_costs']
    return value


def score_confirmation(manifest: ConfirmationManifest, inventory: list[dict]) -> ArtifactRef:
    """Score once after complete authenticated final inventory, or verified readback.

    The public E03 payload_ready protocol authenticates/materializes every byte
    before publication. Reservation/failed/pending opens never authorize reuse.
    """
    started=time.time()
    from .evaluate import (validate_confirmation,existing_protected_result,open_protected_targets,
                          record_protected_result,score_blocks)
    from .analysis.cost import source_intervals
    payload,analysis,reference=_contract(manifest);root=Path(analysis['output_directory'])
    with _exclusive(root):
        validate_confirmation(reference,inventory)
        existing=existing_protected_result(reference)
        if existing is not None:
            value=read_artifact(existing)
            if value.get('inventory_sha256')!=hashlib.sha256(canonical_bytes({'inventory':inventory})).hexdigest():
                raise IntegrityError('Repeated scoring requires exact original terminal inventory')
            return existing
        rows=[read_artifact(ArtifactRef(**row['terminal'])) for row in inventory]
        prepared={}
        terminal_runs={}
        for row in rows:
            if row['status']=='completed':prepared[(row['seed'],row['arm'])]=_scoring_state(row,manifest)
            else:read_artifact(ArtifactRef(**row['costs']))
            terminal_ref=ArtifactRef(**next(r['terminal'] for r in inventory if (r['seed'],r['arm'])==(row['seed'],row['arm'])))
            terminal_runs[(row['seed'],row['arm'])]=_terminal_cost_run(row,terminal_ref,manifest)
        # There are no payload bytes before all contract/terminal/state checks.
        blocks=open_protected_targets(phase='confirm',manifest_ref=reference,inventory=inventory)
        common_walls=[_wall(started)]
        outcomes=[];project_runs=[];curves=[];scored_targets=0
        partial_events=[];partial_targets=0;current_arm=None
        try:
            for row in rows:
                arm_started=time.time();score=None;costs=read_artifact(ArtifactRef(**row['costs']))
                run=terminal_runs[(row['seed'],row['arm'])]
                if row['status']=='completed':
                    _,state=prepared[(row['seed'],row['arm'])];score_start=time.time()
                    current_arm={'seed':row['seed'],'arm':row['arm']};partial_events=[];partial_targets=0
                    def progress(details):
                        nonlocal partial_targets
                        if details['phase']!='block_completed':return
                        partial_targets=details['completed_targets']
                        event=_event(f"{row['seed']}:{row['arm']}:protected-block-{details['completed_blocks']}",
                            details['block_targets'],0,['declared_final_evaluation'],0)
                        block=next(b for b in blocks if (b.document_id,b.target_start,b.target_end)==
                            (details['document_id'],details['target_start'],details['target_end']))
                        event['source_intervals']=source_intervals('synthetic-e04-protected',[
                            {'document_id':block.document_id,'offsets':list(block.scored_offsets())}])
                        partial_events.append(event)
                    score=score_blocks(_score_model(state),blocks,evaluation_context={'partition':'fixture'},progress_callback=progress)
                    scored_targets+=score.target_count
                    run=_cost_run(run['run_id']+':scoring','completed',partial_events,[_wall(score_start)],parents=[run])
                    partial_events=[];partial_targets=0
                deployment=_cost_lower_bound([run])
                outcome={'seed':row['seed'],'arm':row['arm'],'unit':row['lineage']['unit'],
                    'status':row['status'],'bpb':None if score is None else score.bpb,
                    'nll_sum':None if score is None else score.nll_sum,'target_count':None if score is None else score.target_count,
                    'terminal':next(r['terminal'] for r in inventory if (r['seed'],r['arm'])==(row['seed'],row['arm'])),
                    'deployment_costs':deployment,'export_bytes':costs['export_bytes'],
                    'per_article':None if score is None else score.per_article}
                outcomes.append(outcome);curves.append({'seed':row['seed'],'arm':row['arm'],'status':row['status'],
                    'endpoint':payload['endpoint'],'bpb':outcome['bpb'],'deployment_costs':deployment,
                    'boundary':'deployment known lower bounds including attributable warm-start and required publication; footer persistence elapsed unmeasured; offline search/meta separate'})
                # Includes each arm's final score and outcome bookkeeping, without
                # charging other arms' execution to this deployment.
                run=_cost_run(run['run_id']+':bookkeeping',row['status'],[],[_wall(arm_started)],parents=[run])
                project_runs.append(run)
            publication_started=time.time();project=_cost_lower_bound(project_runs)
            value={'schema':'ralup.synthetic_paired_results.v1','synthetic':True,'manifest_sha256':manifest.sha256,
                'inventory_sha256':hashlib.sha256(canonical_bytes({'inventory':inventory})).hexdigest(),
                'outcomes':outcomes,'scored_targets':scored_targets,'project_costs':project,'quality_cost_curves':curves,
                'manifest_reference':_ref(reference),'execution_runs':project_runs,
                'offline_search_meta_costs':None,'common_engineering_costs':None,
                'scope':'numerically executed synthetic fixtures only; genuine scientific confirmation unavailable'}
            result=write_immutable_json(root/'paired-results.json',value)
            result_event=record_protected_result(reference,_ref(result))
            common_walls.append(_wall(publication_started))
            publication={'synthetic':True,'manifest_sha256':manifest.sha256,
                'bindings':{'result':_ref(result),'audit_event_id':result_event['event_id']},
                'run':_cost_run(manifest.sha256+':common-scoring-publication','completed',[],common_walls),
                'unmeasured_components':['publication-cost footer persistence'],
                'cost_status':'known lower bound; complete elapsed unavailable'}
            write_immutable_json(root/'scoring-publication.json',publication)
            return result
        except BaseException as error:
            write_immutable_json(root/f'scoring-failed-{uuid.uuid4().hex}.json',{
                'synthetic':True,'manifest_sha256':manifest.sha256,'reason':str(error),'scored_targets':scored_targets,
                'completed_partial_targets':partial_targets,'completed_partial_events':partial_events,'current_arm':current_arm,
                'retained_completed_runs':project_runs,'unknown_partial_work':True,'wall':_wall(started),
                'disposition':'one-way scorer consumed; no retry or result replacement'})
            raise
