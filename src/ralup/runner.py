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
    return [max(-8,min(7,round(w/.1)))*.1 for w in weights]


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
    directory=root/f'seed-{seed}';directory.mkdir(exist_ok=True)
    receipt_path=directory/'warm-cost.json'
    if receipt_path.exists():
        receipt=read_artifact(_artifact(receipt_path));state_ref=ArtifactRef(**receipt['state'])
        state=read_artifact(state_ref)
        if state['manifest_sha256']!=manifest.sha256 or state['seed']!=seed:
            raise IntegrityError('Shared parent must bind current manifest/source/runtime')
        return state,state_ref,_artifact(receipt_path),receipt
    started=time.time();rng=random.Random(_named_seed(seed,'initialization'))
    streams={name:_named_seed(seed,name) for name in ('initialization','data','quantizer','timing')}
    state={'schema':'ralup.synthetic_full_state.v1','synthetic':True,'manifest_sha256':manifest.sha256,
        'seed':seed,'weights':[rng.uniform(-.01,.01) for _ in range(256)],'momentum':[0.]*256,
        'step':0,'cursor':0,'data_offset':random.Random(streams['data']).randrange(len(config['training_text'].encode())),
        'named_streams':streams,'warm_updates':0,
        'quantizer_state':{'format':'uniform4-logit-v1','fixed_scale':.1},'histories':{},'schedule_step':0}
    warm_config={**config,'learning_rate':.1,'momentum':0.}
    try:state=_advance(state,warm_config,config['warm_start_bytes'])
    except BaseException as error:
        _retained_attempt(directory,manifest,seed,'warm',started,error)
        raise
    state.update(step=0,cursor=0,warm_updates=1,schedule_step=0)
    state_ref=_put(directory/'warm-state.json',state)
    retained=[read_artifact(_artifact(p)) for p in sorted(directory.glob('interrupted-*.json'))]
    cost=_cost_run(f'{manifest.sha256}:seed-{seed}:warm','completed',
        [_training_event(f'{seed}:warm',config['warm_start_bytes'],['warm_start','backward'],config,state,0)],
        [_wall(started)],parents=[r['run'] for r in retained],unknown=bool(retained))
    receipt={'synthetic':True,'state':_ref(state_ref),'run':cost,'reset':'none; identical full synthetic W/momentum/cursor/named streams per arm'}
    return state,state_ref,_put(receipt_path,receipt),receipt


def _retained_attempt(directory,manifest,seed,arm,started,error):
    status='interrupted' if isinstance(error,KeyboardInterrupt) else 'implementation_failure'
    return write_immutable_json(directory/f'{status}-{uuid.uuid4().hex}.json',{
        'synthetic':True,'manifest_sha256':manifest.sha256,'seed':seed,'arm':arm,'status':status,
        'utc':datetime.now(timezone.utc).isoformat(),'local_time':datetime.now().astimezone().isoformat(),
        'reason':str(error),'unknown_partial_work':True,
        'run':_cost_run(f'{manifest.sha256}:retained-{uuid.uuid4().hex}','failed',[],[_wall(started)],unknown=True)})


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
                if terminal_path.exists():refs.append(_artifact(terminal_path));continue
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
                refs.append(_put(terminal_path,terminal))
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


def score_confirmation(manifest: ConfirmationManifest, inventory: list[dict]) -> ArtifactRef:
    """Score once after complete authenticated final inventory, or verified readback.

    The public E03 payload_ready protocol authenticates/materializes every byte
    before publication. Reservation/failed/pending opens never authorize reuse.
    """
    from .evaluate import (validate_confirmation,existing_protected_result,open_protected_targets,
                          record_protected_result,score_blocks)
    from .analysis.cost import _totals,source_intervals
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
        for row in rows:
            if row['status']=='completed':prepared[(row['seed'],row['arm'])]=_scoring_state(row,manifest)
            else:read_artifact(ArtifactRef(**row['costs']))
        # There are no payload bytes before all contract/terminal/state checks.
        blocks=open_protected_targets(phase='confirm',manifest_ref=reference,inventory=inventory)
        outcomes=[];project_runs=[];curves=[];scored_targets=0;started=time.time()
        partial_events=[];partial_targets=0;current_arm=None
        try:
            for row in rows:
                score=None;costs=read_artifact(ArtifactRef(**row['costs']));run=costs['run']
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
                project_runs.append(run);deployment=_totals([run])
                outcome={'seed':row['seed'],'arm':row['arm'],'unit':row['lineage']['unit'],
                    'status':row['status'],'bpb':None if score is None else score.bpb,
                    'nll_sum':None if score is None else score.nll_sum,'target_count':None if score is None else score.target_count,
                    'terminal':next(r['terminal'] for r in inventory if (r['seed'],r['arm'])==(row['seed'],row['arm'])),
                    'deployment_costs':deployment,'export_bytes':costs['export_bytes'],
                    'per_article':None if score is None else score.per_article}
                outcomes.append(outcome);curves.append({'seed':row['seed'],'arm':row['arm'],'status':row['status'],
                    'endpoint':payload['endpoint'],'bpb':outcome['bpb'],'deployment_costs':deployment,
                    'boundary':'complete deployment including attributable warm-start, all adaptation, final evaluation and checkpoint; offline search/meta separate unknown'})
            project=_totals(project_runs)
            value={'schema':'ralup.synthetic_paired_results.v1','synthetic':True,'manifest_sha256':manifest.sha256,
                'inventory_sha256':hashlib.sha256(canonical_bytes({'inventory':inventory})).hexdigest(),
                'outcomes':outcomes,'scored_targets':scored_targets,'project_costs':project,'quality_cost_curves':curves,
                'offline_search_meta_costs':None,'common_engineering_costs':None,
                'scope':'numerically executed synthetic fixtures only; genuine scientific confirmation unavailable'}
            result=write_immutable_json(root/'paired-results.json',value)
            record_protected_result(reference,_ref(result))
            return result
        except BaseException as error:
            write_immutable_json(root/f'scoring-failed-{uuid.uuid4().hex}.json',{
                'synthetic':True,'manifest_sha256':manifest.sha256,'reason':str(error),'scored_targets':scored_targets,
                'completed_partial_targets':partial_targets,'completed_partial_events':partial_events,'current_arm':current_arm,
                'retained_completed_runs':project_runs,'unknown_partial_work':True,'wall':_wall(started),
                'disposition':'one-way scorer consumed; no retry or result replacement'})
            raise
