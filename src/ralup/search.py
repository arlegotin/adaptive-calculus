"""Authenticated S12/S26 development trials; protected scoring has no route here.

Trial configs live at artifacts/search/configs/<hash>.json. These small immutable
manifests point to task-owned permanent output directories. TrialSpec identifies
one prospective run, never an outcome-selected configuration or hidden fallback.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

from .config import canonical_bytes
from .records import ArtifactRef, sha256_file


@dataclass(frozen=True)
class TrialSpec:
    config_hash: str
    family: str
    seed: int
    budget: int
    source_revision: str

    def __post_init__(self):
        ArtifactRef('config', self.config_hash)
        if (not self.family or not self.source_revision or type(self.seed) is not int
                or self.seed < 0 or type(self.budget) is not int or self.budget <= 0):
            raise ValueError('Trial requires family, source, seed and finite positive budget')


def _write(path, value):
    from .calibrate import _write as write
    return write(Path(path), value)


def _ref(ref):
    return {'path': str(ref.path), 'sha256': ref.sha256}


def _read(ref):
    if sha256_file(ref.path) != ref.sha256:
        raise ValueError('Trial evidence hash mismatch')
    return json.loads(ref.path.read_bytes())


def _development_only(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if 'test' in key.lower() or key.lower().startswith('protected'):
                raise ValueError('protected outcome forbidden in selection inputs')
            _development_only(item)
    elif isinstance(value, list):
        for item in value:
            _development_only(item)


def source_identity():
    root = Path(__file__).resolve().parents[2]
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted((root/'src/ralup').rglob('*.py'))}


def _validate_recipe_metadata(recipe):
    """Declared annotations must describe the path actually selected by name.

    Normalization means the extra named matrix normalization order. Original-W
    moment coordinates are independent of that order and of storage grouping.
    """
    from .quantization.scales import grouping_for
    if 'quantizer' in recipe:
        expected = 'none' if recipe['quantizer']=='fp32' else grouping_for(recipe['quantizer'])
        aliases = {'none':{'none'}, 'last_axis_row':{'row','last_axis_row'},
                   'whole_tensor':{'tensor','whole_tensor'}}
        if 'grouping' in recipe and recipe['grouping'] not in aliases[expected]:
            raise ValueError('Declared grouping differs from the named quantizer')
    method = recipe['method']
    normalization = method.removeprefix('muon_plus_').removesuffix('_fp32') if method.startswith('muon_plus_') else 'none'
    if 'normalization' in recipe and recipe['normalization'] != normalization:
        raise ValueError('Declared normalization differs from the named matrix method; moment coordinates are separate')
    if recipe.get('moment_coordinates','original_W') != 'original_W':
        raise ValueError('Moment coordinates must remain original_W')


def register_trial(config):
    """Freeze config before running; existing byte-identical registrations are safe."""
    _development_only(config)
    digest = hashlib.sha256(canonical_bytes(config)).hexdigest()
    path = Path('artifacts/search/configs')/(digest+'.json')
    if path.exists():
        if json.loads(path.read_bytes()) != config: raise ValueError('Config hash collision')
    else:
        _write(path, config)
    return TrialSpec(digest, config['family'], config['seed'], config['budget'], config['source_revision'])


def screen_headroom(trials: list[ArtifactRef]) -> dict:
    rows = [_read(ref) for ref in trials]
    for row in rows: _development_only(row)
    complete = [r for r in rows if r['status'] == 'completed']
    for r in complete:
        bounds = r['development_interval']
        if len(bounds) != 2 or not all(type(v) in (float,int) and math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
            raise ValueError('Finite ordered development uncertainty interval required')
    def stratum(row):
        return (row['model'], row['partition'], row['budget'])
    leaders = {}
    for row in complete:
        key = stratum(row)
        if key not in leaders or row['development_interval'][1] < leaders[key]['development_interval'][1]:
            leaders[key] = row
    advance, expand, reasons, blocked = [], [], {}, []
    for row in rows:
        name = row['trial_id']; why = []
        if row['status'] != 'completed':
            blocked.append(name); reasons[name] = ['operation/resource failure is not defeat']; continue
        if not row.get('settled',False): why.append('unsettled curve')
        if row.get('edge',False) or not row.get('lr_bracketed',False):
            why.append('expand LR bracket'); expand.append(name)
        if row.get('missing_mechanism',False): why.append('scale-dependent mechanism absent')
        if row['development_interval'][0] <= leaders[stratum(row)]['development_interval'][1]: why.append('overlaps leader')
        if row.get('eliminates_headroom',False): why.append('retain headroom-eliminating quantizer')
        if why: advance.append(name)
        reasons[name] = why or ['settled and separated at this measured horizon only']
    return {'advance':advance, 'expand':expand, 'blocked':blocked,'reasons':reasons,
            'selection_ready':bool(complete) and not blocked and not expand and
                all(r.get('settled',False) and not r.get('missing_mechanism',False) for r in complete),
            'reference_gap_is_upper_bound':False,'short_proxy_elimination_permitted':False}


def select_backbone(development_results: list[dict], rationale: dict) -> ArtifactRef:
    _development_only(development_results); _development_only(rationale)
    if not development_results or any(r.get('evidence_kind') != 'real_data' for r in development_results):
        raise ValueError('B0/disposition requires measured real-data evidence')
    disposition = rationale.get('disposition')
    result = {'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
              'reference_gap_is_upper_bound':False,'policy_gain':None,'selection_inputs':development_results,
              'rationale':rationale,'b0':None,'wp04':'inconclusive'}
    if disposition:
        if disposition not in ('resource-limited','no-headroom','unsuitable-proxy') or not rationale.get('reason'):
            raise ValueError('Explicit supported disposition reason required')
        if any(not r.get('cost') or not r.get('source_revision') or not r.get('data_sha256') for r in development_results):
            raise ValueError('Measured disposition needs source/data/cost evidence')
    else:
        for row in development_results: _validate_recipe_metadata(row['recipe'])
        contenders = [r for r in development_results if r['recipe']['quantizer'] != 'fp32']
        if not contenders: raise ValueError('Quantized contender pool required; FP32 is a reference only')
        if any(not r.get('settled') or not r.get('lr_bracketed') or r.get('edge') or
               r.get('missing_mechanism') or r.get('model') != 'core' or not r.get('full_budget')
               for r in contenders):
            raise ValueError('Unsettled or missing full-budget core competitor evidence')
        if len({r['seed'] for r in contenders}) < 2:
            raise ValueError('Fresh development seeds required for B0')
        if not rationale.get('equal_initial_compute') or not rationale.get('parity_evidence'):
            raise ValueError('Equal search information/compute and optimizer parity required')
        pool = {}
        for row in contenders:
            key = hashlib.sha256(canonical_bytes(row['recipe'])).hexdigest()
            pool.setdefault(key, []).append(row)
        required_seeds = set(rationale.get('fresh_seeds', []))
        if len(required_seeds) < 2 or any({r['seed'] for r in rows} != required_seeds for rows in pool.values()):
            raise ValueError('Full finalist pool requires identical fresh seeds')
        if len({r['budget'] for r in contenders}) != 1:
            raise ValueError('Finalist pool needs matched full budgets')
        means = {key: (sum(r['curve'][-1]['development_bpb'] for r in rows)/len(rows),
                       sum(r['cost']['complete_seconds'] for r in rows)) for key, rows in pool.items()}
        best = min(means, key=lambda key: means[key])
        winner = rationale['winner']; selected = [r for r in development_results if r['trial_id'] == winner]
        if len(selected) != 1: raise ValueError('Unique evidenced winner required')
        chosen = selected[0]
        if chosen['recipe']['quantizer'] == 'fp32':
            raise ValueError('FP32 is a reference anchor, not an assisted quantized B0')
        if hashlib.sha256(canonical_bytes(chosen['recipe'])).hexdigest() != best:
            raise ValueError('Requested winner is not minimum mean development BPB/cost')
        result['family_means_and_costs'] = {k:list(v) for k,v in means.items()}
        required = {'method','quantizer','grouping','rotation','backward','schedule','learning_rate','normalization','moment_coordinates'}
        if not required <= chosen['recipe'].keys(): raise ValueError('Incomplete exact B0 recipe')
        result.update(b0={'recipe':chosen['recipe'],'budget':chosen['budget'], 'evidence':chosen},wp04='pass')
    return _write(rationale['output'],result)


def create_trial_state(config):
    """Common current-source construction; all arms use the shared OrderedEmbedding."""
    kind = config.get('parent_kind', 'warmstart')
    if kind not in ('warmstart', 'derived_model_only'):
        raise ValueError('Unknown parent_kind')
    if kind == 'derived_model_only' and not config.get('parent'):
        raise ValueError('derived_model_only parent_kind requires a parent')
    from .config import resolve_config
    from .calibrate import _create, import_warmstart
    from .baselines.registry import build_matrix_method
    recipe = config['recipe']
    if recipe['method'] == 'adamw' and recipe.get('method_config', {}) != {}:
        raise ValueError('AdamW method_config unsupported; declare betas/epsilon in resolved.optimizer_probe')
    _validate_recipe_metadata(recipe)
    payload = json.loads(canonical_bytes(config['resolved']))
    gates = payload.pop('evidence_gates'); payload.pop('purpose')
    resolved = resolve_config(payload, {'gates':gates}, purpose='calibration')
    state = _create(resolved, config['partition'], config['budget'], config['seed'])
    if recipe['quantizer'] == 'fp32':
        state.model._recipe = None
        state.model.clear_quantized_views()
    else:
        state.model.configure_quantization({'name':recipe['quantizer'],'rotation':recipe['rotation'],'backward':recipe['backward']})
    if recipe['method'] != 'adamw':
        state.method = build_matrix_method(recipe['method'],dict(state.model.named_parameters()),state.method.groups,recipe.get('method_config',{}))
        state.optimizer_state = state.method.optimizer_states
    state.microbatch=config['packing']; state.clip_norm=recipe.get('clip_norm')
    if config.get('parent'):
        importer = import_warmstart
        if kind == 'derived_model_only':
            from .warmstart import import_derived_warmstart
            importer = import_derived_warmstart
        state = importer(ArtifactRef(**config['parent']),state,partition=config['partition'],continuation_targets=config['budget'])
    state.configuration.update(search_trial=config, artifact_role='warmstart' if config['role']=='warmstart' else 'continuation')
    return state


def run_search_trial(trial: TrialSpec) -> ArtifactRef:
    """Run a prospectively admitted finite development envelope; failures retained."""
    from .calibrate import _Sampler, _limits, memory_report, _offsets, _event, _timed
    from .data.cursor import BatchCursor
    from .data.batches import next_logical_batch
    from .evaluate import score_blocks
    from .optim.base import SchedulePoint
    from .state import capture_snapshot, save_snapshot
    from .train import select_logical_batch, train_step
    config = json.loads((Path('artifacts/search/configs')/(trial.config_hash+'.json')).read_bytes())
    if hashlib.sha256(canonical_bytes(config)).hexdigest() != trial.config_hash:
        raise ValueError('Config hash mismatch')
    if (config['family'],config['seed'],config['budget'],config['source_revision']) != (trial.family,trial.seed,trial.budget,trial.source_revision):
        raise ValueError('Trial/config mismatch')
    _development_only(config)
    if config['source_identity'] != source_identity(): raise ValueError('Producer source changed after prospective freeze')
    if config['resolved'].get('synthetic') is not False:
        raise ValueError('Real-data headroom cannot consume synthetic payloads')
    if config['purpose'] != 'development-bootstrap' or not config.get('finite_forecast_seconds'):
        raise ValueError('Explicit finite measured-development bootstrap purpose required')
    gate = _read(ArtifactRef(**config['admission']))
    if gate['status'] != 'pass' or gate['source_identity'] != source_identity(): raise ValueError('Current source numerical admission required')
    scope = [config['resolved']['profile'], config['recipe']['method'], config['recipe']['quantizer'], config['recipe']['rotation'], config['recipe']['backward'], config['packing']]
    if scope not in gate['scopes']: raise ValueError('Numerical admission does not cover this actual method/model/packing')
    if config['budget']//8192 > gate['maximum_steps']: raise ValueError('Horizon exceeds prospective admission')
    output=Path(config['output']); output.mkdir(parents=True,exist_ok=False)
    _write(output/'trial.json',{'trial':asdict(trial),'config':config})
    started=time.perf_counter(); state=None; curve=[]; checkpoint=None; status='failed'; failure=None
    query_attempts=[]; unknown_partial_query_work=False; unknown_partial_training_work=False
    try:
        state=create_trial_state(config)
        state.configuration['trial_id']=trial.config_hash
        manifest=config['resolved']['data']['manifest']
        query_cursor=BatchCursor.from_manifest(Path(manifest['path']),manifest['sha256'],partition='development',seed=config['query_seed'],run_target_budget=config['query_bytes'])
        query=next_logical_batch(query_cursor,config['query_bytes'])
        _write(output/'query-inputs.json',{'partition':'development','seed':config['query_seed'],'offsets':_offsets(query)})
        with _Sampler(config['resolved']['runtime']['device'],_limits(config['resolved'])) as sampler:
            def cap():
                sampler.sample()
                if sampler.exceeded: raise RuntimeError('Measured memory cap exceeded')
                if time.perf_counter()-started > config['finite_forecast_seconds']:
                    raise RuntimeError('Frozen complete development envelope exhausted')
            def observe(step):
                nonlocal unknown_partial_query_work
                cap(); t=time.perf_counter()
                completed={'completed_targets':0,'forward_context':0}; stopped_at_boundary=False
                try:
                    with (output/f'query-{step}-progress.jsonl').open('x') as progress_log:
                        def progress(update):
                            nonlocal stopped_at_boundary
                            completed.update(update)
                            progress_log.write(canonical_bytes({'exposure_id':f'{trial.config_hash}:query:{step}',
                                **update,'elapsed_seconds':time.perf_counter()-t}).decode()+'\n');progress_log.flush()
                            stopped_at_boundary=True
                            cap()
                            stopped_at_boundary=False
                        score=score_blocks(state.model,query.block_slices,
                            evaluation_context={'partition':'development','seed':config['query_seed']},progress_callback=progress)
                except Exception:
                    unknown_partial_query_work |= not stopped_at_boundary
                    seconds=time.perf_counter()-t
                    known=completed['completed_targets']
                    _event(state,'failed_development_query',seconds,query=known,additional_loss=known,
                           loss_targets=known,forward_context=completed['forward_context'])
                    query_attempts.append({'exposure_id':f'{trial.config_hash}:query:{step}',
                        'step':step,'status':'failed','attempted_max_targets':query.target_count,
                        'actual_completed_targets':known if stopped_at_boundary else None,
                        'known_completed_targets':known,'unknown_partial_work':not stopped_at_boundary,'seconds':seconds})
                    raise
                else:
                    _event(state,'development_query',time.perf_counter()-t,query=query.target_count,
                           additional_loss=query.target_count,loss_targets=query.target_count,forward_context=sum(len(b.inputs) for b in query.block_slices))
                    query_attempts.append({'exposure_id':f'{trial.config_hash}:query:{step}',
                        'step':step,'status':'completed','actual_completed_targets':score.target_count,
                        'unknown_partial_work':False})
                row={'step':step,'budget':step*8192,'development_bpb':score.bpb,'per_article':score.per_article}
                curve.append(row); _write(output/f'curve-{step}.json',row)
            observe(0)
            total=trial.budget//8192
            if total*8192 != trial.budget: raise ValueError('Real logical budget must be divisible by8192')
            with (output/'steps.jsonl').open('x') as raw:
                for step in range(total):
                    cap(); progress=step/max(1,total-1); recipe=config['recipe']
                    multiplier=1. if recipe['schedule']=='constant' else .5*(1+math.cos(math.pi*progress)) if recipe['schedule']=='cosine' else min(1.,(step+1)/max(1,total//10))*min(1.,(total-step)/max(1,total//5))
                    state.schedule['point']=SchedulePoint(recipe['learning_rate']*multiplier,recipe.get('decay',0.),progress)
                    batch=select_logical_batch(state,8192)
                    inputs=_offsets(batch)
                    event_start=len(state.ledger.events); attempted_status='completed'
                    try:
                        train_step(state,batch)
                    except Exception:
                        attempted_status='failed'; unknown_partial_training_work=True
                        raise
                    finally:
                        raw.write(canonical_bytes({'exposure_id':f'{trial.config_hash}:step:{step}',
                            'step':step,'inputs':inputs,'events':state.ledger.events[event_start:],
                            'attempt_status':attempted_status,'unknown_partial_work':attempted_status=='failed'}).decode()+'\n');raw.flush()
                    cap()
                    if step+1 in config['milestones']: observe(step+1)
                    # Completed last two plus prospectively declared milestones.
                    if step+1 in config['milestones'] or step+1>=total-1:
                        snap,seconds=_timed(state.model,lambda:capture_snapshot(state));_event(state,'snapshot_capture',seconds)
                        checkpoint,seconds=_timed(state.model,lambda:save_snapshot(snap,output/f'checkpoint-{step+1}'))
                        _event(state,'checkpoint_write',seconds)
        status='completed'
    except Exception as error:
        failure=f'{type(error).__name__}: {error}'
    memory=memory_report(sampler.samples) if 'sampler' in locals() else None
    values=[v['bpb'] for v in curve[-1]['per_article'].values()] if curve else []
    center=curve[-1]['development_bpb'] if curve else None
    radius=max([abs(v-center) for v in values],default=0.) if values else None
    report={'schema_version':1,'trial_id':trial.config_hash,'family':trial.family,'seed':trial.seed,'budget':trial.budget,
            'source_revision':trial.source_revision,'source_identity':config['source_identity'],'data_sha256':config['resolved']['data']['manifest']['sha256'],
            'model':config['resolved']['profile'],'partition':config['partition'],'status':status,'failure':failure,
            'query_attempts':query_attempts,'unknown_partial_query_work':unknown_partial_query_work,
            'unknown_partial_training_work':unknown_partial_training_work,
            'failed_count_scope':'known recorded completed work is a lower bound when unknown partial work is flagged; attempted query maxima are not actual completed counts',
            'recipe':config['recipe'],'evidence_kind':'real_data','curve':curve,'full_budget':False,
            'development_interval':[max(0.,center-radius),center+radius] if center is not None else None,
            'uncertainty':'observed article spread, not a confidence interval; single seed is unresolved',
            'settled':False,'lr_bracketed':False,'edge':True,'missing_mechanism':False,'eliminates_headroom':False,
            'cost':{'complete_seconds':time.perf_counter()-started,'ledger':None if state is None else {'stream':state.ledger.stream,'passes':state.ledger.passes,'elapsed':state.ledger.elapsed,'events':state.ledger.events},
                    'parent':config.get('parent'),'common_unknown_engineering_seconds':None},
            'memory':memory,'checkpoint':None if checkpoint is None else _ref(checkpoint),'configuration':config}
    if config['role']=='warmstart' and status=='completed':
        _write(output/'warmstart.json',warmstart_report(checkpoint,config['partition'],report['cost']))
    return _write(output/'result.json',report)


def warmstart_report(checkpoint, partition, cost):
    """H08 consumer schema, preserving complete child/parent cost annotations."""
    ledger = cost['ledger']
    if set(ledger) != {'stream','passes','elapsed','events'}:
        raise ValueError('Warmstart report requires the complete H08 ledger')
    return {'checkpoint':_ref(checkpoint),'source_partition':partition,
            'cost':{**cost,**ledger},'project_seconds':cost['complete_seconds']}


def run_headroom(path):
    path = Path(path)
    envelope=json.loads(path.read_bytes())
    if envelope.get('profile') in ('proxy','core') and 'suite' not in envelope:
        sidecar = path.with_suffix('.headroom.json')
        if not sidecar.is_file():
            raise ValueError('headroom requires an explicit frozen template sidecar')
        envelope = json.loads(sidecar.read_bytes())
        if envelope.get('template_sha256') != sha256_file(path):
            raise ValueError('Frozen headroom template hash mismatch')
    if envelope.get('suite')!='headroom' or envelope.get('purpose')!='development-bootstrap':
        raise ValueError('headroom requires explicit frozen development-bootstrap envelope; no final-gate fallback')
    refs=[run_search_trial(TrialSpec(**trial)) for trial in envelope['trials']]
    return {'trials':[_ref(ref) for ref in refs],'screen':screen_headroom(refs)}
