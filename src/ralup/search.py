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
    attribution_contract: str | None = None

    def __post_init__(self):
        ArtifactRef('config', self.config_hash)
        if (not self.family or not self.source_revision or type(self.seed) is not int
                or self.seed < 0 or type(self.budget) is not int or self.budget <= 0):
            raise ValueError('Trial requires family, source, seed and finite positive budget')
        if self.attribution_contract is not None:
            value = json.loads(self.attribution_contract)
            content = canonical_bytes(value)
            if content.decode() != self.attribution_contract or hashlib.sha256(content).hexdigest() != self.config_hash:
                raise ValueError('Attribution trial must bind its exact canonical contract')
            if (value['seed'], value['budget'], value['variant'], value['source_revision']) != (self.seed, self.budget, self.family, self.source_revision):
                raise ValueError('Attribution trial identity mismatch')


    @property
    def pairing_contract(self):
        if self.attribution_contract is None: return None
        return json.loads(self.attribution_contract)['pairing']

    @property
    def stream_hash(self):
        return None if self.pairing_contract is None else self.pairing_contract['stream']

    @property
    def feature_cost_retained(self):
        return self.attribution_contract is not None and json.loads(self.attribution_contract)['feature_cost_retained']

    @property
    def is_privileged_diagnostic(self):
        return self.attribution_contract is not None and json.loads(self.attribution_contract)['is_privileged_diagnostic']

    @property
    def claim_status(self):
        return None if self.attribution_contract is None else json.loads(self.attribution_contract)['claim_status']

    @property
    def comparison_kind(self):
        return None if self.attribution_contract is None else json.loads(self.attribution_contract)['comparison_kind']

    @property
    def retraining_status(self):
        return None if self.attribution_contract is None else json.loads(self.attribution_contract)['retraining_status']

    @property
    def retuning_status(self):
        return None if self.attribution_contract is None else json.loads(self.attribution_contract)['retuning_status']


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
    if 'schedule' in recipe and recipe['schedule'] not in ('constant','cosine','warmup_stable_decay'):
        raise ValueError('Unsupported schedule name')
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


def operation_contract(config):
    """Exact execution settings; profile labels and absent defaults are not proof."""
    try:
        resolved=config['resolved']; data=resolved['data']
        return {'model':resolved['model'],'runtime':resolved['runtime'],
                'optimizer_probe':resolved['optimizer_probe'],'packing':config['packing'],
                'data_geometry':{k:data[k] for k in ('context_length','logical_batch_bytes','target_block_bytes','vocabulary_size','boundary_token')},
                'recipe':config['recipe']}
    except KeyError as error:
        raise ValueError('Incomplete actual operation admission contract') from error


def _validate_admission(config, gate):
    if gate.get('status')!='pass' or gate.get('source_identity')!=config['source_identity']:
        raise ValueError('Source numerical admission mismatch')
    actual=operation_contract(config)
    if config['budget']%8192: raise ValueError('Operation budget must be divisible by8192')
    for scope in gate.get('operation_scopes',[]):
        if config['budget']//8192>scope['maximum_steps']: continue
        proof=_read(ArtifactRef(**scope['evidence']))
        if (proof.get('status')!='pass' or proof.get('scope')!=scope['contract']
                or proof.get('source_identity')!=config['source_identity']
                or scope['maximum_steps']>proof.get('maximum_steps',0)): continue
        if actual==scope['contract']: return
        # Deliberate scalar rate stress is a separately authenticated analysis,
        # never a blanket waiver for changed model/refresh/schedule/coordinates.
        if config.get('operation_applicability'):
            delta=_read(ArtifactRef(**config['operation_applicability']))
            if (delta.get('schema')!='ralup.scalar_lr_applicability.v1' or delta.get('status')!='pass'
                    or delta.get('source_identity')!=config['source_identity']
                    or delta.get('admitted_contract')!=scope['contract'] or delta.get('requested_contract')!=actual
                    or delta.get('numerical_evidence')!=scope['evidence'] or not delta.get('risk_rationale')): continue
            old=json.loads(canonical_bytes(scope['contract']));new=json.loads(canonical_bytes(actual))
            for recipe in (old['recipe'],new['recipe']):
                rate=recipe.pop('learning_rate',None)
                if type(rate) not in (int,float) or not math.isfinite(rate) or rate<=0:
                    raise ValueError('Finite positive scalar LR applicability required')
                auxiliary=recipe.get('method_config',{}).pop('auxiliary_lr',None)
                if auxiliary is not None and (type(auxiliary) not in (int,float) or not math.isfinite(auxiliary) or auxiliary<=0):
                    raise ValueError('Finite positive auxiliary LR applicability required')
            if old==new: return
    raise ValueError('Numerical admission does not cover the actual operation contract/horizon')


def _final_selection_evidence(rows, rationale):
    """Validate the narrow, file-bound completed analysis before final ranking."""
    if not rationale.get('selection_analysis'):
        raise ValueError('Authenticated selection analysis required')
    analysis=_read(ArtifactRef(**rationale['selection_analysis']))
    references=[row.get('evidence') for row in rows]
    if (analysis.get('schema')!='ralup.backbone_selection_analysis.v1' or analysis.get('status')!='completed'
            or analysis.get('result_refs')!=references or any(not ref for ref in references)):
        raise ValueError('Completed selection analysis must bind every result evidence')
    matched={};protocols=set();methods=set();sources=[]
    for row in rows:
        if row.get('status')!='completed': raise ValueError('Final evidence must be completed')
        terminal=row.get('curve',[])[-1] if row.get('curve') else {}
        score=terminal.get('development_bpb');cost=row.get('cost',{}).get('complete_seconds')
        if type(score) not in (int,float) or not math.isfinite(score) or score<0:
            raise ValueError('Finite nonnegative terminal evidence score required')
        if type(cost) not in (int,float) or not math.isfinite(cost) or cost<=0:
            raise ValueError('Finite positive complete evidence cost required')
        if terminal.get('budget')!=row.get('budget') or type(row.get('budget')) is not int or row['budget']<=0:
            raise ValueError('Actual completed endpoint must equal declared budget')
        bound=_read(ArtifactRef(**row['evidence']))
        if bound!={k:v for k,v in row.items() if k!='evidence'}:
            raise ValueError('Result evidence differs from submitted row')
        required=('source_revision','source_identity','data_sha256','trial_config','query_inputs','parent_protocol','source_applicability')
        if any(not row.get(k) for k in required): raise ValueError('Missing source/data/analysis evidence binding')
        config=_read(ArtifactRef(**row['trial_config']))
        if (any(row[k]!=config.get(k) for k in ('source_revision','source_identity','recipe','seed','budget','partition'))
                or row['data_sha256']!=config['resolved']['data']['manifest']['sha256']
                or row['model']!=config['resolved']['profile'] or row['partition']!='target_train'):
            raise ValueError('Result source/config/data/partition evidence mismatch')
        _validate_admission(config,_read(ArtifactRef(**row['source_applicability'])))
        query=_read(ArtifactRef(**row['query_inputs']))
        if query.get('partition')!='development' or not query.get('offsets'):
            raise ValueError('Completed development input evidence required')
        protocol=_read(ArtifactRef(**row['parent_protocol']))
        if protocol.get('partition')!='target_train': raise ValueError('Target parent protocol required')
        if not config.get('parent'): raise ValueError('Matched parent evidence required')
        _read(ArtifactRef(**config['parent']))
        protocols.add(row['parent_protocol']['sha256'])
        comparison=(row['data_sha256'],row['query_inputs']['sha256'],config['parent']['sha256'],canonical_bytes(config['resolved']['model']))
        if row['seed'] in matched and matched[row['seed']]!=comparison:
            raise ValueError('Finalists need matched per-seed development inputs/model/parent evidence')
        matched[row['seed']]=comparison;methods.add(row['recipe']['method'])
        if row['source_identity'] not in sources:sources.append(row['source_identity'])
    if len(protocols)!=1: raise ValueError('Finalists require one matched parent protocol')
    parity=analysis.get('parity_evidence',[])
    if not parity: raise ValueError('Authenticated parity evidence required')
    covered=set()
    for reference in parity:
        proof=_read(ArtifactRef(**reference))
        if proof.get('status')!='pass' or any(source not in proof.get('source_identities',[]) for source in sources):
            raise ValueError('Parity/source applicability evidence mismatch')
        covered.update(proof.get('methods',[]))
    if not methods<=covered: raise ValueError('Parity evidence misses finalist methods')
    if not analysis.get('equal_search_evidence'): raise ValueError('Bound equal-search analysis required')
    compute=_read(ArtifactRef(**analysis['equal_search_evidence']))
    values=compute.get('family_initial_compute',{})
    if (compute.get('status')!='completed' or compute.get('result_refs')!=references
            or compute.get('equal_initial_compute') is not True or compute.get('equal_development_information') is not True
            or set(values)!={r['family'] for r in rows} or not values
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values.values())
            or len(set(values.values()))!=1):
        raise ValueError('Matched measured initial-search compute/information evidence required')


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
    return _screen_rows(rows)


def _screen_rows(rows):
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
        trial_ids=[r['trial_id'] for r in development_results]
        if len(trial_ids)!=len(set(trial_ids)): raise ValueError('Duplicate trial identity')
        pairs=[(canonical_bytes(r['recipe']),r['seed']) for r in development_results]
        if len(pairs)!=len(set(pairs)): raise ValueError('Duplicate recipe/seed observation')
        _final_selection_evidence(development_results,rationale)
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


def _build_search_method(recipe, parameters, groups):
    """Route the existing named adapters through the same H06 transaction.

    Construction is not parity, numerical admission or scientific eligibility.
    Frozen MetaQuant requires a genuine trained-theta producer that does not yet
    exist; its initialized M10 mechanism fixture must never enter this pool.
    """
    name=recipe['method'];config=recipe.get('method_config',{})
    if name=='metaquant_fp4_frozen':
        raise ValueError('Frozen MetaQuant search requires a genuine META-trained and selected theta artifact; initialized mechanics are ineligible')
    if name=='metaquant_fp4_online_target_training':
        from .baselines.metaquant import MetaQuantLMAdapter,validate_config
        from .baselines.trs import construct_backbone
        c=validate_config(config)
        if c['variant']!='online_target_training':raise ValueError('Named online MetaQuant configuration mismatch')
        method=MetaQuantLMAdapter(construct_backbone(c['backbone']),c)
    elif name=='ewgs_fp4':
        from .baselines.ewgs import EwgsAdapter
        from .optim.adamw import AdamWUpdateRule
        base=recipe['backbone_config']
        return EwgsAdapter(config).bind(parameters,groups,AdamWUpdateRule(betas=tuple(base['betas']),eps=base['eps']))
    elif name=='trs_backbone_adapter':
        from .baselines.trs import TrsBackboneAdapter,construct_backbone,validate_config
        c=validate_config(config);method=TrsBackboneAdapter(construct_backbone(c['backbone']),c)
    elif name=='learned_fp4_scale':
        from .baselines.learned_scale import LearnedScaleAdapter
        method=LearnedScaleAdapter(config)
    elif name in ('damping_fp4','simple_matrix_mask','magma_matrix_mask'):
        from .baselines.noise import NoiseAdapter
        if config['mode']!={'damping_fp4':'damping','simple_matrix_mask':'simple','magma_matrix_mask':'magma'}[name]:
            raise ValueError('Named damping/masking configuration mismatch')
        method=NoiseAdapter(config)
    elif name in ('qar_fp4_adjacent','stochastic_shadow_forward','stochastic_endpoint'):
        from .baselines.qar import QarAdjacentAdapter
        from .baselines.rounding import RoundingAdapter
        if name!='qar_fp4_adjacent' and config['mode']!=('shadow_forward' if name.endswith('forward') else 'endpoint'):
            raise ValueError('Named stochastic rounding configuration mismatch')
        method=QarAdjacentAdapter(config) if name=='qar_fp4_adjacent' else RoundingAdapter(config)
    else:
        from .baselines.registry import build_matrix_method
        return build_matrix_method(name,parameters,groups,config)
    return method.bind(parameters,groups)


def create_trial_state(config):
    """Common current-source construction; all arms use the shared OrderedEmbedding."""
    kind = config.get('parent_kind', 'warmstart')
    if kind not in ('warmstart', 'derived_model_only'):
        raise ValueError('Unknown parent_kind')
    if kind == 'derived_model_only' and not config.get('parent'):
        raise ValueError('derived_model_only parent_kind requires a parent')
    from .config import resolve_config
    from .calibrate import _create, import_warmstart
    recipe = config['recipe']
    if recipe['method'] == 'adamw' and recipe.get('method_config', {}) != {}:
        raise ValueError('AdamW method_config unsupported; declare betas/epsilon in resolved.optimizer_probe')
    _validate_recipe_metadata(recipe)
    payload = json.loads(canonical_bytes(config['resolved']))
    gates = payload.pop('evidence_gates'); payload.pop('purpose')
    resolved = resolve_config(payload, {'gates':gates}, purpose='calibration')
    state = _create(resolved, config['partition'], config['budget'], config['seed'])
    routing=recipe['method'] in ('qar_fp4_adjacent','stochastic_shadow_forward','stochastic_endpoint')
    if routing and not config.get('parent'):
        import torch
        # Fresh independent CPU draws are the existing routing adapter contract.
        # Imported parents retain their exact original named RNG owners/state.
        state.rng_streams['quantizer']=torch.Generator(device='cpu').manual_seed(state.rng_streams['quantizer'].initial_seed())
    if recipe['quantizer'] == 'fp32':
        state.model._recipe = None
        state.model.clear_quantized_views()
    else:
        state.model.configure_quantization({'name':recipe['quantizer'],'rotation':recipe['rotation'],'backward':recipe['backward']})
    if recipe['method'] != 'adamw':
        if recipe['method']=='ewgs_fp4':
            recipe={**recipe,'backbone_config':dict(state.method.rule.configuration)}
        state.method = _build_search_method(recipe,dict(state.model.named_parameters()),state.method.groups)
        state.optimizer_state = state.method.optimizer_states
    state.microbatch=config['packing']; state.clip_norm=recipe.get('clip_norm')
    if config.get('parent'):
        importer = import_warmstart
        if kind == 'derived_model_only':
            from .warmstart import import_derived_warmstart
            importer = import_derived_warmstart
        state = importer(ArtifactRef(**config['parent']),state,partition=config['partition'],continuation_targets=config['budget'])
        if routing and state.rng_streams['quantizer'].device.type!='cpu':
            raise ValueError('Routing requires an independently generated compatible CPU-quantizer-RNG parent; exact imported RNG cannot be converted or reseeded')
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
    _validate_recipe_metadata(config['recipe'])
    _validate_admission(config,gate)
    output=Path(config['output']); output.mkdir(parents=True,exist_ok=False)
    _write(output/'trial.json',{'trial':asdict(trial),'config':config})
    started=time.perf_counter();wall_start=time.time(); state=None; curve=[]; checkpoint=None; status='failed'; failure=None
    executions=[]
    from .analysis.cost import source_intervals,extra_execution_events
    corpus=config['resolved']['data']['manifest']['sha256']
    query_attempts=[]; unknown_partial_query_work=False; unknown_partial_training_work=False
    try:
        state=create_trial_state(config)
        state.configuration['trial_id']=trial.config_hash
        manifest=config['resolved']['data']['manifest']
        query_cursor=BatchCursor.from_manifest(Path(manifest['path']),manifest['sha256'],partition='development',seed=config['query_seed'],run_target_budget=config['query_bytes'])
        # Indexed document IDs carry the authenticated raw-source namespace.
        # A changed partition/manifest must not create fresh global unique bytes.
        corpus=query_cursor.state['source'].get('hashes',{}).get('source_sha256',corpus)
        query=next_logical_batch(query_cursor,config['query_bytes'])
        ordered_query=sorted(query.block_slices,key=lambda b:(b.document_id,b.target_start,b.target_end,b.scored_offsets()))
        _write(output/'query-inputs.json',{'partition':'development','seed':config['query_seed'],
            'offsets':_offsets(query),'blocks':[{k:list(v) if isinstance(v,tuple) else v for k,v in asdict(b).items()} for b in query.block_slices]})
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
                            if update['phase']=='block_completed':
                                block=ordered_query[update['completed_blocks']-1]
                                executions.append({'event_id':f'{trial.config_hash}:query:{step}:block:{update["completed_blocks"]}',
                                    'kind':'loss','targets':sum(block.score_mask),'stream_targets':0,
                                    'classifications':['query','additional_loss'],
                                    'source_intervals':source_intervals(corpus,[{'document_id':block.document_id,'offsets':list(block.scored_offsets())}])})
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
                row={'step':step,'budget':step*8192,'development_bpb':score.bpb,'per_article':score.per_article,
                    'execution_ids':[e['event_id'] for e in executions],
                    'wall_interval':{'clock':'UTC','start':wall_start,'end':wall_start+time.perf_counter()-started}}
                curve.append(row); _write(output/f'curve-{step}.json',row)
            observe(0)
            total=trial.budget//8192
            if total*8192 != trial.budget: raise ValueError('Real logical budget must be divisible by8192')
            with (output/'steps.jsonl').open('x') as raw:
                for step in range(total):
                    cap(); progress=step/max(1,total-1); recipe=config['recipe']
                    if recipe['schedule']=='constant': multiplier=1.
                    elif recipe['schedule']=='cosine': multiplier=.5*(1+math.cos(math.pi*progress))
                    elif recipe['schedule']=='warmup_stable_decay':
                        multiplier=min(1.,(step+1)/max(1,total//10))*min(1.,(total-step)/max(1,total//5))
                    else: raise ValueError('Unsupported schedule name')
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
                        for work in state.ledger.events[event_start:]:
                            for loss in work.get('loss_pass_events',[]):
                                if not loss['forward_completed']:continue
                                executions.append({'event_id':f'{trial.config_hash}:{loss["event_id"]}',
                                    'kind':'loss','targets':loss['targets'],
                                    'stream_targets':loss['targets'] if work['status']=='completed_step' else 0,
                                    'classifications':['ordinary_training']+(['backward'] if loss['backward_launched'] else []),
                                    'source_intervals':source_intervals(corpus,loss['source_offsets'])})
                            executions.extend(extra_execution_events(f'{trial.config_hash}:step:{step}',work.get('extra_pass_events',[])))
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
            'trial_config':_ref(ArtifactRef(Path('artifacts/search/configs')/(trial.config_hash+'.json'),
                sha256_file(Path('artifacts/search/configs')/(trial.config_hash+'.json')))),
            'query_inputs':_ref(ArtifactRef(output/'query-inputs.json',sha256_file(output/'query-inputs.json'))) if (output/'query-inputs.json').exists() else None,
            'parent_protocol':config.get('parent_protocol'),'source_applicability':config['admission'],
            'source_revision':trial.source_revision,'source_identity':config['source_identity'],'data_sha256':config['resolved']['data']['manifest']['sha256'],
            'model':config['resolved']['profile'],'partition':config['partition'],'status':status,'failure':failure,
            'query_attempts':query_attempts,'unknown_partial_query_work':unknown_partial_query_work,
            'executions':executions,
            'unknown_partial_training_work':unknown_partial_training_work,
            'failed_count_scope':'known recorded completed work is a lower bound when unknown partial work is flagged; attempted query maxima are not actual completed counts',
            'recipe':config['recipe'],'evidence_kind':'real_data','curve':curve,'full_budget':False,
            'development_interval':[max(0.,center-radius),center+radius] if center is not None else None,
            'uncertainty':'observed article spread, not a confidence interval; single seed is unresolved',
            'settled':False,'lr_bracketed':False,'edge':True,'missing_mechanism':False,'eliminates_headroom':False,
            'cost':{'complete_seconds':None,'known_interval_seconds':time.perf_counter()-started,
                    'wall_interval':{'clock':'UTC','start':wall_start,'end':wall_start+time.perf_counter()-started},
                    'wall_scope':{'complete':False,'boundary':'before_result_and_warmstart_publication'},
                    'ledger':None if state is None else {'stream':state.ledger.stream,'passes':state.ledger.passes,'elapsed':state.ledger.elapsed,'events':state.ledger.events},
                    'parent':config.get('parent'),'common_unknown_engineering_seconds':None},
            'memory':memory,'checkpoint':None if checkpoint is None else _ref(checkpoint),'configuration':config}
    if config['role']=='warmstart' and status=='completed':
        result_ref=_write(output/'result.json',report)
        _write(output/'warmstart.json',warmstart_report(checkpoint,config['partition'],
            {**report['cost'],'execution_record':_ref(result_ref)}))
        return result_ref
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


def allocate_search(families: list[dict], allowance_seconds: float, evidence: list[dict]) -> list[TrialSpec]:
    """Freeze a round of risk-ordered S26 trials within equal family envelopes.

    allowance_seconds is the TOTAL round allowance, divided over all families,
    including reserved blocked families. Trial counts may differ. Each supplied
    proposal has a measured finite forecast; a separate extension round needs an
    authenticated unsettled/edge/overlap/mechanism observation. No automatic grid
    or conversion from wall seconds to scientific training budgets is performed.
    """
    _development_only(families);_development_only(evidence)
    if type(allowance_seconds) not in (int,float) or not math.isfinite(allowance_seconds) or allowance_seconds<=0:
        raise ValueError('Finite positive synchronized search allowance required')
    names=[f['family'] for f in families]
    if not names or len(set(names))!=len(names):raise ValueError('Distinct search families required')
    if any(f['status'] not in ('eligible','blocked') for f in families):raise ValueError('Explicit family eligibility required')
    active=[f for f in families if f['status']=='eligible']
    if active and any(f['generic_improvements']!=active[0]['generic_improvements'] for f in active):
        raise ValueError('Compatible generic improvements must be shared across families')
    if active and any(f['development_information']!=active[0]['development_information'] for f in active):
        raise ValueError('Equal development information required')
    blocked={f['family']:f['reason'] for f in families if f['status']=='blocked'}
    if any(not reason for reason in blocked.values()):raise ValueError('Blocked family reason required')
    per_family=allowance_seconds/len(families);queues=[]
    for family in active:
        reasons=[];stage=family.get('stage','initial')
        if stage not in ('initial','extension'):raise ValueError('Known allocation stage required')
        if stage=='extension':
            ref=ArtifactRef(**family['extension_evidence']);observed=_read(ref)
            if observed not in evidence or observed['family']!=family['family']:
                raise ValueError('Extension must bind its actual family evidence')
            screen=_screen_rows(evidence);reasons=screen['reasons'][observed['trial_id']]
            if observed['trial_id'] not in screen['advance']:
                raise ValueError('Extension requires unresolved development evidence')
        if family.get('policy_family') or family['family'] in ('affine45','learned_lr49','static3'):
            access=family.get('information_contract',{})
            if access!={'features':14,'history':True,'loss_ema':True,'meta_information':'matched','own_policy_refresh':True}:
                raise ValueError('Each policy family needs full features/history/EMA and own-policy refresh')
        remaining=per_family;queue=[]
        for proposal in family['trials']:
            config=json.loads(canonical_bytes(proposal));forecast=config['finite_forecast_seconds']
            if type(forecast) not in (int,float) or not math.isfinite(forecast) or forecast<=0:
                raise ValueError('Finite positive measured trial forecast required')
            if config['family']!=family['family']:raise ValueError('Proposal family mismatch')
            if forecast>remaining:break
            config['allocation']={'stage':stage,'initial_allowance_seconds':per_family,
                'round_allowance_seconds':allowance_seconds,'extension_reasons':reasons,
                'extension_evidence':family.get('extension_evidence'),'blocked_families':blocked,
                'generic_improvements':family['generic_improvements'],
                'development_information':family['development_information'],
                'fixed_trial_count_fairness':False}
            queue.append(register_trial(config));remaining-=forecast
        queues.append(queue)
    return [queue[i] for i in range(max(map(len,queues),default=0)) for queue in queues if i<len(queue)]


def _rank_development(rows, fresh_seeds):
    """Pure fixture-testable ranking; public selection authenticates first."""
    seeds=set(fresh_seeds)
    if len(seeds)<2 or len(seeds)!=len(fresh_seeds) or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError('Distinct full-budget fresh development seeds required')
    groups={};ids=set();budgets=set()
    for row in rows:
        if row['trial_id'] in ids:raise ValueError('Duplicate trial identity')
        ids.add(row['trial_id'])
        if row['status']!='completed':raise ValueError('Failed/diverged contender cannot be silently replaced by completed runs')
        if (any(row.get(k) is not True for k in ('full_budget','fresh_seed','settled','lr_bracketed'))
                or row.get('edge') is not False or row.get('missing_mechanism') is not False or row['model']!='core'):
            raise ValueError('Unsettled/bracket-edge or missing fresh full-budget core evidence')
        if row['curve'][-1]['budget']!=row['budget']:raise ValueError('Incomplete full-budget endpoint')
        score=row['curve'][-1]['development_bpb'];seconds=row['cost']['complete_seconds']
        if (type(score) not in (int,float) or not math.isfinite(score) or score<0
                or (seconds is not None and (type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0))):
            raise ValueError('Finite development score and positive complete cost required')
        key=hashlib.sha256(canonical_bytes(row['recipe'])).hexdigest()
        groups.setdefault(key,[]).append(row);budgets.add(row['budget'])
    if not groups or len(budgets)!=1:raise ValueError('Matched full-budget contender pool required')
    alternatives=[]
    for key,observations in groups.items():
        if {r['seed'] for r in observations}!=seeds or len(observations)!=len(seeds):
            raise ValueError('Every alternative requires identical fresh seeds without duplicates')
        costs=[r['cost']['complete_seconds'] for r in observations]
        total=None if any(value is None for value in costs) else sum(costs)
        alternatives.append({'recipe_sha256':key,'recipe':observations[0]['recipe'],
            'mean_bpb':sum(r['curve'][-1]['development_bpb'] for r in observations)/len(observations),
            'mean_complete_deployment_seconds':None if total is None else total/len(observations),
            'sum_complete_deployment_seconds':total,
            'end_to_end_seconds':total,  # Retained aggregate alias; selection uses the explicit mean.
            'deployment_cost_unit':'seconds',
            'deployment_cost_boundary':'warm_creation_online_training_evaluation_checkpoint_publication',
            'excluded_costs':['offline_META','offline_tuning','observer_and_receipt_administration'],
            'seeds':sorted(seeds),'results':[r['trial_id'] for r in observations]})
    for row in alternatives:
        if (row['mean_complete_deployment_seconds'] is None
                and sum(other['mean_bpb']==row['mean_bpb'] for other in alternatives)>1):
            raise ValueError('Exact BPB tie requires known complete deployment cost')
    alternatives.sort(key=lambda r:(r['mean_bpb'],
        math.inf if r['mean_complete_deployment_seconds'] is None else r['mean_complete_deployment_seconds'],r['recipe_sha256']))
    return {'winner':alternatives[0],'alternatives':alternatives}


def _prerequisite_disposition(eligibility):
    """Read the indexed measured scientific gates, never a caller's gate label."""
    root=Path(__file__).resolve().parents[2]
    b0_index=json.loads((root/'docs/harness/m07-closing-evidence.json').read_bytes())
    meta_index=json.loads((root/'docs/harness/m11-evidence.json').read_bytes())
    b0_ref=b0_index['disposition']
    meta_ref=next(a['files']['disposition.json'] for a in meta_index['attempts'] if a['attempt']=='closure-v1')
    if eligibility.get('b0_evidence')!=b0_ref or eligibility.get('controller_evidence')!=meta_ref:
        raise ValueError('Disposition requires indexed measured B0 and controller prerequisite authority')
    b0=_read(ArtifactRef(**b0_ref));meta=_read(ArtifactRef(**meta_ref))
    if not (b0.get('wp04')=='inconclusive' and b0.get('b0') is None and
            meta.get('status')=='NEEDS_REVISION' and meta.get('selected_controller') is None):
        raise ValueError('Historical prerequisite disposition no longer describes the indexed gates')
    for ref in meta['evidence'].values():_read(ArtifactRef(**ref))
    return {'status':'inconclusive','WP04':'inconclusive','WP06':'NEEDS_REVISION',
        'B0':None,'B_star':None,'candidate':None,'learned_lr':None,'static':None,
        'prerequisite_evidence':{'B0':b0_ref,'controller':meta_ref},
        'reason':meta['reason'], 'required_advancement':meta['required_advancement']+[
            {'owner':'E01','action':'Independent strongest B* tournament: all locally competitive families, LR/betas/decay/schedule brackets, genuine META-trained frozen MetaQuant, full-budget fresh development replication and data/compute-augmented ordinary training.'}],
        'measured_non_advantage':False,'completed_tournament':False,'global_unaffordability':False}


def _lineage_producer(receipt, source, inputs, outputs, *, completed=True):
    """Bind prospective inputs and actual outputs to one completed producer."""
    startup=_read(ArtifactRef(**receipt['startup']));finish=_read(ArtifactRef(**receipt['finish']))
    if ({k:v['sha256'] for k,v in startup['source'].items() if k.startswith('src/')}!=source
            or any(ref not in startup.get('inputs',[]) for ref in inputs)):
        raise ValueError('Seed lineage producer source/prospective input mismatch')
    produced={(str((Path(receipt['finish']['path']).parent/row['path']).resolve()),row['sha256'])
              for row in finish['completed_objects']}
    if any((str(Path(ref['path']).resolve()),ref['sha256']) not in produced for ref in outputs):
        raise ValueError('Seed lineage output lacks completed producer authentication')
    start=datetime.fromisoformat(startup['utc']);end=datetime.fromisoformat(finish['completed_utc'])
    if start.tzinfo is None or end.tzinfo is None or end<start or (completed and finish['exit_code']!=0):
        raise ValueError('Seed lineage requires chronological completed producer evidence')
    return end


def _seed_parent_lineage(record, source, manifest):
    """Read actual initialization and complete typed parent ancestry; never restore.

    Seed labels, differently packaged files and distinct checkpoint hashes are
    insufficient. Origin weights, named initialization RNG and data permutation
    are checked independently of packaging, then the entire parent chain is
    bound to that origin. This is a future selection prerequisite, not an H07
    compatibility waiver or a mutation of imported state.
    """
    import random
    import torch
    from .state import load_snapshot,_validated,_Encoder,tensor_bytes
    from .data.cursor import BatchCursor
    seed=record['seed'];origin_config=_read(ArtifactRef(**record['origin_config']))
    def payload(ref):
        value=_validated(load_snapshot(ArtifactRef(**ref)))
        identities={k:v.sha256 for k,v in value['identities']['automatic'].items() if k.startswith('src/')}
        if identities!=source:raise ValueError('Seed lineage checkpoint producer source mismatch')
        return value
    def digest(value):
        encoder=_Encoder();metadata=encoder.encode(value)
        return hashlib.sha256(canonical_bytes(metadata)+tensor_bytes(encoder.tensors)).hexdigest()
    initial=payload(record['initial_checkpoint']);config=initial['configuration'];cursor=initial['cursor']
    if (type(seed) is not int or seed<0 or config!=origin_config or config.get('seed')!=seed
            or config.get('warmstart_parent') or config.get('source_partition')!='target_train'
            or config.get('artifact_role')!='warmstart' or initial['schedule']['step']!=0
            or initial['ledger']['stream']['targets']!=0 or cursor is None or cursor['cumulative_target_count']!=0):
        raise ValueError('Seed lineage requires an authenticated fresh originating initialization')
    if (cursor['source']['kind']!='indexed' or cursor['source']['sha256']!=manifest['sha256']
            or cursor['source']['partition']!='target-training'
            or config['resolved']['data']['manifest']!=manifest):
        raise ValueError('Seed lineage originating data source mismatch')
    expected=BatchCursor.from_manifest(Path(manifest['path']),manifest['sha256'],partition='target-training',
                                      seed=seed,run_target_budget=cursor['run_target_budget'])
    if expected.state!=cursor:raise ValueError('Seed lineage originating data permutation/RNG differs from its seed')
    streams=initial['rng_streams']
    if set(streams)!={'initialization','quantizer','meta','timing'}:
        raise ValueError('Complete originating named RNG lineage required')
    initialization=streams['initialization']
    if initialization['__rng_kind__']!='torch_rng' or initialization['device']!='cpu':
        raise ValueError('Origin initialization requires its actual CPU generator')
    generator=torch.Generator().set_state(initialization['state'])
    if generator.initial_seed()!=seed:raise ValueError('Origin initialization RNG seed differs from declared lineage')
    from .models.transformer import ByteTransformer
    # _create initializes on CPU before device transfer. Reproduce only that
    # finite initialization, without a forward/update or global RNG mutation.
    with torch.random.fork_rng():
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        expected_model=ByteTransformer(initial['model_config'])
        expected_initialization=torch.get_rng_state()
    if (digest(dict(expected_model.state_dict()))!=digest(initial['weights'])
            or not torch.equal(expected_initialization,initialization['state'])):
        raise ValueError('Origin weights/initialization RNG are not the declared seeded construction')
    del expected_model
    stream_seed=lambda name:int.from_bytes(hashlib.sha256(f'ralup-rng-v1:{seed}:{name}'.encode()).digest()[:8],'little')
    for name in ('meta','timing'):
        if streams[name]['__rng_kind__']!='python_rng' or streams[name]['state']!=random.Random(stream_seed(name)).getstate():
            raise ValueError('Origin named RNG does not belong to this seed lineage')
    # Compare in the recorded representation; never convert/reseed imported RNG.
    if streams['quantizer']['__rng_kind__']!='torch_rng':raise ValueError('Origin quantizer RNG lineage missing')
    expected_quantizer=torch.Generator(device=streams['quantizer']['device']).manual_seed(stream_seed('quantization'))
    if not torch.equal(expected_quantizer.get_state(),streams['quantizer']['state']):
        raise ValueError('Origin quantizer RNG does not belong to this seed lineage')
    if not record['parents']:raise ValueError('Complete warm/training parent lineage required')
    previous=None;last=None;last_checkpoint=None
    for index,stage in enumerate(record['parents']):
        parent=_read(ArtifactRef(**stage['artifact']));checkpoint=parent['checkpoint']
        current=payload(checkpoint);current_config=current['configuration']
        if index==0:
            if stage['kind']!='warmstart' or current_config!=config or current_config.get('warmstart_parent'):
                raise ValueError('Seed lineage root must be the declared warmstart producer')
            inputs=[record['origin_config']];outputs=[record['initial_checkpoint'],stage['artifact'],checkpoint]
        else:
            from .warmstart import _original,_project_cost
            if stage['kind']!='derived_model_only':raise ValueError('Later seed lineage stages require typed derived parents')
            bound,_,_,_=_original(ArtifactRef(**checkpoint))
            if set(parent)!=set(bound)|{'project_cost','creation'} or any(parent[k]!=v for k,v in bound.items()):
                raise ValueError('Seed lineage typed derived parent binding differs')
            project=parent['project_cost']
            if project!=_project_cost([ArtifactRef(**r) for r in project['completed_attempts']],
                                      [ArtifactRef(**r) for r in project['selection_evidence']],ArtifactRef(**checkpoint)):
                raise ValueError('Seed lineage derived parent costs differ')
            inherited=current_config.get('warmstart_parent',{})
            trial=_read(ArtifactRef(**stage['trial_config']))
            if (inherited.get('artifact')!=previous or inherited.get('checkpoint')!=last_checkpoint
                    or current_config.get('search_trial')!=trial or trial.get('parent')!=previous
                    or trial.get('seed')!=seed):
                raise ValueError('Seed lineage omits or relabels an imported training parent')
            inputs=[stage['trial_config'],previous];outputs=[stage['artifact'],checkpoint]
        _lineage_producer(stage['producer'],source,inputs,outputs)
        if (current_config.get('seed')!=seed or parent['source_partition']!='target_train'
                or current['cursor']['source']!=cursor['source'] or current['model_config']!=initial['model_config']
                or digest(current['rng_streams']['initialization'])!=digest(initialization)
                or current['cursor']['cumulative_target_count']<= (0 if last is None else last['cursor']['cumulative_target_count'])):
            raise ValueError('Parent initialization/data/RNG trajectory does not belong to its seed lineage')
        previous=stage['artifact'];last=current;last_checkpoint=checkpoint
    return {'parent':previous,'initial_weights':digest(initial['weights']),
            'cost_producers':{stage['artifact']['sha256']:stage['producer'] for stage in record['parents']},
            'initialization_rng':digest(initialization),'initial_data_rng':digest(cursor['rng_states']),
            'terminal_state':digest({k:last[k] for k in ('weights','cursor','rng_streams','global_rng')})}


def _replication_authority(results, eligibility):
    """Finite prospective full-budget contract, separate from bootstrap labels.

    The completed M07 analysis binds the raw results. This additional design
    binds the entire proposed inventory BEFORE work, including failed runs;
    post-run assessments can retain an uncertain contender but cannot silently
    convert an old short run into a fresh full-budget replication.
    """
    analysis=_read(ArtifactRef(**eligibility['selection_analysis']))
    if not analysis.get('replication_design') or not analysis.get('replication_completion'):
        raise ValueError('Prospective full-budget replication authority missing; bootstrap flags are ineligible')
    design_ref=analysis['replication_design'];design=_read(ArtifactRef(**design_ref))
    completion=_read(ArtifactRef(**analysis['replication_completion']))
    if (design.get('schema')!='ralup.finalist_replication_design.v1' or design.get('synthetic') is not False
            or design.get('source_identity')!=source_identity()):
        raise ValueError('Current genuine prospective replication design required')
    if completion.get('schema')!='ralup.finalist_replication_completion.v1' or completion.get('status')!='completed' or completion.get('design')!=design_ref:
        raise ValueError('Completed replication inventory authority required')
    refs=[r['evidence'] for r in results]
    if completion.get('result_refs')!=refs or len(design['trial_configs'])!=len(refs) or len(completion['producers'])!=len(refs):
        raise ValueError('Every planned replication, including failures, must remain in the inventory')
    if design['fresh_seeds']!=eligibility['fresh_seeds'] or set(design['fresh_seeds'])&set(design['tuning_seeds']):
        raise ValueError('Replication seeds must be prospective and disjoint from tuning')
    required={'AdamW','Muon','Muon+','SOAP','Madam','EWGS','TRS','QuEST','QAR',
              'stochastic_rounding','learned_scale','damping_masking','MetaQuant'}
    if set(design['family_status'])!=required:raise ValueError('Complete required family inventory must preserve unresolved competitors')
    for family,record in design['family_status'].items():
        if record['status'] not in ('included','blocked','unresolved') or not record['reason'] or not record['evidence']:
            raise ValueError('Explicit source/fidelity and applicability disposition required for every family')
        for ref in record['evidence']:_read(ArtifactRef(**ref))
    if {r['family'] for r in results}!={k for k,v in design['family_status'].items() if v['status']=='included'}:
        raise ValueError('Every included family must receive full-budget replication; unresolved families cannot be silently omitted')
    assessed=[]
    from .analysis.cost import trial_cost_record,cost_quality_frontier
    from .meta.selection import _query_count,DATA_INDEX
    accepted=json.loads(DATA_INDEX.read_bytes());corpus=_read(ArtifactRef(**accepted['manifest']))
    all_seeds=[*design['tuning_seeds'],*design['fresh_seeds']]
    if any(type(seed) is not int or seed<0 for seed in all_seeds) or len(set(all_seeds))!=len(all_seeds):
        raise ValueError('Unique disjoint tuning/replication seed inventory required')
    if (len(design.get('seed_lineages',[]))!=len(all_seeds)
            or {row['seed'] for row in design.get('seed_lineages',[])}!=set(all_seeds)):
        raise ValueError('Complete seed-specific warm/training parent lineage inventory required')
    lineages={row['seed']:_seed_parent_lineage(row,design['source_identity'],accepted['manifest']) for row in design['seed_lineages']}
    for key in ('initial_weights','initialization_rng','initial_data_rng','terminal_state'):
        if len({row[key] for row in lineages.values()})!=len(lineages):
            raise ValueError('Independent seed lineages reuse actual initialization/data/RNG or parent state')
    used_tuning=set();tuning_refs=set()
    for trial in design.get('tuning_runs',[]):
        row=_read(ArtifactRef(**trial['result']));config=_read(ArtifactRef(**trial['config']))
        identity=canonical_bytes(trial['result'])
        if identity in tuning_refs:raise ValueError('Duplicate frozen tuning result inventory')
        tuning_refs.add(identity);seed=row['seed']
        if (type(seed) is not int or seed not in lineages or config.get('seed')!=seed
                or row.get('trial_config')!=trial['config'] or row.get('source_identity')!=design['source_identity']
                or config.get('source_identity')!=design['source_identity']
                or row['trial_id']!=hashlib.sha256(canonical_bytes(config)).hexdigest()
                or config.get('parent')!=lineages[seed]['parent'] or row['cost'].get('parent')!=config.get('parent')):
            raise ValueError('Retained tuning result/config/actual parent seed lineage mismatch')
        ended=_lineage_producer(trial['producer'],design['source_identity'],[trial['config']],[trial['result']],completed=False)
        if ended>datetime.fromisoformat(design['created_utc']):raise ValueError('Tuning inventory must finish before replication freeze')
        used_tuning.add(seed)
    if used_tuning!=set(design['tuning_seeds']) or used_tuning&set(design['fresh_seeds']):
        raise ValueError('Actual retained tuning seeds differ from frozen inventory or reuse replication seeds')
    assessed_tuning=set()
    for row,config_ref,receipt in zip(results,design['trial_configs'],completion['producers']):
        config=_read(ArtifactRef(**config_ref));startup=_read(ArtifactRef(**receipt['startup']));finish=_read(ArtifactRef(**receipt['finish']))
        source={k:v['sha256'] for k,v in startup['source'].items() if k.startswith('src/')}
        if source!=design['source_identity'] or design_ref not in startup.get('inputs',[]):
            raise ValueError('Replication producer must bind exact complete source and frozen design before work')
        if row['source_identity']!=source or config['source_identity']!=source or row['trial_id']!=hashlib.sha256(canonical_bytes(config)).hexdigest():
            raise ValueError('Replication result/config differ from exact producing source and trial identity')
        dates=[datetime.fromisoformat(v) for v in (design['created_utc'],startup['utc'],startup['launcher_started_utc'])]
        if any(d.tzinfo is None for d in dates) or dates[0]>dates[1] or dates[2]>dates[1]:
            raise ValueError('Replication design postdates producer startup')
        if row['trial_config']!=config_ref or config['budget']!=design['full_budget'] or config['seed'] not in design['fresh_seeds']:
            raise ValueError('Replication config differs from prospective full-budget seed inventory')
        if row.get('seed')!=config['seed'] or config.get('parent')!=lineages[config['seed']]['parent']:
            raise ValueError('Replication requires its actual seed-specific parent lineage matched across methods')
        if row['cost'].get('parent')!=config.get('parent'):
            raise ValueError('Replication cost must retain its exact model parent ancestry')
        if config['resolved']['data']['manifest']['sha256']!=accepted['manifest']['sha256']:
            raise ValueError('Replication needs the indexed accepted corpus authority')
        _query_count(row['query_inputs'],'development',corpus,accepted)
        produced={(str((Path(receipt['finish']['path']).parent/o['path']).resolve()),o['sha256']) for o in finish['completed_objects']}
        if (str(Path(row['evidence']['path']).resolve()),row['evidence']['sha256']) not in produced:
            raise ValueError('Replication result lacks authenticated completed producer receipt')
        if finish['exit_code']!=0 or row['status']!='completed':raise ValueError('Failed full-budget replication must retain substantive failure status')
        cost_run=trial_cost_record(ArtifactRef(**row['evidence']),producer=receipt,
            parent_producers=lineages[config['seed']]['cost_producers'])
        if cost_run['unknown_ancestor_cost'] or cost_run['unknown_partial_work']:
            raise ValueError('Complete physical execution and parent costs required for full-budget selection')
        totals=cost_quality_frontier([cost_run],'total')['totals']
        if totals['total_targets'] is None or totals['seconds'] is None:
            raise ValueError('Unresolved complete replication cost')
        recipe_key=hashlib.sha256(canonical_bytes(row['recipe'])).hexdigest()
        judgment=completion['assessments'][recipe_key]
        if not judgment.get('tuning_evidence'):raise ValueError('Measured bracket/schedule/generic-improvement assessment required')
        tuning=_read(ArtifactRef(**judgment['tuning_evidence']))
        if tuning.get('replication_design')!=design_ref or tuning.get('status')!='completed' or not tuning.get('result_refs'):
            raise ValueError('Tuning assessment must bind actual retained trial results and replication design')
        for ref in tuning['result_refs']:
            _read(ArtifactRef(**ref));assessed_tuning.add(canonical_bytes(ref))
        assessed.append({**row,'full_budget':True,'fresh_seed':True,
            **{k:judgment[k] for k in ('settled','lr_bracketed','edge','missing_mechanism')},
            'cost':{**row['cost'],'complete_seconds':totals['seconds']}})
    if len(assessed)!=len(results):raise ValueError('Missing completed replication producer inventory')
    if assessed_tuning!=tuning_refs:raise ValueError('Assessed tuning results differ from complete frozen tuning inventory')
    return assessed,design


def _export_finalist_configs(value, output):
    refs={};directory=Path(output).parent/'finalist-configs'
    for name in ('B_star','candidate','learned_lr','static'):
        payload=value[name];digest=hashlib.sha256(canonical_bytes(payload)+b'\n').hexdigest();path=directory/(digest+'.json')
        if path.exists():
            if sha256_file(path)!=digest:raise ValueError('Existing immutable finalist config changed')
            ref=ArtifactRef(path,digest)
        else:ref=_write(path,payload)
        refs[name]=_ref(ref)
    return refs


def select_finalists(results: list[dict], eligibility: dict) -> ArtifactRef:
    """Authenticate independent B* evidence and M11 two-stage family nominations.

    A supported prerequisite disposition deliberately publishes null finalists.
    Current M11 selection authority is exploratory, so even genuine nominated
    policies cannot silently authorize confirmation or resolve accepted B0.
    """
    _development_only(results);_development_only(eligibility)
    if eligibility.get('disposition')=='inconclusive':
        if results:raise ValueError('Prerequisite disposition cannot relabel trial rows as finalists')
        value=_prerequisite_disposition(eligibility)
        value.update(alternatives=[],defeated_methods=[],blocked_methods=eligibility.get('blocked_methods',{}),
                     unresolved_methods=eligibility.get('unresolved_methods',{}))
    else:
        if not eligibility.get('selection_analysis') or not eligibility.get('controller_stages'):
            raise ValueError('Finalist prerequisite authority required: independent B* analysis and matched controller stages')
        # Same authenticated M07 recipe/source/numerics/parent/query/parity
        # boundary; B* ranking is independent of B0's method or adapter support.
        assessed,design=_replication_authority(results,eligibility)
        _final_selection_evidence(results,eligibility)
        for row in results:
            if row.get('evidence_kind')!='real_data':raise ValueError('Real full-budget development evidence required')
            _validate_recipe_metadata(row['recipe'])
        rank=_rank_development(assessed,eligibility['fresh_seeds'])
        from .meta.selection import select_controller
        stages=eligibility['controller_stages']
        select_controller(stages['meta-development'],stages['development'])
        nominees={}
        for row in stages['development']:
            artifact=row['controller'];value_json=artifact.to_json()
            if artifact.family in nominees and nominees[artifact.family]!=value_json:
                raise ValueError('Official stage must have one nominee per policy family')
            nominees[artifact.family]=value_json
        value={'status':'selected_exploratory','B0':None,'B_star':rank['winner'],
               'candidate':nominees['affine45'],'learned_lr':nominees['learned_lr49'],'static':nominees['static3'],
               'alternatives':rank['alternatives'],'selection_inputs':results,
               'family_source_fidelity_status':design['family_status'],
               'selection_authority':eligibility['selection_analysis'],
               'fresh_development_seeds':design['fresh_seeds'],'full_budget_targets':design['full_budget'],
               'blocked_methods':{k:v for k,v in design['family_status'].items() if v['status']!='included'},'defeated_methods':[],
               'scope':'Independent measured B* and exploratory M11 nominees; accepted B0/recipe transfer remains unresolved.'}
        value['configs']=_export_finalist_configs(value,eligibility['output'])
    value.update(schema='ralup.finalist_selection.v1',created_utc=datetime.now(timezone.utc).isoformat(),
                 uses_protected_scores=False,confirmation_eligible=False,source_identity=source_identity())
    return _write(eligibility['output'],value)


def run_finalist_tournament(path) -> ArtifactRef:
    """Explicit CLI actions; a failed resolution never falls back to execution."""
    envelope=json.loads(Path(path).read_bytes());_development_only(envelope)
    action=envelope.get('action')
    if action in ('disposition','select'):
        eligibility=envelope['eligibility']
        if (action=='disposition')!=(eligibility.get('disposition')=='inconclusive'):
            raise ValueError('Explicit selection versus prerequisite disposition action required')
        if action=='select':
            from .policy.controllers import ControllerArtifact
            for rows in eligibility['controller_stages'].values():
                for row in rows:row['controller']=ControllerArtifact.from_json(row['controller'])
        results=[{**_read(ArtifactRef(**ref)),'evidence':ref} for ref in envelope['results']]
        return select_finalists(results,eligibility)
    if action not in ('allocate','search'):raise ValueError('Explicit allocate/search/select/disposition action required')
    trials=allocate_search(envelope['families'],envelope['allowance_seconds'],envelope['evidence'])
    result={'schema':'ralup.finalist_search.v1','action':action,'trials':[asdict(t) for t in trials],
        'results':[],'unrun_trials':[],'family_elapsed_seconds':{},'defeated_methods':[],
        'blocked_families':{f['family']:f['reason'] for f in envelope['families'] if f['status']=='blocked'},
        'confirmation_eligible':False,'uses_protected_scores':False}
    if action=='search':
        cost_runs=[]
        for trial in trials:
            config=json.loads((Path('artifacts/search/configs')/(trial.config_hash+'.json')).read_bytes())
            spent=result['family_elapsed_seconds'].get(trial.family,0.)
            if config['finite_forecast_seconds']>config['allocation']['initial_allowance_seconds']-spent:
                result['unrun_trials'].append({'trial':asdict(trial),'reason':'remaining family allowance below frozen forecast'})
                continue
            started=time.perf_counter();wall_start=datetime.now(timezone.utc).timestamp();cost_run=None
            try:
                ref=run_search_trial(trial);result['results'].append(_ref(ref))
                from .analysis.cost import trial_cost_record
                cost_run=trial_cost_record(ref)
            except (ValueError,OSError,RuntimeError) as error:
                # Pre-launch/admission failure is also retained. A bad arm is
                # neither silently replaced nor turned into another purpose.
                result['unrun_trials'].append({'trial':asdict(trial),'reason':f'{type(error).__name__}: {error}'})
            finally:
                elapsed=time.perf_counter()-started
                result['family_elapsed_seconds'][trial.family]=spent+elapsed
                if cost_run is None:
                    cost_run={'run_id':trial.config_hash,'family':trial.family,'status':'failed','parents':[],
                        'events':[],'development_bpb':None,'unknown_partial_work':True,'common_engineering_seconds':None}
                cost_run['wall_intervals']=[{'clock':'UTC','start':wall_start,'end':wall_start+elapsed}]
                cost_run['wall_scope']={'complete':False,'boundary':'search_call_before_enclosing_report_publication'}
                if cost_run.get('checkpoints'):
                    cost_run['checkpoints'][-1]['wall_intervals']=cost_run['wall_intervals']
                    cost_run['checkpoints'][-1]['wall_scope']=cost_run['wall_scope']
                cost_runs.append(cost_run)
        from .analysis.cost import cost_quality_frontier
        result['cost_quality_curves']={ledger:cost_quality_frontier(cost_runs,ledger) for ledger in ('stream','total','seconds','unique')}
    return _write(envelope['output'],result)
