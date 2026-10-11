"""Finite M08 engineering evidence under the typed exploratory protocol.

This performs actual reference/target feature scans, frozen META-only fitting,
clipping and strict checkpoint replay. No selection, query or protected scoring.
"""
import gc
import json
import math
import os
import subprocess
from pathlib import Path
import time
import torch
from .features import collect_features
from .history import MatrixHistory,LossObservation
from .normalizer import fit_normalizer
from ..records import ArtifactRef,sha256_file
from ..config import canonical_bytes
from ..quantization.context import prepare_context

ROOT=Path(__file__).resolve().parents[3]


def _read(ref):
    ref=ArtifactRef(**ref) if type(ref) is dict else ref
    if sha256_file(ref.path)!=ref.sha256:raise ValueError('M08 artifact hash mismatch')
    return json.loads(ref.path.read_bytes())


def _ref(path):return {'path':str(Path(path).resolve()),'sha256':sha256_file(path)}


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(canonical_bytes(value)+b'\n');stream.flush();os.fsync(stream.fileno())


def read_exploratory_backbone(path):
    indexed=json.loads((ROOT/'docs/harness/m07-closing-evidence.json').read_bytes())['exploratory_backbone']
    if Path(path).resolve()!=Path(indexed['path']).resolve():raise ValueError('Explicit indexed typed exploratory backbone required; B0 remains null')
    d=_read(indexed)
    if (d['schema'],d['status'],d['b0'],d['selected_b0'],d['confirmation_eligible'])!=('ralup.exploratory_backbone.v1','EXPLORATORY',None,False,False):
        raise ValueError('Typed exploratory engineering-only protocol required')
    return d,indexed


def hand_fixture(device):
    u=torch.tensor([[0.,.4,1.,7.]],device=device);d=torch.tensor([[0.,.1,.2,-.1]],device=device)
    g=torch.ones_like(u);m=torch.tensor([[1.,0.,0.,0.]],device=device)
    c=prepare_context(u,{'name':'row_fixed_max','previous_scales':torch.ones(1,1,device=device)},0,None)
    h=MatrixHistory(transition_fraction=.25,reversal_fraction=.5,disagreement_fraction=.75,updates=3,reversal_valid=True,realized_valid=True)
    f=collect_features(u,d,g,m,h,c,LossObservation(2.,4.),.3)
    want=torch.tensor([math.log10(math.sqrt(12.54)),0.,math.log10(math.sqrt(.015/12.54)),.5,1.,19/70,0.,.25,.25,.5,.75,.05,.3,-2.],device=device)
    torch.testing.assert_close(f.values,want,atol=1e-6,rtol=1e-6)
    return {'status':'pass','values':f.values.cpu().tolist(),'coverage':f.coverage,'atol':1e-6,'rtol':1e-6}


def check_features(backbone_path,design_path,device,output,*,purpose):
    from ..search import source_identity,operation_contract,create_trial_state
    from ..optim.base import SchedulePoint
    from ..state import capture_snapshot,save_snapshot,load_snapshot,restore_snapshot,exact_cpu_state_hash
    from ..train import select_logical_batch,train_step
    from ..calibrate import _Sampler,_limits,memory_report,_offsets
    if purpose!='exploratory-development-bootstrap':raise ValueError('Explicit exploratory-development-bootstrap purpose required')
    backbone,backbone_ref=read_exploratory_backbone(backbone_path)
    design_ref=_ref(design_path);design=_read(design_ref)
    if design.get('schema')!='ralup.m08_feature_design.v1' or design.get('purpose')!=purpose:raise ValueError('Frozen finite M08 feature design required')
    if design['source_identity']!=source_identity() or design['backbone']!=backbone_ref:raise ValueError('M08 source/backbone changed after freeze')
    for key in ('helper','supervisor'):
        if sha256_file(design[key]['path'])!=design[key]['sha256']:raise ValueError('M08 helper changed after freeze')
    for observed in design['runtime_commands']:
        actual_runtime=subprocess.run(observed['command'],capture_output=True,text=True,check=True)
        if actual_runtime.stdout!=observed['stdout'] or actual_runtime.stderr!=observed['stderr']:raise ValueError('Runtime changed after M08 freeze')
    if device!='mps' or not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':
        raise ValueError('Actual native MPS with fallback0 required for measured M08 design')
    if design['continuation_horizon']!=16 or design['episode_steps']!=8 or design['resume_step']!=4 or design['trace_cadence']!=4:
        raise ValueError('Only the prospectively frozen M08 eight-step diagnostic scope is implemented')
    if [c['partition'] for c in design['cases']]!=['meta_train','target_train']:raise ValueError('Separate meta-reference and target parents required')
    admission=_read(design['prior_admission'])
    unchanged=design['unchanged_numerical_sources']
    if not unchanged or any(admission['source_identity'][name]!=source_identity()[name] for name in unchanged):
        raise ValueError('Prior primitive source applicability is not preserved')
    for config in design['cases']:
        if config['source_identity']!=source_identity() or config['parent']!=backbone['parents'][config['partition']]['artifact']:
            raise ValueError('Task-owned current-source typed parent required')
        if config['recipe']!=backbone['recipe'] or config['budget']!=16*8192 or config['purpose']!='development-bootstrap':
            raise ValueError('Only the exact exploratory operation recipe and full horizon are admitted')
        actual=operation_contract(config)
        candidates=[s for s in admission['operation_scopes'] if s['contract']==actual and s['maximum_steps']>=8]
        if not candidates:raise ValueError('Exact primitive operation scope is not supported')
        proof=_read(candidates[0]['evidence'])
        if proof['scope']!=actual or proof['status']!='pass' or proof['source_identity']!=admission['source_identity']:
            raise ValueError('Unauthenticated original primitive proof')
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    attempt=Path(os.environ['M08_ATTEMPT_DIR']);producer=_ref(attempt/'startup.json')
    started=time.perf_counter();report={'schema':'ralup.m08_feature_evidence.v1','status':'failed','design':design_ref,
        'backbone':backbone_ref,'source_identity':source_identity(),'device':device,'fallback':0,
        'hand_fixture':hand_fixture(device),'cases':[],'normalizer':None,'new_training_targets':0,'new_query_targets':0,
        'limits':'M08 feature/reference-normalizer/replay/overhead diagnostic only; no trained-controller distribution, proxy-to-core transfer, B0/WP04/S31 or policy-benefit claim'}
    normalizer=None
    try:
        for config in design['cases']:
            case=attempt/config['partition'];case.mkdir(exist_ok=False)
            row={'partition':config['partition'],'operation_contract':operation_contract(config),'parent':config['parent'],
                 'continuation_horizon':16,'episode_steps':8,'inherited_updates':None,'steps':[],'replay':[],'checkpoints':[],
                 'feature_scan_seconds':0.,'history_scan_seconds':0.,'clipped':0,'scalars':0,'fit_source':None}
            report['cases'].append(row);state=None;feature_rows=[];scope_start=time.perf_counter()
            with _Sampler('mps',_limits(config['resolved'])) as sampler:
                def cap():
                    sampler.sample()
                    if sampler.exceeded or time.perf_counter()-started>design['finite_forecast_seconds']:
                        raise RuntimeError('Frozen M08 time/memory envelope exhausted')
                def save(label):
                    io=time.perf_counter();ref=save_snapshot(capture_snapshot(state),case/label)
                    row['checkpoints'].append({'label':label,**_ref(ref.path),'io_seconds':time.perf_counter()-io})
                    return ref
                cap();state=create_trial_state(config)
                row['inherited_updates']=state.ledger.stream['updates'];row['inherited_targets']=state.ledger.stream['targets']
                if state.schedule['step']!=0 or state.histories or state.quantizer_state:raise AssertionError('Typed model-only resets failed')
                state.configuration['policy_features']={'enabled':True,'full_horizon':16,'trace_cadence':4}
                state.configuration['m08_design']=design_ref
                if normalizer is not None:
                    state.histories['feature_normalizer']=normalizer
                    row['fit_source']=normalizer.source_manifest
                ids={name:id(p) for name,p in state.model.named_parameters()}
                initial_cursor=state.cursor.state;row['initial_cursor']=initial_cursor
                midpoint=None
                with (case/'steps.jsonl').open('x') as raw:
                    for index in range(8):
                        cap();state.schedule['point']=SchedulePoint(.02,0.,index/16)
                        batch=select_logical_batch(state,8192);events=len(state.ledger.events);status='failed'
                        try:result=train_step(state,batch);status='completed'
                        finally:
                            current={'execution_id':f'{attempt.name}:{config["partition"]}:reference:{index}',
                                     'continuation_step':index,'inputs':_offsets(batch),'events':state.ledger.events[events:],
                                     'status':status,'unknown_partial_work':status=='failed'}
                            raw.write(canonical_bytes(current).decode()+'\n');raw.flush();os.fsync(raw.fileno())
                            report['new_training_targets']+=sum(e['stream'] for e in current['events'] if e['status']=='completed_step')
                        row['steps'].append({'step':index,'work':result.work,'feature_trace':result.telemetry['features']})
                        fw=result.work['feature_work'];row['feature_scan_seconds']+=fw['scan_seconds'];row['clipped']+=fw['clipped'];row['scalars']+=fw['scalars']
                        hw=result.work.get('history_work',{});row['history_scan_seconds']+=hw.get('resolve_seconds',0.)+hw.get('commit_seconds',0.)
                        for name in state.model._contexts:
                            values=state.histories[name].last_features.cpu().tolist()
                            if abs(values[12]-index/16)>1e-6:raise AssertionError('Local rollout or inherited-ledger progress leaked into feature13')
                            feature_rows.append({'step':index,'matrix':name,'values':values})
                        if {name:id(p) for name,p in state.model.named_parameters()}!=ids:raise AssertionError('Parameter owner changed')
                        if index==3:midpoint=save('milestone-4')
                        if index in (6,7):save('completed-'+str(index+1))
                expected=exact_cpu_state_hash(state)
                row['final_history_bytes']=sum(h.tensor_bytes for h in state.histories.values() if type(h) is MatrixHistory)
                row['complete_reference_seconds']=time.perf_counter()-scope_start
                state=restore_snapshot(load_snapshot(midpoint),'mps');replay_ids={name:id(p) for name,p in state.model.named_parameters()}
                with (case/'replay.jsonl').open('x') as raw:
                    for index in range(4,8):
                        cap();state.schedule['point']=SchedulePoint(.02,0.,index/16)
                        batch=select_logical_batch(state,8192);events=len(state.ledger.events);status='failed'
                        try:result=train_step(state,batch);status='completed'
                        finally:
                            current={'execution_id':f'{attempt.name}:{config["partition"]}:replay:{index}',
                                     'continuation_step':index,'inputs':_offsets(batch),'events':state.ledger.events[events:],
                                     'status':status,'unknown_partial_work':status=='failed'}
                            raw.write(canonical_bytes(current).decode()+'\n');raw.flush();os.fsync(raw.fileno())
                            report['new_training_targets']+=sum(e['stream'] for e in current['events'] if e['status']=='completed_step')
                        row['replay'].append({'step':index,'work':result.work})
                        if {name:id(p) for name,p in state.model.named_parameters()}!=replay_ids:raise AssertionError('Resume replaced Parameter owners during updates')
                row['resume_exact']=exact_cpu_state_hash(state)==expected
                if not row['resume_exact']:raise AssertionError('Pending history/full-owner strict replay differs')
                row['replay_state_sha256']=expected;row['memory']=memory_report(sampler.samples)
                row['complete_scope_seconds']=time.perf_counter()-scope_start
                trajectory={'schema':'ralup.feature_trajectory.v1','partition':config['partition'],'role':'reference',
                    'synthetic':False,'source_identity':source_identity(),'data_manifest':config['resolved']['data']['manifest'],
                    'producer':producer,'backbone':backbone_ref,'parent':config['parent'],'continuation_horizon':16,'rows':feature_rows}
                # Target diagnostic rows remain separate and never enter fitting.
                if config['partition']=='meta_train':
                    _write(case/'reference-trajectory.json',trajectory)
                    t=time.perf_counter();normalizer=fit_normalizer([ArtifactRef(**_ref(case/'reference-trajectory.json'))])
                    report['normalizer']={'mean':normalizer.mean.tolist(),'std':normalizer.std.tolist(),'source_manifest':normalizer.source_manifest,
                                          'fit_seconds':time.perf_counter()-t,'trajectory':_ref(case/'reference-trajectory.json'),
                                          'fit_partition':'meta_train','reference_rows':len(feature_rows),'floor':1e-6,'clip':[-5,5]}
                    _write(attempt/'normalizer.json',report['normalizer'])
                else:
                    _write(case/'target-features.json',{'partition':'target_train','normalizer':_ref(attempt/'normalizer.json'),
                        'rows':feature_rows,'clipped':row['clipped'],'scalars':row['scalars'],'clipping_rate':row['clipped']/row['scalars']})
                del state;gc.collect();torch.mps.empty_cache()
        report['status']='pass'
    except BaseException as error:
        report['failure']=f'{type(error).__name__}: {error}'
        report['complete_seconds']=time.perf_counter()-started
        _write(out,report);raise
    report['complete_seconds']=time.perf_counter()-started
    report['retention']='per-partition completed last-two steps7/8 plus declared milestone4; no old artifacts moved/deleted; replay has reconstructible raw receipts'
    report['cost_contract']='whole physical attempt encloses reference, fit, replay, setup/import/checkpoint costs; inherited288 updates retain immutable lineage and are not new work; unknown enclosing/editing costs remain unknown'
    _write(out,report)
    return ArtifactRef(out,sha256_file(out))


# H07 audit exclusions remain unchanged. Additional removals below concern only
# the initialized-policy constructor metadata and independently verified owners.
IDENTITY_PROJECTION_EXCLUSIONS={
    'process_audit':'H07 existing runtime/process audit', 'run_ledger':'H07 existing artifact path/metadata audit wrapper',
    'cursor_audit':'H07 existing cursor process audit','ledger.elapsed':'H07 existing complete audit clocks retained separately',
    'ledger.events[*].elapsed_seconds':'H07 existing complete step clocks',
    'ledger.events[*].policy_seconds':'H07 existing scan/action audit clock',
    'ledger.events[*].feature_work.scan_seconds':'H07 existing feature scan audit clock',
    'ledger.events[*].history_work.resolve_seconds':'H07 existing history audit clock',
    'ledger.events[*].history_work.commit_seconds':'H07 existing history audit clock',
    'ledger.events[*].optimizer_work.<TIMING_FIELDS>':'H07 existing optimizer clocks; every counter remains',
    'ledger.events[*].ewgs_work.seconds':'H07 existing EWGS audit clock',
    'ledger.events[*].learned_scale_work.scale_seconds':'H07 existing scale audit clock',
    'ledger.events[*].noise_work.mask_seconds':'H07 existing mask audit clock',
    'ledger.events[*].routing_rounding_work.context_seconds':'H07 existing routing audit clock',
    'ledger.events[*].routing_rounding_work.probability_seconds':'H07 existing routing audit clock',
    'method_schema':'initialized identity only: constructor tag maps to its ordinary/matrix authoritative baseline',
    'method.snapshot_schema.adapter_mutable_state':'initialized identity only: frozen-controller schema tag maps to none.v1',
    'method.controller_state':'independently require exact45 zero weights/[0,0,1] biases and authenticated deployment digest',
    'method.normalizer':'independently authenticate selected normalizer; shared histories.feature_normalizer remains exactly compared',
    'method.normalizer_ref':'independently authenticate selected file/digest',
    'method.artifact':'independently authenticated deployment provenance; no artifact in initialized engineering scope',
    'method.deployment_sha256':'independently recomputed frozen-owner identity',
    'method.last_actions':'independently require exact identity dictionary for every matrix; reject any extra/numeric difference'}


def identity_backbone_trajectory_hash(state):
    """Exact shared scientific projection; NEVER a full H07 state-hash claim."""
    import hashlib
    from ..state import _validated,capture_snapshot,_Encoder
    from safetensors.torch import save as tensor_bytes
    from ..optim.adamw import AdamWUpdateRule
    from ..optim.soap import TIMING_FIELDS
    from .adapter import PolicyAdapter,validate_deployment
    p=_validated(capture_snapshot(state))
    if type(state.method) is PolicyAdapter:
        validate_deployment(vars(state.method),state.histories,state.configuration)
        c=state.method.controller_state
        if c.family not in ('affine45','initialized') or not torch.equal(c.parameters.cpu(),torch.tensor([0.]*42+[0.,0.,1.])) or state.method.artifact is not None:
            raise ValueError('Backbone projection only admits independently verified exact initialized candidate')
        expected={'alpha':0.,'h':1.,'tau':1.,'identity_action':True,'original_W_commit':True,'assistance_bypassed':True}
        want={n:expected for n in state.method.groups[0].parameter_names} if state.schedule['step'] else {}
        if state.method.last_actions!=want:raise ValueError('Identity projection rejects changed actions')
        p['method_schema']='ordinary_adapter.v1' if type(state.method.rule) is AdamWUpdateRule else 'matrix_adapter.v1'
        for key in ('controller_state','normalizer','normalizer_ref','artifact','deployment_sha256','last_actions'):p['method'].pop(key)
        p['method']['snapshot_schema']['adapter_mutable_state']='none.v1'
    elif p['method_schema'] not in ('ordinary_adapter.v1','matrix_adapter.v1'):
        raise ValueError('Identity projection requires authoritative ordinary/matrix reference')
    for key in ('process_audit','run_ledger','cursor_audit'):p.pop(key)
    p['ledger'].pop('elapsed')
    for row in p['ledger']['events']:
        for key in ('elapsed_seconds','policy_seconds'):row.pop(key,None)
        row.get('feature_work',{}).pop('scan_seconds',None)
        for key in ('resolve_seconds','commit_seconds'):row.get('history_work',{}).pop(key,None)
        for key in TIMING_FIELDS:row.get('optimizer_work',{}).pop(key,None)
        for family,clock in (('ewgs_work','seconds'),('learned_scale_work','scale_seconds'),('noise_work','mask_seconds')):row.get(family,{}).pop(clock,None)
        for key in ('context_seconds','probability_seconds'):row.get('routing_rounding_work',{}).pop(key,None)
    encoder=_Encoder();metadata=encoder.encode(p)
    return hashlib.sha256(canonical_bytes(metadata)+tensor_bytes(encoder.tensors)).hexdigest()


def controller_composition_probe(device):
    """Small NEW rotated composition probe, not a replay of the F05 suite."""
    from ..quantization.rotation import to_storage,from_storage
    from .actions import compose_action,BOUND_RTOL,BOUND_ATOL
    from ..models.checks import TOLERANCES
    w=torch.tensor([[.001,-.31,1.7,-4.2],[0.,.4,1.,.7]],dtype=torch.float32)
    dw=torch.tensor([[.002,.13,-.7,1.9],[0.,.1,.2,-.1]],dtype=torch.float32)*.01
    rows=[]
    for action in ((.5,1.2,.5),(0.,32.,1.)):
        outputs=[];infos=[]
        for target in ('cpu',device):
            u=to_storage(w.to(target),'full_hadamard');d=to_storage(dw.to(target),'full_hadamard')
            ctx=prepare_context(u,{'name':'row_fixed_max','previous_scales':torch.ones(2,1,device=target)},0,None)
            out,info=compose_action(u,d,*action,ctx)
            outputs.append(from_storage(out,'full_hadamard').cpu());infos.append(info)
        torch.testing.assert_close(outputs[0],outputs[1],atol=TOLERANCES['rotation_atol'],rtol=TOLERANCES['rotation_rtol'])
        rows.append({'action':list(action),'roundoff_max_abs':float((outputs[0]-outputs[1]).abs().max()),
            'bound_rtol':BOUND_RTOL,'bound_atol':BOUND_ATOL,'accepted_count':infos[-1]['accepted_count'],
            'accepted_bound_excess_max':infos[-1]['accepted_bound_excess_max'],
            'observed_bound_excess_max':infos[-1]['observed_bound_excess_max'],
            'rejected_count':infos[-1]['rejected_count'],'unrepresentable_nudge_count':infos[-1]['unrepresentable_nudge_count']})
    return {'scope':'two NEW hand-sized rotated action compositions; CPU/MPS measured output contract only',
        'atol':TOLERANCES['rotation_atol'],'rtol':TOLERANCES['rotation_rtol'],'rows':rows}


def check_controllers(backbone_path,design_path,device,output,*,purpose):
    """WP05 bounded M09 engineering scope, not accepted B0 or final learning."""
    from .controllers import Affine45,LearnedLR49,Static3,count_parameters
    from .adapter import PolicyAdapter
    from .normalizer import FrozenNormalizer
    from ..search import source_identity,create_trial_state,operation_contract
    from ..optim.base import SchedulePoint
    from ..state import capture_snapshot,save_snapshot,load_snapshot,restore_snapshot,exact_cpu_state_hash
    from ..train import select_logical_batch,train_step,_equal,_rng_state,_restore_rng
    from ..calibrate import _Sampler,_limits,memory_report,_offsets
    if purpose!='exploratory-development-bootstrap':raise ValueError('Explicit exploratory purpose required')
    backbone,backbone_ref=read_exploratory_backbone(backbone_path)
    design_ref=_ref(design_path);design=_read(design_ref)
    if design.get('schema')!='ralup.m09_controller_design.v1' or design.get('purpose')!=purpose or design['source_identity']!=source_identity() or design['backbone']!=backbone_ref:
        raise ValueError('Prospectively frozen current-source M09 design required')
    for key in ('helper','supervisor'):
        if sha256_file(design[key]['path'])!=design[key]['sha256']:raise ValueError('M09 frozen helper changed')
    for observed in design['runtime_commands']:
        actual=subprocess.run(observed['command'],capture_output=True,text=True,check=True)
        if actual.stdout!=observed['stdout'] or actual.stderr!=observed['stderr']:raise ValueError('M09 runtime changed after freeze')
    if device!='mps' or not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise ValueError('Actual native MPS fallback0 required')
    if design['episode_steps']!=4 or design['resume_step']!=2 or design['continuation_horizon']!=16 or design['trace_cadence']!=1:
        raise ValueError('Only frozen four-step/two-step-resume M09 diagnostic scope admitted')
    if design['tolerances']!={'rotation_atol':8e-6,'rotation_rtol':2e-6,'assistance_bound_atol':0.,'assistance_bound_rtol':2.384185791015625e-7,'same_device_identity':'exact shared-owner projection','same_device_resume':'exact strict H07'}:raise ValueError('Frozen M09 numerical tolerances must remain fail-closed')
    selected=json.loads((ROOT/'docs/harness/m08-feature-evidence.json').read_bytes())['normalizer']
    # Index structure is intentionally explicit; a user-provided substitute is rejected.
    selected=selected['artifact'] if 'artifact' in selected else selected
    if design['normalizer']!=selected:raise ValueError('Selected M08 diagnostic reference normalizer required')
    n=_read(selected);normalizer=FrozenNormalizer(torch.tensor(n['mean']),torch.tensor(n['std']),n['source_manifest'])
    admission=_read(design['prior_admission'])
    if any(admission['source_identity'][name]!=source_identity()[name] for name in design['unchanged_numerical_sources']):raise ValueError('Prior primitive source changed')
    if [(c['partition'],c['mode']) for c in design['cases']]!=[('target_train','initialized_identity'),('meta_train','changed_rotated')]:raise ValueError('Only frozen core identity and proxy rotated scopes admitted')
    for case in design['cases']:
        cfg=case['config']
        expected_bias=[0.,0.,1.] if case['mode']=='initialized_identity' else [.5,.1823215567939546,.5]
        expected_rotation='identity' if case['mode']=='initialized_identity' else 'full_hadamard'
        if case['bias']!=expected_bias or cfg['partition']!=case['partition'] or cfg['recipe']!={**backbone['recipe'],'rotation':expected_rotation}:raise ValueError('Only the frozen named action/partition/recipe compositions admitted')
        if cfg['source_identity']!=source_identity() or cfg['parent']!=backbone['parents'][cfg['partition']]['artifact'] or cfg['purpose']!='development-bootstrap' or cfg['parent_kind']!='derived_model_only' or cfg['budget']!=16*8192:raise ValueError('Current typed model-only/reset parent required')
        inherited={**cfg,'recipe':{**cfg['recipe'],'rotation':'identity'}}
        scopes=[s for s in admission['operation_scopes'] if s['contract']==operation_contract(inherited) and s['maximum_steps']>=4]
        if not scopes:raise ValueError('No applicable original primitive operation proof')
        proof=_read(scopes[0]['evidence'])
        if proof['status']!='pass' or proof['scope']!=operation_contract(inherited) or proof['source_identity']!=admission['source_identity']:raise ValueError('Unauthenticated primitive proof')
    attempt=Path(os.environ['M09_ATTEMPT_DIR']);started=time.perf_counter()
    report={'schema':'ralup.m09_controller_evidence.v1','status':'failed','design':design_ref,'backbone':backbone_ref,
        'normalizer':selected,'source_identity':source_identity(),'device':'mps','fallback':0,
        'capacities':{'affine45':count_parameters(Affine45()),'learned_lr49':count_parameters(LearnedLR49()),'static3':count_parameters(Static3())},
        'identity_projection_exclusions':IDENTITY_PROJECTION_EXCLUSIONS,'composition':controller_composition_probe(device),
        'cases':[],'new_training_targets':0,'new_training_updates':0,'new_query_targets':0,
        'limits':'WP05 scoped exploratory engineering diagnostic only: actual core initialized identity and proxy rotated changed action, four-step continuation/two-step replay, frozen diagnostic M08 normalizer. No accepted B0/WP04/S31, learned distribution, policy benefit, universal source/model/horizon/action-range transfer or protected scoring.'}
    try:
        for case in design['cases']:
            cfg=case['config'];folder=attempt/case['mode'];folder.mkdir(exist_ok=False)
            row={'partition':case['partition'],'mode':case['mode'],'parent':cfg['parent'],'steps':[],'replay':[],'checkpoints':[],
                'continuation_horizon':16,'episode_steps':4,'recipe':cfg['recipe'],'parent_state_route':'authenticated model-only/reset with cursor/global/named RNG and inherited expenditure','inherited_updates':288,'inherited_targets':2359296}
            report['cases'].append(row)
            with _Sampler(device,_limits(cfg['resolved'])) as sampler:
                def cap():
                    sampler.sample()
                    if sampler.exceeded or time.perf_counter()-started>design['finite_forecast_seconds']:raise RuntimeError('M09 frozen time/memory envelope exhausted')
                state=create_trial_state(cfg);controller=Affine45()
                with torch.no_grad():controller.linear.bias.copy_(torch.tensor(case['bias']))
                state.method=PolicyAdapter(state.method.rule,controller,normalizer,normalizer_ref=ArtifactRef(**selected)).bind(dict(state.model.named_parameters()),state.method.groups)
                state.optimizer_state=state.method.optimizer_states
                state.configuration['policy_features']={'enabled':True,'full_horizon':16,'trace_cadence':1};state.histories['feature_normalizer']=normalizer
                base=None
                if case['mode']=='initialized_identity':
                    base=create_trial_state(cfg);base.configuration['policy_features']=dict(state.configuration['policy_features']);base.histories['feature_normalizer']=normalizer
                    initial_rng=_rng_state();hashes=[];baseline_seconds=0.;row['baseline_steps']=[]
                    for i in range(4):
                        cap();base.schedule['point']=SchedulePoint(.02,0.,i/16)
                        batch=select_logical_batch(base,8192);events=len(base.ledger.events);status='failed'
                        try:result=train_step(base,batch);status='completed'
                        finally:
                            receipt={'execution_id':f'{attempt.name}:baseline:{i}','step':i,'inputs':_offsets(batch),'events':base.ledger.events[events:],'status':status,'unknown_partial_work':status=='failed'}
                            with (folder/'baseline.jsonl').open('a') as raw:raw.write(canonical_bytes(receipt).decode()+'\n');raw.flush();os.fsync(raw.fileno())
                            report['new_training_targets']+=sum(e['stream'] for e in receipt['events'] if e['status']=='completed_step');report['new_training_updates']+=sum(e['status']=='completed_step' for e in receipt['events'])
                        baseline_seconds+=result.work['elapsed_seconds'];row['baseline_steps'].append({'step':i,'work':result.work})
                        hashes.append(identity_backbone_trajectory_hash(base))
                    base_hash=hashes[-1];base_final_rng=_rng_state();_restore_rng(initial_rng)
                    row['authoritative_baseline_projection_sha256']=base_hash
                    row['authoritative_baseline_whole_step_seconds']=baseline_seconds
                ids={n:id(p) for n,p in state.model.named_parameters()}
                with (folder/'steps.jsonl').open('x') as raw:
                    for i in range(4):
                        cap();state.schedule['point']=SchedulePoint(.02,0.,i/16)
                        batch=select_logical_batch(state,8192);events=len(state.ledger.events);status='failed'
                        try:result=train_step(state,batch);status='completed'
                        finally:
                            receipt={'execution_id':f'{attempt.name}:{case["mode"]}:step:{i}','step':i,'inputs':_offsets(batch),'events':state.ledger.events[events:],'status':status,'unknown_partial_work':status=='failed'}
                            raw.write(canonical_bytes(receipt).decode()+'\n');raw.flush();os.fsync(raw.fileno())
                            report['new_training_targets']+=sum(e['stream'] for e in receipt['events'] if e['status']=='completed_step');report['new_training_updates']+=sum(e['status']=='completed_step' for e in receipt['events'])
                        row['steps'].append({'step':i,'work':result.work,'actions':state.method.last_actions})
                        if {n:id(p) for n,p in state.model.named_parameters()}!=ids:raise AssertionError('Parameter owner changed')
                        if base is not None:
                            actual=identity_backbone_trajectory_hash(state)
                            if actual!=hashes[i]:raise AssertionError('Complete initialized backbone trajectory differs')
                            row.setdefault('identity_projection_hashes',[]).append(actual)
                        if i in (1,3):
                            t=time.perf_counter();ref=save_snapshot(capture_snapshot(state),folder/('midpoint-2' if i==1 else 'completed-4'))
                            row['checkpoints'].append({'label':'midpoint-2' if i==1 else 'completed-4',**_ref(ref.path),'io_seconds':time.perf_counter()-t})
                            if i==1:midpoint=ref
                if base is not None:
                    # Direct exact equality includes every shared numerical owner.
                    owners={'W_aux':_equal(base.model.state_dict(),state.model.state_dict()),'retained_gradients':_equal({n:p.grad for n,p in base.model.named_parameters()},{n:p.grad for n,p in state.model.named_parameters()}),
                        'base_moments':_equal(base.optimizer_state,state.optimizer_state),'proposals':_equal(base.method.proposals,state.method.proposals),
                        'quantizer':_equal(base.quantizer_state,state.quantizer_state),'cache_context':_equal({k:getattr(base.model,k) for k in ('_contexts','_context_step','_runtime','_recipe')},{k:getattr(state.model,k) for k in ('_contexts','_context_step','_runtime','_recipe')}),
                        'history_features_EMA_normalizer':_equal(base.histories,state.histories),'cursor':base.cursor.state==state.cursor.state,'schedule':_equal(base.schedule,state.schedule),'global_rng':_equal(base_final_rng,_rng_state()),
                        'named_rng':_equal({n:g.getstate() if hasattr(g,'getstate') else g.get_state() for n,g in base.rng_streams.items()},{n:g.getstate() if hasattr(g,'getstate') else g.get_state() for n,g in state.rng_streams.items()}),
                        'work_counts':_equal(base.ledger.stream,state.ledger.stream) and _equal(base.ledger.passes,state.ledger.passes) and _equal(base.ledger.coverage,state.ledger.coverage)}
                    row['direct_shared_owner_equality']=owners
                    if not all(owners.values()):raise AssertionError('Shared complete owner equality failed')
                    del base;gc.collect()
                expected=exact_cpu_state_hash(state);row['strict_final_state_sha256']=expected
                state=restore_snapshot(load_snapshot(midpoint),device);ids={n:id(p) for n,p in state.model.named_parameters()}
                for i in range(2,4):
                    cap();state.schedule['point']=SchedulePoint(.02,0.,i/16);batch=select_logical_batch(state,8192);events=len(state.ledger.events);status='failed'
                    try:result=train_step(state,batch);status='completed'
                    finally:
                        receipt={'execution_id':f'{attempt.name}:{case["mode"]}:replay:{i}','step':i,'inputs':_offsets(batch),'events':state.ledger.events[events:],'status':status,'unknown_partial_work':status=='failed'}
                        with (folder/'replay.jsonl').open('a') as raw:raw.write(canonical_bytes(receipt).decode()+'\n');raw.flush();os.fsync(raw.fileno())
                        report['new_training_targets']+=sum(e['stream'] for e in receipt['events'] if e['status']=='completed_step');report['new_training_updates']+=sum(e['status']=='completed_step' for e in receipt['events'])
                    row['replay'].append({'step':i,'work':result.work})
                    if {n:id(p) for n,p in state.model.named_parameters()}!=ids:raise AssertionError('Resumed owner replaced')
                row['strict_resume_exact']=exact_cpu_state_hash(state)==expected
                if not row['strict_resume_exact']:raise AssertionError('Complete frozen-controller strict replay differs')
                row['memory']=memory_report(sampler.samples);row['policy_frozen_digest']=state.method.deployment_sha256
                del state;gc.collect();torch.mps.empty_cache()
        report['status']='pass'
    except BaseException as error:
        report['failure']=f'{type(error).__name__}: {error}';report['complete_seconds']=time.perf_counter()-started;_write(output,report);raise
    report['complete_seconds']=time.perf_counter()-started;_write(output,report)
    return ArtifactRef(output,sha256_file(output))
