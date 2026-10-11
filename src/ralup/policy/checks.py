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
