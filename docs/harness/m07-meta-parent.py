"""Prospective partition-separated proxy meta warm protocol and exact current-source proof."""
import copy,json,os,sys,time
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
from ralup.search import source_identity,create_trial_state,warmstart_report
from ralup.records import ArtifactRef,sha256_file
from ralup.config import canonical_bytes
from ralup.calibrate import _write,_offsets,_Sampler,_limits,memory_report,_event,_timed
from ralup.state import capture_snapshot,restore_snapshot,save_snapshot,load_snapshot,exact_cpu_state_hash
from ralup.train import train_step,select_logical_batch
from ralup.optim.base import SchedulePoint
from ralup.data.cursor import BatchCursor
from ralup.data.batches import next_logical_batch
from ralup.evaluate import score_blocks

root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
designpath=root/'docs/harness/m07-meta-parent-design-v1.json'
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
def auth(r):
    if sha256_file(r['path'])!=r['sha256']:raise ValueError('Meta design input hash mismatch')
def batch(config):
    m=config['resolved']['data']['manifest'];cursor=BatchCursor.from_manifest(Path(m['path']),m['sha256'],
        partition='meta-development',seed=1707,run_target_budget=262144)
    return next_logical_batch(cursor,262144)

if sys.argv[1]=='freeze':
    original=json.loads((root/'docs/harness/m07-design-v1.json').read_bytes())['cases'][0]
    warm=copy.deepcopy(original);warm.update(source_identity=source_identity(),partition='meta_train',role='warmstart',
        budget=32*8192,seed=707,label='proxy-meta-fp32-adamw32',parent=None)
    warm['recipe'].update(method='adamw',quantizer='fp32',learning_rate=.001,method_config={})
    continuation=copy.deepcopy(warm);continuation.update(role='continuation',budget=256*8192,label='proxy-meta-fp32-muon256')
    continuation['recipe'].update(method='muon_project_fp32',learning_rate=.02,method_config={})
    inputs=batch(warm)
    design={'created_utc':datetime.now(timezone.utc).isoformat(),'source_identity':source_identity(),'helper':ref(Path(__file__)),
        'purpose':'measured partition-separated proxy meta warm-duration protocol and necessary new proxy FP32Muon exact scope',
        'warm_config':warm,'continuation_config':continuation,'warm_steps':32,'continuation_steps':256,
        'paired_arms':2,'comparisons':[64,128,192,256],'strict_restore_step':128,
        'finite_forecast_seconds':450,'forecast_bytes':512*1024**2,'maximum_new_steps':544,
        'forecast_basis':'measured proxy20-34centiseconds/step plus authenticated capture/restore/source and declared milestone I/O allowance',
        'prior_adamw_admission':ref(base/'applicability-v3/admission-applicability-32.json'),
        'query':{'partition':'meta-development','seed':1707,'target_bytes':262144,'offsets':_offsets(inputs)},
        'query_points':[{'step':s,'view':'fp32'} for s in (64,128,192,256)]+[{'step':s,'view':'row_dynamic_max'} for s in (128,256)],
        'query_forecast_seconds':180,'query_forecast_bytes':64*1024**2,
        'retention':'warm completed31/32; arm0 declared64/128/192 plus completed255/256; arm1 strict128 plus completed255/256; raw every-step inputs/loss/cost; failures preserved',
        'scope':'all arithmetic FP32; shared OrderedEmbedding; constant LR/zero decay/clip1/packing64, original-W moments, source-stable current runtime',
        'query_contract':'six predeclared read-only views, same input list; FP32 reference and same-W row_dynamic identity/STE conversion; no target/protected scoring or outcome-dependent expansion',
        'cost_contract':'one32-step prefix and two physical256-step arms; prefix charged once globally, duplicate proof arms not independent seeds; query costs separate, inherited ledgers never summed',
        'limits':'development calibration only; no final horizon/parent/B0 selection from this scope alone'}
    _write(designpath,design);print(json.dumps(ref(designpath)))
elif sys.argv[1]=='run':
    d=json.loads(designpath.read_bytes());assert d['source_identity']==source_identity();auth(d['helper']);auth(d['prior_adamw_admission'])
    out=Path(os.environ['M07_ATTEMPT_DIR']);started=time.perf_counter();arms=[];failure=None;state=None;warmref=None
    with _Sampler('mps',_limits(d['warm_config']['resolved'])) as sampler:
        def cap():
            sampler.sample()
            if sampler.exceeded:raise RuntimeError('Frozen meta physical memory cap exceeded')
            if time.perf_counter()-started>d['finite_forecast_seconds']:raise RuntimeError('Frozen meta complete envelope exhausted')
        def save(label):
            snap,secs=_timed(state.model,lambda:capture_snapshot(state));_event(state,'snapshot_capture',secs)
            r,secs=_timed(state.model,lambda:save_snapshot(snap,out/label));_event(state,'checkpoint_write',secs)
            return ref(r.path)
        def step(raw,index,execution):
            cap();state.schedule['point']=SchedulePoint(.001 if execution=='warm' else .02,0.,index/(31 if execution=='warm' else 255))
            logical=select_logical_batch(state,8192);events=len(state.ledger.events);status='failed'
            try:train_step(state,logical);status='completed'
            finally:
                raw.write(canonical_bytes({'execution_id':f'{out.name}:{execution}:{index}', 'step':index+1,
                    'inputs':_offsets(logical),'events':state.ledger.events[events:],'attempt_status':status,
                    'unknown_partial_work':status=='failed'}).decode()+'\n');raw.flush()
            cap()
        try:
            cap();state=create_trial_state(d['warm_config']);saved=[]
            with (out/'warm.jsonl').open('x') as raw:
                for index in range(32):
                    step(raw,index,'warm')
                    if index>=30:saved.append(save(f'warm-completed-{index+1}'))
            cost={'complete_seconds':time.perf_counter()-started,'ledger':{k:getattr(state.ledger,k) for k in ('stream','passes','elapsed','events')},
                'common_unknown_engineering_seconds':None}
            warmref=_write(out/'warmstart.json',warmstart_report(ArtifactRef(**saved[-1]),'meta_train',cost))
            config=copy.deepcopy(d['continuation_config']);config['parent']=ref(warmref.path)
            state=create_trial_state(config);initial=capture_snapshot(state);expected={}
            for arm in range(2):
                cap();state=restore_snapshot(initial,'mps');armstart=time.perf_counter();saved=[]
                with (out/f'arm-{arm}.jsonl').open('x') as raw:
                    for index in range(256):
                        if arm==1 and index==128:
                            r=save('arm1-strict128');saved.append({'step':128,**r})
                            state=restore_snapshot(load_snapshot(ArtifactRef(**r)),'mps')
                        step(raw,index,f'arm{arm}')
                        if index+1 in d['comparisons']:
                            digest=exact_cpu_state_hash(state);_write(out/f'comparison-{arm}-{index+1}.json',{'step':index+1,'state_sha256':digest})
                            if arm==0:expected[index+1]=digest
                            elif expected[index+1]!=digest:raise RuntimeError(f'Exact meta owners differ at step{index+1}')
                        if (arm==0 and index+1 in d['comparisons']) or index>=254:
                            saved.append({'step':index+1,**save(f'arm{arm}-completed-{index+1}')})
                row={'arm':arm,'saved':saved,'complete_seconds':time.perf_counter()-armstart,
                    'stream':state.ledger.stream,'passes':state.ledger.passes,'elapsed':state.ledger.elapsed}
                arms.append(row);_write(out/f'arm-{arm}-completed.json',row)
        except Exception as error:
            failure={'error':str(error),'completed_step':None if state is None else state.schedule['step']}
            if state is not None:
                try:failure['retained_state']=save('failure-state')
                except Exception as e:failure['retention_error']=str(e)
    report={'status':'blocked' if failure else 'pass','failure':failure,'design':ref(designpath),'source_identity':source_identity(),
        'scopes':[] if failure else [['proxy','muon_project_fp32','fp32','identity','ste',64]],'maximum_steps':256,
        'warmstart':None if warmref is None else ref(warmref.path),'arms':arms,'memory':memory_report(sampler.samples),
        'complete_seconds':time.perf_counter()-started,'cost_contract':d['cost_contract'],'scope':d['scope'],'limits':d['limits']}
    _write(out/'admission.json',report);print(json.dumps({'report':ref(out/'admission.json'),'status':report['status']}));raise SystemExit(1 if failure else 0)
elif sys.argv[1]=='query':
    d=json.loads(designpath.read_bytes());assert d['source_identity']==source_identity();auth(d['helper'])
    native=base/'native-meta-parent';admission=json.loads((native/'admission.json').read_bytes())
    assert admission['status']=='pass' and admission['source_identity']==source_identity()
    assert json.loads((native/'finish.json').read_bytes())['exit_code']==0
    out=Path(os.environ['M07_ATTEMPT_DIR']);started=time.perf_counter();inputs=batch(d['warm_config'])
    assert _offsets(inputs)==d['query']['offsets'];_write(out/'query-inputs.json',d['query']);rows=[];failure=None
    with _Sampler('mps',_limits(d['warm_config']['resolved'])) as sampler:
        def cap():
            sampler.sample()
            if sampler.exceeded or time.perf_counter()-started>d['query_forecast_seconds']:raise RuntimeError('Frozen meta query cap exceeded')
        for point in d['query_points']:
            row={'point':point,'status':'failed','actual_query_targets':None};t=time.perf_counter()
            try:
                cap();r=next(r for r in admission['arms'][0]['saved'] if r['step']==point['step'])
                row['checkpoint']={k:r[k] for k in ('path','sha256')};state=restore_snapshot(load_snapshot(ArtifactRef(**row['checkpoint'])),'mps')
                before=exact_cpu_state_hash(state);old=[getattr(state.model,k) for k in ('_recipe','_runtime','_contexts','_context_step')]
                try:
                    if point['view']!='fp32':state.model.configure_quantization({'name':'row_dynamic_max','rotation':'identity','backward':'ste'})
                    score,seconds=_timed(state.model,lambda:score_blocks(state.model,inputs.block_slices,evaluation_context={'partition':'meta-development','seed':1707}))
                finally:
                    for key,value in zip(('_recipe','_runtime','_contexts','_context_step'),old):setattr(state.model,key,value)
                row.update(score=asdict(score),actual_query_targets=score.target_count,additional_loss=score.target_count,
                    loss_targets=score.target_count,forward_context=sum(len(b.inputs) for b in inputs.block_slices),backward=0,score_seconds=seconds,
                    before_state_sha256=before,after_state_sha256=exact_cpu_state_hash(state))
                assert row['before_state_sha256']==row['after_state_sha256'];cap();row['status']='completed'
            except Exception as e:failure={'point':point,'error':str(e)};row['failure']=failure
            row['complete_seconds']=time.perf_counter()-t;rows.append(row);_write(out/f"query-{point['step']}-{point['view']}.json",row)
            if failure:break
    report={'status':'failed' if failure else 'completed','failure':failure,'design':ref(designpath),'source_identity':source_identity(),
        'admission':ref(native/'admission.json'),'complete_ancestry':ref(native/'finish.json'),'rows':rows,
        'complete_seconds':time.perf_counter()-started,'new_training_steps':0,
        'actual_completed_query_targets':sum(r['actual_query_targets'] or 0 for r in rows),
        'unknown_partial_query_work':any(r['actual_query_targets'] is None for r in rows),'memory':memory_report(sampler.samples),
        'limits':'single seed, paired replay not independent replication, meta-development only; no final parent/horizon/B0 decision'}
    _write(out/'query-results.json',report);print(json.dumps({'report':ref(out/'query-results.json'),'status':report['status'],
        'curves':[{'point':r['point'],'bpb':r.get('score',{}).get('bpb')} for r in rows]}));raise SystemExit(1 if failure else 0)
else:raise ValueError('Explicit freeze/run/query required')
