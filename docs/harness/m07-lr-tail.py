"""Prospectively declared current-source LR duration extension without prefix replay."""
import copy,hashlib,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
from ralup.search import source_identity
from ralup.records import ArtifactRef,sha256_file
from ralup.config import canonical_bytes
from ralup.calibrate import _write,_offsets,_Sampler,_limits,memory_report,_timed,_event
from ralup.state import load_snapshot,restore_snapshot,capture_snapshot,save_snapshot,exact_cpu_state_hash
from ralup.data.cursor import BatchCursor
from ralup.data.batches import next_logical_batch
from ralup.train import train_step,select_logical_batch
from ralup.optim.base import SchedulePoint
from ralup.evaluate import score_blocks

root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
designpath=root/'docs/harness/m07-lr-tail-design-v1.json'
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
digest=lambda x:hashlib.sha256(canonical_bytes(x)).hexdigest()
def authenticate(r):
    if sha256_file(r['path'])!=r['sha256']:raise ValueError('LR-tail input hash mismatch')
def query_batch(config):
    m=config['resolved']['data']['manifest']
    c=BatchCursor.from_manifest(Path(m['path']),m['sha256'],partition='development',seed=1707,run_target_budget=8192)
    return next_logical_batch(c,8192)

if sys.argv[1]=='freeze':
    entries=[]
    for label in ('muon-bracket-initial-m0.1-32','muon-bracket-refinement-m3.16228-32','muon-bracket-initial-m10-32'):
        p=base/'science'/label;result=json.loads((p/'result.json').read_bytes())
        config=json.loads((p/'trial.json').read_bytes())['config']
        assert result['status']=='completed' and result['source_identity']==source_identity()
        assert config['recipe']['schedule']=='constant'
        entries.append({'label':label,'parent':result['checkpoint'],'result':ref(p/'result.json'),
            'config':config,'inputs':ref(p/'steps.jsonl'),'query_inputs':ref(p/'query-inputs.json')})
    batch=query_batch(entries[0]['config'])
    d={'created_utc':datetime.now(timezone.utc).isoformat(),'research_cutoff':'2026-10-03','source_identity':source_identity(),
        'purpose':'carry-state32→256 LR duration assessment; retain unsettled lower/interior/recovering upper candidates; not fresh-seed final selection',
        'entries':entries,'from_step':32,'to_step':256,'query_steps':[64,128,192,256],
        'query':{'partition':'development','seed':1707,'target_bytes':8192,'offsets':_offsets(batch)},
        'default_reuse':[ref(base/'query-admission128/query-results.json'),ref(base/'query-extension256/query-results.json')],
        'numerical_scope':{'evidence':ref(base/'native-extension256/admission.json'),
            'rationale':'same core shape,packing32,shared OrderedEmbedding,original-W Muon/moments,row-dynamic/identity/STE primitive path and256 horizon; only scalar effective matrix/auxiliary LR values differ from admitted default; no new kernels or owner types',
            'limit':'default-LR exact duplicate/resume proof is scoped; these actual tuned-LR scientific tails are finite-guarded ordinary-path applications, not a universal exact recipe replay claim'},
        'extension_contract':'strict old source/runtime complete restore before modification; change only final cursor envelope and explicit configuration annotation; preserve position/permutation/carry/RNG/moments/history/constant schedule; no optimizer reset or repeated prefix',
        'maximum_new_steps':672,'maximum_new_query_targets':12*8192,'finite_forecast_seconds':1000,'forecast_bytes':2*1024**3,
        'forecast_basis':'measured default Muon256 paired512-step tails375.792s including snapshots/checks;672 steps plus12~0.5s queries,15 endpoint/milestone snapshots and extra strict I/O bounded by1000s',
        'retention':'raw every step losses/inputs/seed/exposure IDs; snapshots64/128/192 and final255/256 per trajectory; preserve stopped complete state on failure when possible',
        'cost_contract':'full original warm32 and LR32 parents retained; sum distinct physical tail executions, not inherited ledger totals; complete query/checkpoint/restore/attempt costs; unknown common engineering remains unknown'}
    r=_write(designpath,d);print(json.dumps(ref(r.path)))
elif sys.argv[1]=='run':
    d=json.loads(designpath.read_bytes());assert d['source_identity']==source_identity()
    authenticate(d['numerical_scope']['evidence'])
    for r in d['default_reuse']:authenticate(r)
    out=Path(os.environ['M07_ATTEMPT_DIR']);started=time.perf_counter();runs=[];failure=None
    batchq=query_batch(d['entries'][0]['config']);assert _offsets(batchq)==d['query']['offsets'];_write(out/'query-inputs.json',d['query'])
    for item in d['entries']:
        case=out/item['label'];case.mkdir();run={'entry':item,'curve':[],'saved':[],'status':'failed'};begin=time.perf_counter();state=None
        try:
            for k in ('parent','result','inputs','query_inputs'):authenticate(item[k])
            assert json.loads(Path(item['query_inputs']['path']).read_bytes())['offsets']==d['query']['offsets']
            state=restore_snapshot(load_snapshot(ArtifactRef(**item['parent'])),'mps');assert state.schedule['step']==32
            old_hash=exact_cpu_state_hash(state);old=state.cursor.state;old_config=state.configuration
            assert old['run_target_budget']==old['cumulative_target_count']
            new={**old,'run_target_budget':old['cumulative_target_count']+224*8192};cursor=BatchCursor(new)
            assert {k:v for k,v in cursor.state.items() if k!='run_target_budget'}=={k:v for k,v in old.items() if k!='run_target_budget'}
            config=copy.deepcopy(old_config);config['horizon_extension']={'contract':'M07-scientific-LR32-to256-v1','from_step':32,'to_step':256,'parent':item['parent'],'old_limit':old['run_target_budget'],'new_limit':new['run_target_budget']}
            assert exact_cpu_state_hash(state)==old_hash
            state.cursor=cursor;state.configuration=config
            run['extension']={'old_complete_state_sha256':old_hash,'extended_complete_state_sha256':exact_cpu_state_hash(state),
                'old_cursor_sha256':digest(old),'new_cursor_sha256':digest(new),'old_configuration_sha256':digest(old_config),'new_configuration_sha256':digest(config),
                'old_limit':old['run_target_budget'],'new_limit':new['run_target_budget'],'inherited_stream':dict(state.ledger.stream),'inherited_passes':dict(state.ledger.passes)}
            _write(case/'extension.json',run['extension'])
            with _Sampler('mps',_limits(item['config']['resolved'])) as sampler:
                def cap():
                    sampler.sample()
                    if sampler.exceeded or time.perf_counter()-started>d['finite_forecast_seconds']:raise RuntimeError('Frozen LR-tail envelope exhausted')
                with (case/'steps.jsonl').open('x') as raw:
                    for step in range(32,256):
                        cap();recipe=item['config']['recipe'];state.schedule['point']=SchedulePoint(recipe['learning_rate'],recipe.get('decay',0.),1.)
                        batch=select_logical_batch(state,8192);inputs=_offsets(batch);event_start=len(state.ledger.events)
                        try:train_step(state,batch)
                        finally:
                            raw.write(canonical_bytes({'execution_id':f'{out.name}:{item["label"]}:{step}',
                                'step':step+1,'inputs':inputs,'events':state.ledger.events[event_start:]}).decode()+'\n');raw.flush()
                        if step+1 in d['query_steps']:
                            cap();before=exact_cpu_state_hash(state)
                            score,seconds=_timed(state.model,lambda:score_blocks(state.model,batchq.block_slices,evaluation_context={'partition':'development','seed':1707}))
                            after=exact_cpu_state_hash(state);assert before==after
                            _event(state,'development_query',seconds,query=score.target_count,additional_loss=score.target_count,loss_targets=score.target_count,forward_context=sum(len(b.inputs) for b in batchq.block_slices))
                            row={'step':step+1,'score':asdict(score),'score_seconds':seconds,'before_state_sha256':before,'after_state_sha256':after,
                                'exposure_id':f'{out.name}:{item["label"]}:query:{step+1}'}
                            run['curve'].append(row);_write(case/f'curve-{step+1}.json',row)
                        if step+1 in (64,128,192,255,256):
                            r=save_snapshot(capture_snapshot(state),case/f'checkpoint-{step+1}');run['saved'].append(ref(r.path))
                        cap()
                run['memory']=memory_report(sampler.samples)
            run['status']='completed'
        except Exception as error:
            failure={'label':item['label'],'error':str(error)};run['failure']=failure
            if state is not None:
                try:r=save_snapshot(capture_snapshot(state),case/'stopped-state');run['stopped_state']=ref(r.path)
                except Exception as save_error:run['stopped_state_failure']=str(save_error)
        finally:
            run['complete_seconds']=time.perf_counter()-begin
            if state is not None:run.update(completed_step=state.schedule['step'],stream=state.ledger.stream,passes=state.ledger.passes,elapsed=state.ledger.elapsed)
            runs.append(run);_write(case/'result.json',run)
        if failure:break
    report={'status':'failed' if failure else 'completed','failure':failure,'runs':runs,'design':ref(designpath),'source_identity':source_identity(),
        'complete_seconds':time.perf_counter()-started,'common_unknown_engineering_seconds':None}
    r=_write(out/'results.json',report);print(json.dumps({'result':ref(r.path),'status':report['status']}));raise SystemExit(1 if failure else 0)
else:raise ValueError('Explicit freeze/run mode required')
