"""Prior-declared held-out queries for completed ordinary geometry scopes."""
import json,os,sys,time
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
from ralup.search import source_identity
from ralup.records import ArtifactRef,sha256_file
from ralup.config import canonical_bytes
from ralup.calibrate import _write,_offsets,_timed,_Sampler,_limits,memory_report
from ralup.state import load_snapshot,restore_snapshot,exact_cpu_state_hash
from ralup.evaluate import score_blocks
from ralup.data.cursor import BatchCursor
from ralup.data.batches import next_logical_batch

root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
designpath=root/'docs/harness/m07-ordinary-query-design-v1.json'
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
def auth(r):
    if sha256_file(r['path'])!=r['sha256']:raise ValueError('Ordinary query input hash mismatch')
def inputs(config):
    m=config['resolved']['data']['manifest'];cursor=BatchCursor.from_manifest(Path(m['path']),m['sha256'],partition='development',seed=1707,run_target_budget=8192)
    return next_logical_batch(cursor,8192)

if sys.argv[1]=='freeze':
    priorpath=root/'docs/harness/m07-ordinary-geometry-design-v1.json';prior=json.loads(priorpath.read_bytes())
    native=base/'native-ordinary-geometry';admission=json.loads((native/'admission.json').read_bytes())
    assert admission['source_identity']==source_identity()
    assert json.loads((native/'finish.json').read_bytes())['exit_code']==0
    assert admission['status']=='pass'
    points=[{'label':'shared-row-dynamic-initial','step':0,'checkpoint':admission['runs'][0]['initial']}]
    for run in admission['runs']:
        for step in (8,16):
            saved=next(r for r in run['arms'][0]['saved'] if r['step']==step)
            points.append({'label':run['config']['label'],'step':step,'checkpoint':{k:saved[k] for k in ('path','sha256')}})
    config=admission['runs'][0]['config'];batch=inputs(config);assert _offsets(batch)==prior['query']['offsets']
    d={'created_utc':datetime.now(timezone.utc).isoformat(),'source_identity':source_identity(),'helper':ref(Path(__file__)),
       'supervisor':ref(root/'docs/harness/m07-supervise.py'),'prior_geometry_design':ref(priorpath),
       'admission':ref(native/'admission.json'),'completed_attempt':ref(native/'finish.json'),
       'purpose':'complete the prior-declared same-input ordinary bootstrap held-out points; retain all ranks, no short-curve winner',
       'config':config,'points':points,'query':prior['query'],'new_training_steps':0,
       'maximum_query_targets':len(points)*8192,'finite_forecast_seconds':180,'forecast_bytes':64*1024**2,
       'forecast_basis':'nine identity-rotation8192 queries, measured prior core query ~11s including strict restore/hash;180s includes source/optimizer-state I/O allowance',
       'cost_contract':'full geometry and mixed parent ancestry once globally, plus actual new query/restoration/I/O; physical replay arms not independent seeds',
       'limits':'one seed707/8192-byte development subset/16-step bootstrap; missing LR brackets and long ranks remain unsettled; no B0 or source-release parity inference'}
    _write(designpath,d);print(json.dumps(ref(designpath)))
elif sys.argv[1]=='run':
    d=json.loads(designpath.read_bytes());assert d['source_identity']==source_identity()
    for key in ('helper','supervisor','prior_geometry_design','admission','completed_attempt'):auth(d[key])
    out=Path(os.environ['M07_ATTEMPT_DIR']);start=time.perf_counter();batch=inputs(d['config']);assert _offsets(batch)==d['query']['offsets']
    _write(out/'query-inputs.json',d['query']);rows=[];failure=None
    with _Sampler('mps',_limits(d['config']['resolved'])) as sampler:
        def cap():
            sampler.sample()
            if sampler.exceeded or time.perf_counter()-start>d['finite_forecast_seconds']:raise RuntimeError('Frozen ordinary query envelope exhausted')
        for index,point in enumerate(d['points']):
            row={'point':point,'status':'failed','actual_query_targets':None};t=time.perf_counter();progress=[]
            try:
                cap();auth(point['checkpoint']);state=restore_snapshot(load_snapshot(ArtifactRef(**point['checkpoint'])),'mps')
                before=exact_cpu_state_hash(state)
                with (out/f'query-{index}-progress.jsonl').open('x') as raw:
                    def completed(event):
                        event={**event,'elapsed_seconds':time.perf_counter()-start};progress.append(event)
                        raw.write(canonical_bytes(event).decode()+'\n');raw.flush();cap()
                    score,seconds=_timed(state.model,lambda:score_blocks(state.model,batch.block_slices,
                        evaluation_context={'partition':'development','seed':1707},progress_callback=completed))
                row.update(score=asdict(score),score_seconds=seconds,actual_query_targets=score.target_count,
                    additional_loss=score.target_count,loss_targets=score.target_count,backward=0,
                    forward_context=sum(len(b.inputs) for b in batch.block_slices),before_state_sha256=before,after_state_sha256=exact_cpu_state_hash(state))
                assert row['after_state_sha256']==before;cap();row['status']='completed'
            except Exception as e:
                failure={'point':point,'error':str(e)};row['failure']=failure
                row['completed_target_lower_bound']=max((r['completed_targets'] for r in progress),default=0)
                row['attempted_target_upper_bound']=8192;row['unknown_partial_work']=True
            row['complete_seconds']=time.perf_counter()-t;rows.append(row);_write(out/f'query-{index}.json',row)
            if failure:break
    report={'status':'failed' if failure else 'completed','failure':failure,'design':ref(designpath),'source_identity':source_identity(),
        'rows':rows,'complete_seconds':time.perf_counter()-start,'new_training_steps':0,
        'actual_completed_query_targets':sum(r['actual_query_targets'] or 0 for r in rows),
        'memory':memory_report(sampler.samples),'cost_contract':d['cost_contract'],'limits':d['limits']}
    _write(out/'query-results.json',report);print(json.dumps({'report':ref(out/'query-results.json'),'status':report['status'],
        'curves':[{'label':r['point']['label'],'step':r['point']['step'],'bpb':r.get('score',{}).get('bpb')} for r in rows]}));raise SystemExit(1 if failure else 0)
else:raise ValueError('Explicit freeze/run required')
