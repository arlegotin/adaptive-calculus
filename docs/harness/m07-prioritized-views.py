"""Smaller matched-input representation pilot after the preserved Hadamard overrun."""
import copy,json,os,sys,time
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
from ralup.search import source_identity,create_trial_state
from ralup.records import sha256_file
from ralup.config import canonical_bytes
from ralup.calibrate import _write,_offsets,_timed,_Sampler,_limits,memory_report
from ralup.state import exact_cpu_state_hash
from ralup.evaluate import score_blocks
from ralup.data.cursor import BatchCursor
from ralup.data.batches import next_logical_batch

root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
designpath=root/'docs/harness/m07-prioritized-views-design-v1.json'
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
def auth(r):
    if sha256_file(r['path'])!=r['sha256']:raise ValueError('Prioritized view input hash mismatch')
def inputs(config):
    m=config['resolved']['data']['manifest'];cursor=BatchCursor.from_manifest(Path(m['path']),m['sha256'],partition='development',seed=1707,run_target_budget=8192)
    return next_logical_batch(cursor,8192)

if sys.argv[1]=='freeze':
    config=copy.deepcopy(json.loads((root/'docs/harness/m07-ordinary-geometry-design-v1.json').read_bytes())['cases'][0])
    config.update(label='strong-parent-prioritized-views',source_identity=source_identity(),budget=8192)
    config['recipe'].update(method='muon_project_fp32',learning_rate=.02,method_config={})
    views=[{'label':'fp32','recipe':None},
        {'label':'max-row-identity','recipe':{'name':'row_dynamic_max','rotation':'identity','backward':'ste'}},
        {'label':'quest-row-identity','recipe':{'name':'quest_fp4_row_reference','rotation':'identity','backward':'trust'}},
        {'label':'quest-tensor-identity','recipe':{'name':'quest_fp4_tensor_reference','rotation':'identity','backward':'trust'}},
        {'label':'max-row-hadamard','recipe':{'name':'row_dynamic_max','rotation':'full_hadamard','backward':'ste'}}]
    d={'created_utc':datetime.now(timezone.utc).isoformat(),'source_identity':source_identity(),'helper':ref(Path(__file__)),
       'supervisor':ref(root/'docs/harness/m07-supervise.py'),'prior_failed_attempt':ref(base/'query-ordinary-views/finish.json'),
       'parent':config['parent'],'config':config,'views':views,
       'query':{'partition':'development','seed':1707,'target_bytes':8192,'offsets':_offsets(inputs(config))},
       'purpose':'risk-ordered smaller-input representation/actual block-cost pilot; fresh matched FP32 and row-max baselines; preserve all signs',
       'finite_forecast_seconds':180,'forecast_bytes':64*1024**2,'new_training_steps':0,'maximum_query_targets':40960,
       'forecast_basis':'identity8192 core queries previously approximately11s including full restore; one shared model-only import; uncertain Hadamard view last under unchanged180s finite cap with block progress and external interruption',
       'precision_change':'8192 fixed development bytes rather than previous262144; no numerical score reused across different lists; no adaptive expansion based on signs',
       'parent_contract':'fresh explicit authenticated model-only import; exact original AdamW32->FP32Muon256 W/aux/cursor/RNG, declared resets; never cross-source full-state restore',
       'representation':'canonical E2M1 signed codebook +/-6, FP32 scale/aux; detached row-max/6 or independently implemented deterministic Gaussian-fit row/tensor reference; disclosed full Hadamard',
       'provenance':'independent S14 mathematical QuEST reference; upstream release/grouping parity unresolved; no restricted source copied',
       'cost_contract':'all mixed parent ancestry, historical failed query and supervision overlap charged once globally; new model-only import/query/I/O charged here; no independent seed claim',
       'limits':'static conversion cannot select B0 or eliminate a quantizer; STE/trust/clipped backward and longer continuations remain distinct; QuEST+Hadamard remains unattempted in this finite scope'}
    _write(designpath,d);print(json.dumps(ref(designpath)))
elif sys.argv[1]=='run':
    d=json.loads(designpath.read_bytes());assert d['source_identity']==source_identity()
    for key in ('helper','supervisor','prior_failed_attempt','parent'):auth(d[key])
    out=Path(os.environ['M07_ATTEMPT_DIR']);start=time.perf_counter();batch=inputs(d['config']);assert _offsets(batch)==d['query']['offsets']
    _write(out/'query-inputs.json',d['query']);rows=[];failure=None
    with _Sampler('mps',_limits(d['config']['resolved'])) as sampler:
        def cap():
            sampler.sample()
            if sampler.exceeded or time.perf_counter()-start>d['finite_forecast_seconds']:raise RuntimeError('Frozen prioritized-view envelope exhausted')
        state=create_trial_state(d['config']);cap()
        for index,view in enumerate(d['views']):
            row={'view':view,'status':'failed','actual_query_targets':None};t=time.perf_counter();progress=[]
            try:
                cap();before=exact_cpu_state_hash(state);old=[getattr(state.model,k) for k in ('_recipe','_runtime','_contexts','_context_step')]
                try:
                    state.model.configure_quantization(view['recipe'])
                    with (out/f'query-{index}-progress.jsonl').open('x') as raw:
                        def completed(event):
                            event={**event,'elapsed_seconds':time.perf_counter()-start};progress.append(event)
                            raw.write(canonical_bytes(event).decode()+'\n');raw.flush();cap()
                        score,seconds=_timed(state.model,lambda:score_blocks(state.model,batch.block_slices,
                            evaluation_context={'partition':'development','seed':1707},progress_callback=completed))
                finally:
                    for key,value in zip(('_recipe','_runtime','_contexts','_context_step'),old):setattr(state.model,key,value)
                row.update(score=asdict(score),score_seconds=seconds,actual_query_targets=score.target_count,
                    additional_loss=score.target_count,loss_targets=score.target_count,backward=0,
                    forward_context=sum(len(b.inputs) for b in batch.block_slices),before_state_sha256=before,after_state_sha256=exact_cpu_state_hash(state))
                assert row['after_state_sha256']==before;cap();row['status']='completed'
            except Exception as e:
                failure={'view':view,'error':str(e)};row['failure']=failure
                row['completed_target_lower_bound']=max((r['completed_targets'] for r in progress),default=0)
                row['attempted_target_upper_bound']=8192;row['unknown_partial_work']=True
            row['complete_seconds']=time.perf_counter()-t;rows.append(row);_write(out/f'query-{index}.json',row)
            if failure:break
    report={'status':'failed' if failure else 'completed','failure':failure,'design':ref(designpath),'source_identity':source_identity(),
        'rows':rows,'complete_seconds':time.perf_counter()-start,'new_training_steps':0,
        'actual_completed_query_targets':sum(r['actual_query_targets'] or 0 for r in rows),
        'memory':memory_report(sampler.samples),'cost_contract':d['cost_contract'],'limits':d['limits']}
    _write(out/'query-results.json',report);print(json.dumps({'report':ref(out/'query-results.json'),'status':report['status'],
        'curves':[{'label':r['view']['label'],'bpb':r.get('score',{}).get('bpb')} for r in rows]}));raise SystemExit(1 if failure else 0)
else:raise ValueError('Explicit freeze/run required')
