"""Read-only same-W warm-duration conversion, with authenticated cached FP128."""
import json, os, sys, time
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
from ralup.search import source_identity
from ralup.records import ArtifactRef, sha256_file
from ralup.calibrate import _write, _offsets, _Sampler, _limits, memory_report, _timed
from ralup.data.cursor import BatchCursor
from ralup.data.batches import next_logical_batch

root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
native=base/'native-warm-extension256'
stage=sys.argv[2] if len(sys.argv)>2 else 'initial'
assert stage in ('initial','expanded')
query_bytes=8192 if stage=='initial' else 262144
designpath=root/('docs/harness/m07-conversion-design-v1.json' if stage=='initial' else 'docs/harness/m07-conversion-design-v2.json')
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
def authenticate(r):
    if sha256_file(r['path'])!=r['sha256']:raise ValueError('Conversion input hash mismatch')
def batch_for(config):
    m=config['resolved']['data']['manifest']
    c=BatchCursor.from_manifest(Path(m['path']),m['sha256'],partition='development',seed=1707,run_target_budget=query_bytes)
    return next_logical_batch(c,query_bytes)

if sys.argv[1]=='freeze':
    admission=json.loads((native/'admission.json').read_bytes())
    assert admission['status']=='pass' and admission['source_identity']==source_identity()
    run=admission['runs'][0];config=run['entry']['config']
    cached_path=base/'query-admission128/core-fp32-warm128-128.json'
    cached=json.loads(cached_path.read_bytes())
    assert cached['status']=='completed' and cached['before_state_sha256']==cached['after_state_sha256']
    assert cached['entry']['checkpoint']==run['entry']['parents'][0]
    old_inputs=base/'query-admission128/query-inputs.json';batch=batch_for(config)
    if stage=='initial':assert json.loads(old_inputs.read_bytes())['offsets']==_offsets(batch)
    entries=[]
    for step,parent,views in ((128,run['entry']['parents'][0],['row_dynamic_max'] if stage=='initial' else ['fp32','row_dynamic_max']),
                              (256,run['arms'][0]['saved'][-1],['fp32','row_dynamic_max'])):
        entries.append({'step':step,'checkpoint':parent,'views':views,'config':config,
            'expected_state_sha256':cached['before_state_sha256'] if step==128 else
                json.loads((native/'core-fp32-warm128/comparison-0-256.json').read_bytes())['state_sha256']})
    design={'created_utc':datetime.now(timezone.utc).isoformat(),'research_cutoff':'2026-10-03',
        'purpose':'same-W instantaneous row-dynamic FP4 damage at both retained core target warm durations, before optimizer continuation; no parent selection yet',
        'source_identity':source_identity(),'entries':entries,'admission':ref(native/'admission.json'),
        'completed_attempt':ref(native/'finish.json'),'cached_fp32_128':ref(cached_path),
        'cached_input_list':ref(old_inputs),'cached_query_ancestry':ref(base/'query-admission128/finish.json'),
        'query':{'partition':'development','seed':1707,'target_bytes':query_bytes,'offsets':_offsets(batch)},
        'recipe':{'name':'row_dynamic_max','rotation':'identity','backward':'ste'},
        'representation':'canonical signed E2M1 ±6 codebook; FP32 detached per-row max(abs(W))/6, zero rows one; same original W and nonquantized parameters; evaluation computes fresh scales; no optimizer or history updates',
        'owner_contract':'strict source/runtime restore, authenticate exact full-state hash; temporary evaluation recipe only; restore original recipe/runtime/context objects in finally; exact full-state hash after every view including global/named RNG and numerical owners',
        'new_query_passes':3 if stage=='initial' else 4,'new_query_targets':query_bytes*(3 if stage=='initial' else 4),'new_training_steps':0,
        'finite_forecast_seconds':180,'forecast_bytes':64*1024**2,
        'forecast_basis':'initial three8192-byte queries33.463s including restore/checks; expanded four262144-byte passes at measured0.30–0.58s per8192 plus I/O allowance',
        'cost_contract':'initial stage reuses cached FP128 only; expanded stage four fresh passes with larger predeclared list, old small-query scores never substituted; all ancestry counted once globally plus new complete query/conversion/I/O; common historical engineering remains unknown',
        'stage':stage,'expansion_reason':'small8192-byte instantaneous damage near zero or reversed; larger development coverage to assess same-W signal, no new optimization or outcome-selected subset',
        'scope':'seed707 core4x256 target warm from initialization, not core32-parent Muon continuation; meta32 separate and not qualified by these outcomes'}
    r=_write(designpath,design);print(json.dumps(ref(r.path)))
elif sys.argv[1]=='run':
    from ralup.state import load_snapshot,restore_snapshot,exact_cpu_state_hash
    from ralup.evaluate import score_blocks
    d=json.loads(designpath.read_bytes());assert d['source_identity']==source_identity()
    for key in ('admission','completed_attempt','cached_fp32_128','cached_input_list','cached_query_ancestry'):authenticate(d[key])
    output=Path(os.environ['M07_ATTEMPT_DIR']);started=time.perf_counter();rows=[];failure=None
    batch=batch_for(d['entries'][0]['config']);assert _offsets(batch)==d['query']['offsets'];_write(output/'query-inputs.json',d['query'])
    with _Sampler('mps',_limits(d['entries'][0]['config']['resolved'])) as sampler:
        def cap():
            sampler.sample()
            if sampler.exceeded or time.perf_counter()-started>d['finite_forecast_seconds']:raise RuntimeError('Frozen conversion envelope exhausted')
        for item in d['entries']:
            cap();authenticate(item['checkpoint']);stamp=time.perf_counter()
            state=restore_snapshot(load_snapshot(ArtifactRef(**item['checkpoint'])),'mps')
            load_seconds=time.perf_counter()-stamp
            assert state.schedule['step']==item['step'] and state.model._recipe is None
            assert exact_cpu_state_hash(state)==item['expected_state_sha256']
            for view in item['views']:
                row={'entry':item,'view':view,'status':'failed','score':None,'actual_query_targets':None,
                     'exposure_id':f'{output}:{item["step"]}:{view}','load_restore_seconds':load_seconds}
                load_seconds=0.;point=time.perf_counter();model=state.model
                old={name:getattr(model,name) for name in ('_recipe','_runtime','_contexts','_context_step')}
                try:
                    cap();before,seconds=_timed(model,lambda:exact_cpu_state_hash(state))
                    row.update(before_state_sha256=before,before_check_seconds=seconds)
                    stamp=time.perf_counter()
                    if view!='fp32':model.configure_quantization(d['recipe'])
                    row['view_configuration_seconds']=time.perf_counter()-stamp
                    try:
                        score,seconds=_timed(model,lambda:score_blocks(model,batch.block_slices,evaluation_context={'partition':'development','seed':1707}))
                        row.update(score=asdict(score),score_including_scale_conversion_seconds=seconds,
                            actual_query_targets=score.target_count,query=score.target_count,additional_loss=score.target_count,
                            loss_targets=score.target_count,forward_context=sum(len(b.inputs) for b in batch.block_slices),backward=0)
                    finally:
                        for name,value in old.items():setattr(model,name,value)
                    after,seconds=_timed(model,lambda:exact_cpu_state_hash(state))
                    row.update(after_state_sha256=after,after_check_seconds=seconds)
                    if before!=after:raise RuntimeError('Conversion query changed scientific owners')
                    cap();row['status']='completed'
                except Exception as error:
                    failure={'step':item['step'],'view':view,'error':str(error)};row['failure']=failure
                finally:
                    for name,value in old.items():setattr(model,name,value)
                    row['complete_seconds']=time.perf_counter()-point;rows.append(row)
                    _write(output/f'conversion-{item["step"]}-{view}.json',row)
                if failure:break
            if failure:break
    cached=json.loads(Path(d['cached_fp32_128']['path']).read_bytes())
    curves=[{'step':128,'view':'fp32','bpb':cached['score']['bpb'],'reused':d['cached_fp32_128']}] if stage=='initial' else []
    curves.extend({'step':r['entry']['step'],'view':r['view'],'bpb':None if r['score'] is None else r['score']['bpb']} for r in rows)
    report={'status':'failed' if failure else 'completed','failure':failure,'design':ref(designpath),'source_identity':source_identity(),
        'rows':rows,'curves':curves,'complete_seconds':time.perf_counter()-started,'memory':memory_report(sampler.samples),
        'actual_completed_query_targets':sum(r['actual_query_targets'] or 0 for r in rows),'new_training_steps':0,
        'unknown_partial_query_work':any(r['actual_query_targets'] is None for r in rows),
        'ancestry':[d['completed_attempt'],d['cached_query_ancestry']],'common_unknown_engineering_seconds':None}
    r=_write(output/'conversion-results.json',report);print(json.dumps({'result':ref(r.path),'curves':curves,'status':report['status']}))
    raise SystemExit(1 if failure else 0)
else:raise ValueError('Explicit freeze/run mode required')
