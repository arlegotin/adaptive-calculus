"""One separate same-W diagnostic on retained mixed AdamW32→FP32Muon256 state."""
import json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
from ralup.search import source_identity
from ralup.records import ArtifactRef,sha256_file
from ralup.calibrate import _write,_offsets,_Sampler,_limits,memory_report,_timed
from ralup.data.cursor import BatchCursor
from ralup.data.batches import next_logical_batch

root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
designpath=root/'docs/harness/m07-strong-conversion-design-v1.json'
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
def authenticate(r):
    if sha256_file(r['path'])!=r['sha256']:raise ValueError('Strong-conversion input hash mismatch')
def batch_for(config):
    m=config['resolved']['data']['manifest'];c=BatchCursor.from_manifest(Path(m['path']),m['sha256'],partition='development',seed=1707,run_target_budget=262144)
    return next_logical_batch(c,262144)

if sys.argv[1]=='freeze':
    native=base/'native-extension256';admission=json.loads((native/'admission.json').read_bytes())
    assert admission['status']=='pass' and admission['source_identity']==source_identity()
    run=next(r for r in admission['runs'] if r['entry']['label']=='core-muon-fp32128')
    config=run['entry']['config'];parent=run['arms'][0]['saved'][-1]
    assert config['recipe']['quantizer']=='fp32' and config['recipe']['method']=='muon_project_fp32'
    inputs=base/'query-conversion-expanded/query-inputs.json';batch=batch_for(config)
    assert _offsets(batch)==json.loads(inputs.read_bytes())['offsets']
    d={'created_utc':datetime.now(timezone.utc).isoformat(),'research_cutoff':'2026-10-03','source_identity':source_identity(),
        'purpose':'separate S12/WP04 undertrained-warm-optimizer diagnostic, not a selected warm parent or new optimizer trial',
        'checkpoint':parent,'config':config,'admission':ref(native/'admission.json'),'completed_attempt':ref(native/'finish.json'),
        'expected_state_sha256':json.loads((native/'core-muon-fp32128/comparison-0-256.json').read_bytes())['state_sha256'],
        'original_artifact_role':'continuation; remains unchanged',
        'ancestry':{'warm_parent':config['parent'],'warm_optimizer':'FP32 AdamW','warm_steps':32,'warm_targets':262144,
            'continuation_optimizer':'FP32 project Muon','continuation_steps':256,'continuation_targets':2097152,
            'total_optimizer_targets':2359296,'optimizer_reset':'existing original core32 model-only import reset; no new reset here',
            'extension':'existing strict128→256 carry-state extension; cursor/moments/history/global+named RNG preserved'},
        'query':{'partition':'development','seed':1707,'target_bytes':262144,'offsets':_offsets(batch),'same_expanded_input_reference':ref(inputs)},
        'views':['fp32','row_dynamic_max'],'recipe':{'name':'row_dynamic_max','rotation':'identity','backward':'ste'},
        'representation':'same original W, canonical signed E2M1±6, detached FP32 row max(abs(W))/6 and zero-row one; no parameter/optimizer/history change',
        'owner_contract':'strict full-state source/runtime restore; unchanged artifact role/configuration; temporary evaluation recipe restores exact original objects; full scientific hash including RNG before/after each score',
        'new_training_steps':0,'new_query_targets':524288,'finite_forecast_seconds':120,'forecast_bytes':64*1024**2,
        'forecast_basis':'four expanded262144-byte queries65.5706s measured including two restores/checks; this is two scores/one restore with120s allowance',
        'retention':'retain both signs/outcomes and raw article losses/inputs; no repeated expansion until positive',
        'cost_contract':'full AdamW32→Muon128→Muon256 ancestry once globally plus two new query/conversion/I/O costs; old AdamW-only sensitivity unchanged; unknown engineering remains unknown'}
    r=_write(designpath,d);print(json.dumps(ref(r.path)))
elif sys.argv[1]=='run':
    from ralup.state import load_snapshot,restore_snapshot,exact_cpu_state_hash
    from ralup.evaluate import score_blocks
    d=json.loads(designpath.read_bytes());assert d['source_identity']==source_identity()
    for k in ('checkpoint','admission','completed_attempt'):authenticate(d[k])
    authenticate(d['query']['same_expanded_input_reference'])
    out=Path(os.environ['M07_ATTEMPT_DIR']);start=time.perf_counter();rows=[];failure=None
    batch=batch_for(d['config']);assert _offsets(batch)==d['query']['offsets'];_write(out/'query-inputs.json',d['query'])
    stamp=time.perf_counter();state=restore_snapshot(load_snapshot(ArtifactRef(**d['checkpoint'])),'mps');load_seconds=time.perf_counter()-stamp
    assert state.schedule['step']==256 and state.model._recipe is None and state.cursor.state['cumulative_target_count']==2359296
    assert exact_cpu_state_hash(state)==d['expected_state_sha256']
    with _Sampler('mps',_limits(d['config']['resolved'])) as sampler:
        def cap():
            sampler.sample()
            if sampler.exceeded or time.perf_counter()-start>d['finite_forecast_seconds']:raise RuntimeError('Frozen strong-conversion envelope exhausted')
        for view in d['views']:
            row={'view':view,'status':'failed','score':None,'actual_query_targets':None,'exposure_id':f'{out}:{view}'};point=time.perf_counter();model=state.model
            old={k:getattr(model,k) for k in ('_recipe','_runtime','_contexts','_context_step')}
            try:
                cap();before,seconds=_timed(model,lambda:exact_cpu_state_hash(state));row.update(before_state_sha256=before,before_check_seconds=seconds)
                if view!='fp32':model.configure_quantization(d['recipe'])
                try:
                    score,seconds=_timed(model,lambda:score_blocks(model,batch.block_slices,evaluation_context={'partition':'development','seed':1707}))
                    row.update(score=asdict(score),score_including_conversion_seconds=seconds,actual_query_targets=score.target_count,
                        query=score.target_count,additional_loss=score.target_count,loss_targets=score.target_count,
                        forward_context=sum(len(b.inputs) for b in batch.block_slices),backward=0)
                finally:
                    for k,v in old.items():setattr(model,k,v)
                after,seconds=_timed(model,lambda:exact_cpu_state_hash(state));row.update(after_state_sha256=after,after_check_seconds=seconds)
                if before!=after:raise RuntimeError('Strong-conversion mutated scientific state')
                cap();row['status']='completed'
            except Exception as error:failure={'view':view,'error':str(error)};row['failure']=failure
            finally:
                for k,v in old.items():setattr(model,k,v)
                row['complete_seconds']=time.perf_counter()-point;rows.append(row);_write(out/f'conversion-{view}.json',row)
            if failure:break
    report={'status':'failed' if failure else 'completed','failure':failure,'design':ref(designpath),'source_identity':source_identity(),
        'rows':rows,'load_restore_seconds':load_seconds,'complete_seconds':time.perf_counter()-start,'memory':memory_report(sampler.samples),
        'actual_completed_query_targets':sum(r['actual_query_targets'] or 0 for r in rows),'new_training_steps':0,
        'unknown_partial_query_work':any(r['actual_query_targets'] is None for r in rows),'common_unknown_engineering_seconds':None}
    r=_write(out/'conversion-results.json',report);print(json.dumps({'result':ref(r.path),'status':report['status'],'scores':[None if r['score'] is None else r['score']['bpb'] for r in rows]}));raise SystemExit(1 if failure else 0)
else:raise ValueError('Explicit freeze/run mode required')
