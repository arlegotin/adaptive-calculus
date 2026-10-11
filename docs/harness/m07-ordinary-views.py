"""Risk-ordered same-W representation sensitivity, not optimizer competition."""
import json,os,sys,time
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
from ralup.search import source_identity
from ralup.records import ArtifactRef,sha256_file
from ralup.calibrate import _write,_offsets,_timed,_Sampler,_limits,memory_report
from ralup.state import load_snapshot,restore_snapshot,exact_cpu_state_hash
from ralup.evaluate import score_blocks
from ralup.data.cursor import BatchCursor
from ralup.data.batches import next_logical_batch

root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
designpath=root/'docs/harness/m07-ordinary-views-design-v1.json'
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
def auth(r):
    if sha256_file(r['path'])!=r['sha256']:raise ValueError('Ordinary view input hash mismatch')
def inputs(config):
    m=config['resolved']['data']['manifest'];c=BatchCursor.from_manifest(Path(m['path']),m['sha256'],partition='development',seed=1707,run_target_budget=262144)
    return next_logical_batch(c,262144)

if sys.argv[1]=='freeze':
    proofpath=base/'native-derived-parent/import-results.json';proof=json.loads(proofpath.read_bytes())
    assert proof['status']=='pass' and proof['source_identity']==source_identity()
    parentdesign=json.loads((root/'docs/harness/m07-derived-parent-design-v1.json').read_bytes())
    priorquery=json.loads((root/'docs/harness/m07-strong-conversion-design-v1.json').read_bytes())
    config=parentdesign['fresh_config'];query=inputs(config)
    assert _offsets(query)==priorquery['query']['offsets']
    rows=[{'label':'max-row-hadamard','name':'row_dynamic_max','rotation':'full_hadamard','backward':'ste'},
          {'label':'quest-row-identity','name':'quest_fp4_row_reference','rotation':'identity','backward':'trust'},
          {'label':'quest-row-hadamard','name':'quest_fp4_row_reference','rotation':'full_hadamard','backward':'trust'},
          {'label':'quest-tensor-identity','name':'quest_fp4_tensor_reference','rotation':'identity','backward':'trust'}]
    design={'created_utc':datetime.now(timezone.utc).isoformat(),'source_identity':source_identity(),'helper':ref(Path(__file__)),
        'purpose':'S14/S26 development representation sensitivity to prioritize strong quantizers before continuation; every outcome retained',
        'checkpoint':proof['current_state'],'expected_state_sha256':proof['current_state_sha256'],
        'import_proof':ref(proofpath),'import_complete_attempt':ref(base/'native-derived-parent/finish.json'),
        'original_strong_query':ref(base/'query-strong-conversion/conversion-results.json'),'original_strong_query_complete_attempt':ref(base/'query-strong-conversion/finish.json'),
        'original_query_design':ref(root/'docs/harness/m07-strong-conversion-design-v1.json'),
        'config':config,'views':rows,'query':{'partition':'development','seed':1707,'target_bytes':262144,'offsets':_offsets(query)},
        'finite_forecast_seconds':180,'forecast_bytes':64*1024**2,'new_training_steps':0,'maximum_new_query_targets':4*262144,
        'representation':'canonical E2M1 signed codebook ±6, FP32 auxiliary/scale precision, detached row max/6 or deterministic Gaussian-fit scale; original W unchanged',
        'provenance':'existing independently implemented S14 mathematical QuEST reference; exact scale/grouping/rotation disclosed, upstream release/grouping parity unverified; no restricted source copied',
        'cache_contract':'reuse prior FP32 and row_dynamic identity scores only because actual import proof verified exact all W/aux, and exact original262144 input list matches',
        'cost_contract':'full original warm/continuation/query/admission/import ancestry charged once globally plus four new queries and I/O; source-derived current import is not a new independent seed',
        'limits':'forward-only sensitivity does not test STE/trust gradients, establish longer training ranks, defeat any optimizer/quantizer, or freeze B0'}
    _write(designpath,design);print(json.dumps(ref(designpath)))
elif sys.argv[1]=='run':
    d=json.loads(designpath.read_bytes());assert d['source_identity']==source_identity()
    for key in ('helper','checkpoint','import_proof','import_complete_attempt','original_strong_query','original_strong_query_complete_attempt','original_query_design'):auth(d[key])
    out=Path(os.environ['M07_ATTEMPT_DIR']);start=time.perf_counter();query=inputs(d['config']);assert _offsets(query)==d['query']['offsets']
    _write(out/'query-inputs.json',d['query']);rows=[];failure=None
    with _Sampler('mps',_limits(d['config']['resolved'])) as sampler:
        def cap():
            sampler.sample()
            if sampler.exceeded or time.perf_counter()-start>d['finite_forecast_seconds']:raise RuntimeError('Frozen ordinary view cap exhausted')
        cap();state=restore_snapshot(load_snapshot(ArtifactRef(**d['checkpoint'])),'mps')
        assert exact_cpu_state_hash(state)==d['expected_state_sha256']
        for view in d['views']:
            row={'view':view,'status':'failed','actual_query_targets':None};t=time.perf_counter()
            try:
                cap();before=exact_cpu_state_hash(state);old=[getattr(state.model,k) for k in ('_recipe','_runtime','_contexts','_context_step')]
                try:
                    state.model.configure_quantization({k:view[k] for k in ('name','rotation','backward')})
                    score,seconds=_timed(state.model,lambda:score_blocks(state.model,query.block_slices,evaluation_context={'partition':'development','seed':1707}))
                finally:
                    for k,v in zip(('_recipe','_runtime','_contexts','_context_step'),old):setattr(state.model,k,v)
                row.update(score=asdict(score),score_seconds=seconds,actual_query_targets=score.target_count,
                    additional_loss=score.target_count,loss_targets=score.target_count,forward_context=sum(len(b.inputs) for b in query.block_slices),backward=0,
                    before_state_sha256=before,after_state_sha256=exact_cpu_state_hash(state))
                assert row['after_state_sha256']==before;cap();row['status']='completed'
            except Exception as e:failure={'view':view,'error':str(e)};row['failure']=failure
            row['complete_seconds']=time.perf_counter()-t;rows.append(row);_write(out/(view['label']+'.json'),row)
            if failure:break
    report={'status':'failed' if failure else 'completed','failure':failure,'design':ref(designpath),'source_identity':source_identity(),'rows':rows,
        'cached_references':d['original_strong_query'],'complete_seconds':time.perf_counter()-start,'new_training_steps':0,
        'actual_completed_query_targets':sum(r['actual_query_targets'] or 0 for r in rows),'unknown_partial_query_work':any(r['actual_query_targets'] is None for r in rows),
        'memory':memory_report(sampler.samples),'cost_contract':d['cost_contract'],'limits':d['limits']}
    _write(out/'query-results.json',report);print(json.dumps({'report':ref(out/'query-results.json'),'status':report['status'],
        'curves':[{'label':r['view']['label'],'bpb':r.get('score',{}).get('bpb')} for r in rows]}));raise SystemExit(1 if failure else 0)
else:raise ValueError('Explicit freeze/run required')
