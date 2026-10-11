"""Prospective read-only development scoring of qualified retained states."""
import json, os, sys, time
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
from ralup.search import source_identity
from ralup.records import ArtifactRef, sha256_file
from ralup.config import canonical_bytes
from ralup.calibrate import _write, _offsets, _Sampler, _limits, memory_report, _timed
from ralup.data.cursor import BatchCursor
from ralup.data.batches import next_logical_batch

root=Path.cwd()
base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
stage=sys.argv[2] if len(sys.argv)>2 else 'horizon128'
assert stage in ('horizon128','extension256')
native=base/('native-horizon128' if stage=='horizon128' else 'native-extension256')
designpath=root/('docs/harness/m07-query-design-v1.json' if stage=='horizon128' else 'docs/harness/m07-query-design-v2.json')
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}

def authenticate(value):
    if sha256_file(value['path'])!=value['sha256']:
        raise ValueError('Immutable query input hash mismatch')

def query_batch(config):
    manifest=config['resolved']['data']['manifest']
    cursor=BatchCursor.from_manifest(Path(manifest['path']),manifest['sha256'],
        partition='development',seed=1707,run_target_budget=8192)
    return next_logical_batch(cursor,8192)

if sys.argv[1]=='freeze':
    terminal=json.loads((native/'finish.json').read_bytes())
    admission=json.loads((native/'admission.json').read_bytes())
    assert (terminal['exit_code'],admission['status'])==((1,'blocked') if stage=='horizon128' else (0,'pass'))
    assert admission['source_identity']==source_identity()
    entries=[]
    for run in admission['runs']:
        if run.get('failure') or len(run['arms'])!=2:
            continue
        original=run if stage=='horizon128' else run['entry']
        case=native/original['label']
        comparisons=[]
        for step in ((1,32,64,128) if stage=='horizon128' else (192,256)):
            paths=[case/f'comparison-{arm}-{step}.json' for arm in (0,1)]
            values=[json.loads(p.read_bytes()) for p in paths]
            assert values[0]==values[1]
            comparisons.extend(ref(p) for p in paths)
        steps=(64,128) if stage=='horizon128' else (192,256)
        for step,arm,saved in ((steps[0],1,run['arms'][1]['saved'][0]),(steps[1],0,run['arms'][0]['saved'][-1])):
            authenticate(saved)
            entries.append({'label':original['label'],'step':step,'arm':arm,'checkpoint':saved,
                'config':original['config'],'scope_result':ref(case/'scope-result.json'),
                'comparisons':comparisons,'parent':original['config'].get('parent'),
                'training_origin':('stateful extension128-to256 from admitted core32-parent continuation' if stage=='extension256' else
                    'admission warmstart' if original['config']['role']=='warmstart' else 'admission continuation from core32 warm parent'),
                'warm_parent_training_targets':0 if not original['config'].get('parent') else 32*8192,
                'trajectory_training_targets':step*8192,
                'raw_training_inputs':[ref(case/f'arm-{a}.jsonl') for a in (0,1)]})
    batch=query_batch(entries[0]['config'])
    payload={'created_utc':datetime.now(timezone.utc).isoformat(),'research_cutoff':'2026-10-03',
        'purpose':'development query only; admission-origin duration doublings; no independent replications or final selection',
        'source_identity':source_identity(),'admission':ref(native/'admission.json'),
        'completed_attempt':ref(native/'finish.json'),'entries':entries,
        'query':{'partition':'development','seed':1707,'target_bytes':8192,'offsets':_offsets(batch)},
        'finite_forecast_seconds':600,'forecast_basis':f'{len(entries)}8192-byte evaluations plus strict load/restore/full-state read-only hashes; no training',
        'forecast_bytes':64*1024**2,'new_training_steps':0,
        'origin_contract':f'arm1 strict{steps[0]} midpoint and arm0 completed{steps[1]}; exact matched producer source; paired duplicate arms share seed707',
        'cost_contract':'charge complete original admission ancestry plus new query/I/O/checking; ancestor counted once globally, not per point; unknown common engineering cost remains unknown'}
    result=_write(designpath,payload);print(json.dumps(ref(result.path)))
elif sys.argv[1]=='run':
    from ralup.state import load_snapshot,restore_snapshot,exact_cpu_state_hash
    from ralup.evaluate import score_blocks
    design=json.loads(designpath.read_bytes())
    assert design['source_identity']==source_identity()
    authenticate(design['admission']);authenticate(design['completed_attempt'])
    output=Path(os.environ['M07_ATTEMPT_DIR']);started=time.perf_counter();rows=[];failure=None
    batch=query_batch(design['entries'][0]['config'])
    assert _offsets(batch)==design['query']['offsets']
    _write(output/'query-inputs.json',design['query'])
    with _Sampler('mps',_limits(design['entries'][0]['config']['resolved'])) as sampler:
        def cap():
            sampler.sample()
            if sampler.exceeded:raise RuntimeError('Query memory cap exceeded')
            if time.perf_counter()-started>design['finite_forecast_seconds']:
                raise RuntimeError('Frozen query envelope exhausted')
        for item in design['entries']:
            row={'entry':item,'exposure_id':str(output)+':'+item['label']+':'+str(item['step']),
                 'status':'failed','score':None,'actual_query_targets':None}
            point_started=time.perf_counter()
            try:
                cap();authenticate(item['checkpoint']);authenticate(item['scope_result'])
                for value in item['comparisons']+item['raw_training_inputs']:authenticate(value)
                stamp=time.perf_counter()
                state=restore_snapshot(load_snapshot(ArtifactRef(**item['checkpoint'])),'mps')
                row['load_restore_seconds']=time.perf_counter()-stamp
                assert state.schedule['step']==item['step']
                before,seconds=_timed(state.model,lambda:exact_cpu_state_hash(state))
                row['before_state_sha256']=before;row['before_check_seconds']=seconds
                cap()
                score,seconds=_timed(state.model,lambda:score_blocks(state.model,batch.block_slices,
                    evaluation_context={'partition':'development','seed':1707}))
                row.update(score=asdict(score),score_seconds=seconds,actual_query_targets=score.target_count,
                    additional_loss=score.target_count,loss_targets=score.target_count,
                    forward_context=sum(len(b.inputs) for b in batch.block_slices),backward=0)
                after,seconds=_timed(state.model,lambda:exact_cpu_state_hash(state))
                row['after_state_sha256']=after;row['after_check_seconds']=seconds
                if before!=after:raise RuntimeError('Evaluation changed scientific state')
                cap();row['status']='completed'
            except Exception as error:
                failure={'entry':item['label'],'step':item['step'],'error':str(error)}
                row['failure']=failure
            finally:
                row['complete_seconds']=time.perf_counter()-point_started
                rows.append(row);_write(output/f"{item['label']}-{item['step']}.json",row)
            if failure:break
    report={'status':'failed' if failure else 'completed','failure':failure,'rows':rows,
        'design':ref(designpath),'source_identity':source_identity(),'complete_seconds':time.perf_counter()-started,
        'full_admission_ancestry':design['completed_attempt'],'ancestry_counting':'once globally; not repeated per curve point',
        'actual_completed_query_targets':sum(r['actual_query_targets'] or 0 for r in rows),
        'unknown_partial_query_work':any(r['status']=='failed' and r['actual_query_targets'] is None for r in rows),
        'new_training_steps':0,'memory':memory_report(sampler.samples),'common_unknown_engineering_seconds':None}
    result=_write(output/'query-results.json',report)
    print(json.dumps({'result':ref(result.path),'status':report['status'],
        'curves':[{'label':r['entry']['label'],'step':r['entry']['step'],'bpb':None if r['score'] is None else r['score']['bpb']} for r in rows]}))
    raise SystemExit(1 if failure else 0)
else:
    raise ValueError('Explicit freeze/run mode required')
