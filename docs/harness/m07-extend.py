"""Source-exact stateful128→256 continuation; no qualified prefix is replayed."""
import copy,json,os,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
from ralup.search import source_identity
from ralup.records import ArtifactRef,sha256_file
from ralup.config import canonical_bytes
from ralup.calibrate import _write,_offsets,_Sampler,_limits,memory_report
from ralup.state import load_snapshot,restore_snapshot,capture_snapshot,save_snapshot,exact_cpu_state_hash
from ralup.data.cursor import BatchCursor
from ralup.train import train_step,select_logical_batch
from ralup.optim.base import SchedulePoint

root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
stage=sys.argv[2] if len(sys.argv)>2 else 'muon'
assert stage in ('muon','warm')
native=base/'native-horizon128'
designpath=root/('docs/harness/m07-extension-design-v1.json' if stage=='muon' else 'docs/harness/m07-warm-extension-design-v1.json')
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
digest=lambda x:hashlib.sha256(canonical_bytes(x)).hexdigest()
def authenticate(r):
    if sha256_file(r['path'])!=r['sha256']:raise ValueError('Extension input hash mismatch')

if sys.argv[1]=='freeze':
    admission=json.loads((native/'admission.json').read_bytes());assert admission['source_identity']==source_identity()
    entries=[]
    for label in (('core-muon-fp32128','core-muon-dynamic128') if stage=='muon' else ('core-fp32-warm128',)):
        case=native/label;run=json.loads((case/'scope-result.json').read_bytes())
        assert len(run['arms'])==2 and not run.get('failure')
        comparisons=[json.loads((case/f'comparison-{a}-128.json').read_bytes()) for a in (0,1)]
        assert comparisons[0]==comparisons[1]
        entries.append({'label':label,'config':run['config'],'scope':ref(case/'scope-result.json'),
            'parents':[a['saved'][-1] for a in run['arms']],
            'old_state_sha256':comparisons[0]['state_sha256'],
            'old_comparisons':[ref(case/f'comparison-{a}-128.json') for a in (0,1)]})
    design={'created_utc':datetime.now(timezone.utc).isoformat(),'source_identity':source_identity(),
        'purpose':'measured development duration doubling with strict current-source stateful extension',
        'from_step':128,'to_step':256,'comparisons':[192,256],'strict_restore_step':192,
        'entries':entries,'admission_ancestry':ref(native/'finish.json'),'old_admission':ref(native/'admission.json'),
        'additional_targets_per_arm':128*8192,'maximum_new_steps':256*len(entries),
        'finite_forecast_seconds':900 if stage=='muon' else 450,'forecast_bytes':len(entries)*1024**3,
        'forecast_basis':'measured identity128 arms65-110s; declared128-step tails plus checkpoint/hash/source/restore allowance',
        'extension_contract':'authenticate complete old state first; change only cursor final byte envelope and explicit common configuration annotation; preserve cursor position/permutation/carry/source and global/named RNG, original-W moments/history/gradients; no optimizer reset',
        'schedule_contract':'same constant matrix/auxiliary LRs and zero decay; terminal progress1 retained; no schedule reset',
        'retention':'completed255/256 each arm plus arm1 strict192 midpoint and failure-state snapshot when possible; raw every-step input/loss/cost; no every-step full snapshot',
        'cost_contract':'all admission parents and new tails charged; historical prefix executed once per existing arm, never relabeled as new data'}
    r=_write(designpath,design);print(json.dumps(ref(r.path)))
elif sys.argv[1]=='run':
    design=json.loads(designpath.read_bytes());assert design['source_identity']==source_identity()
    for r in (design['admission_ancestry'],design['old_admission']):authenticate(r)
    output=Path(os.environ['M07_ATTEMPT_DIR']);started=time.perf_counter();runs=[];scopes=[];failure=None
    for item in design['entries']:
        case=output/item['label'];case.mkdir();run={'entry':item,'arms':[]};expected={};extension_hash=None
        try:
            with _Sampler('mps',_limits(item['config']['resolved'])) as sampler:
                def cap():
                    sampler.sample()
                    if sampler.exceeded:raise RuntimeError('Frozen extension memory cap exceeded')
                    if time.perf_counter()-started>design['finite_forecast_seconds']:
                        raise RuntimeError('Frozen complete extension envelope exhausted')
                for arm,parent in enumerate(item['parents']):
                    cap();authenticate(parent);armstart=time.perf_counter()
                    state=restore_snapshot(load_snapshot(ArtifactRef(**parent)),'mps')
                    if exact_cpu_state_hash(state)!=item['old_state_sha256']:
                        raise RuntimeError('Authenticated old complete state differs from qualified endpoint')
                    assert state.schedule['step']==128
                    old_cursor=state.cursor;old_configuration=state.configuration
                    old=old_cursor.state;assert old['run_target_budget']==old['cumulative_target_count']
                    new={**old,'run_target_budget':old['cumulative_target_count']+design['additional_targets_per_arm']}
                    new_cursor=BatchCursor(new)
                    if {k:v for k,v in new_cursor.state.items() if k!='run_target_budget'}!={k:v for k,v in old.items() if k!='run_target_budget'}:
                        raise RuntimeError('Extension changed cursor position, permutation, source or RNG')
                    new_configuration=copy.deepcopy(old_configuration)
                    new_configuration['horizon_extension']={'contract':'M07-stateful128-to256-v1',
                        'from_completed_step':128,'to_completed_step':256,'source_admission':design['old_admission'],
                        'old_final_byte_limit':old['run_target_budget'],'new_final_byte_limit':new['run_target_budget']}
                    if exact_cpu_state_hash(state)!=item['old_state_sha256']:
                        raise RuntimeError('Extension preparation mutated old scientific owners')
                    state.cursor=new_cursor;state.configuration=new_configuration
                    current=exact_cpu_state_hash(state)
                    if extension_hash is None:extension_hash=current
                    elif current!=extension_hash:raise RuntimeError('Paired extended initial owners differ')
                    receipt={'arm':arm,'parent':parent,'old_complete_state_sha256':item['old_state_sha256'],
                        'extended_initial_state_sha256':current,'old_cursor_sha256':digest(old),
                        'new_cursor_sha256':digest(new),'old_limit':old['run_target_budget'],'new_limit':new['run_target_budget'],
                        'old_configuration_sha256':digest(old_configuration),'new_configuration_sha256':digest(new_configuration),
                        'preserved_owner_check':'old scientific hash unchanged after constructing new cursor/config; paired full extended state exact',
                        'saved':[]}
                    _write(case/f'extension-{arm}.json',receipt)
                    with (case/f'arm-{arm}.jsonl').open('x') as raw:
                        for step in range(128,256):
                            cap()
                            if arm==1 and step==192:
                                snapshot=capture_snapshot(state);r=save_snapshot(snapshot,case/'strict-midpoint192')
                                receipt['saved'].append(ref(r.path));state=restore_snapshot(snapshot,'mps')
                            recipe=item['config']['recipe'];state.schedule['point']=SchedulePoint(recipe['learning_rate'],recipe.get('decay',0.),1.)
                            batch=select_logical_batch(state,8192);inputs=_offsets(batch);event_start=len(state.ledger.events)
                            try:train_step(state,batch)
                            finally:
                                raw.write(canonical_bytes({'execution_id':f'{output.name}:{item["label"]}:{arm}:{step}',
                                    'step':step+1,'inputs':inputs,'events':state.ledger.events[event_start:]}).decode()+'\n');raw.flush()
                            cap()
                            if step+1 in design['comparisons']:
                                value=exact_cpu_state_hash(state);_write(case/f'comparison-{arm}-{step+1}.json',{'step':step+1,'state_sha256':value})
                                if arm==0:expected[step+1]=value
                                elif value!=expected[step+1]:raise RuntimeError(f'Complete scientific owners differ at{step+1}')
                            if step+1>=255:
                                r=save_snapshot(capture_snapshot(state),case/f'completed-{arm}-{step+1}');receipt['saved'].append(ref(r.path));cap()
                    receipt.update(complete_seconds=time.perf_counter()-armstart,stream=state.ledger.stream,passes=state.ledger.passes,
                        elapsed=state.ledger.elapsed,parent_ancestry=design['admission_ancestry'])
                    run['arms'].append(receipt);_write(case/f'arm-{arm}-completed.json',receipt)
                run['memory']=memory_report(sampler.samples)
            c=item['config'];scopes.append([c['resolved']['profile'],c['recipe']['method'],c['recipe']['quantizer'],c['recipe']['rotation'],c['recipe']['backward'],c['packing']])
        except Exception as error:
            failure={'label':item['label'],'error':str(error)};run['failure']=failure
            if 'state' in locals():
                try:
                    r=save_snapshot(capture_snapshot(state),case/'stopped-state');run['stopped_state']=ref(r.path)
                except Exception as save_error:run['stopped_state_failure']=str(save_error)
        finally:
            runs.append(run);_write(case/'scope-result.json',run)
        if failure:break
    report={'status':'blocked' if failure else 'pass','failure':failure,'source_identity':source_identity(),
        'design':ref(designpath),'scopes':scopes,'maximum_steps':256,'runs':runs,
        'complete_seconds':time.perf_counter()-started,'ancestry':design['admission_ancestry'],
        'proof_scope':'old128 complete exact proof composed with declared source-exact stateful128-step tail; no prefix replay',
        'common_unknown_engineering_seconds':None}
    r=_write(output/'admission.json',report);print(json.dumps({'admission':ref(r.path),'status':report['status'],'failure':failure}))
    raise SystemExit(1 if failure else 0)
else:raise ValueError('Explicit freeze/run mode required')
