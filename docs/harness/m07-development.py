"""Prospective M07 current-runtime gate and finite real-shape admission.

Run only through m07-attempt.py (archives producer before product imports).
Runtime scope is October11; research cutoff October3. No protected data route.
"""
import json,os,sys,time,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path.cwd();output=Path(os.environ['M07_ATTEMPT_DIR']);mode=sys.argv[1]
from ralup.config import canonical_bytes
from ralup.records import ArtifactRef,sha256_file
from ralup.search import source_identity,create_trial_state
from ralup.calibrate import _write,_event,_timed,_offsets

def ref(path):return {'path':str(path),'sha256':sha256_file(path)}
def write(name,value):return _write(output/name,value)

if mode=='gate':
    from ralup.environment import write_environment_report,probe_operations
    # Probe's fixed relative output is task-owned permanent Data, not Git volume.
    os.chdir(output)
    environment=write_environment_report(output/'environment.json')
    operations=probe_operations('mps')
    os.chdir(root)
    old=json.loads((root/'docs/data/wp00-local-gate-v2.json').read_bytes())
    evidence=[v for v in old['evidence'] if not v['path'].startswith('artifacts/environment/')]
    evidence=[{'path':str(root/v['path']),'sha256':v['sha256']} for v in evidence]
    evidence += [ref(environment.path)]+[ref(v.path) for v in operations.evidence]
    evidence += [ref(root/'docs/data/wp00-local-acceptance-v2.json')]
    report={'status':operations.status,'blockers':list(operations.blockers),'evidence':evidence}
    write('wp00-local-gate-v3.json',report)
    write('scope.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'source_identity':source_identity(),
        'historical_acceptance':'v2 retained; local research rights only',
        'current_operation_status':operations.status,'scientific_admission':None,
        'source_copy_grants':'unchanged; equation-local routes separately declared',
        'research_cutoff':'2026-10-03','new_observation':'2026-10-11'})
    print(json.dumps({'environment':ref(environment.path),'gate':ref(output/'wp00-local-gate-v3.json'),'status':operations.status}))
    raise SystemExit(0 if operations.status=='pass' else 1)

if mode=='numerics':
    from ralup.baselines.checks import check_adamw
    from ralup.baselines.muon_checks import check_muon
    from ralup.train import train_step,select_logical_batch
    from ralup.state import capture_snapshot,restore_snapshot,save_snapshot,exact_cpu_state_hash
    import torch
    design=json.loads(Path(sys.argv[2]).read_bytes())
    assert design['source_identity']==source_identity()
    started=time.perf_counter();parity=[];scopes=[];runs=[];failure=None
    for name,check in [('adamw',check_adamw),('muon',check_muon)]:
        r=check('mps',output/(name+'-parity.json'));value=json.loads(r.path.read_bytes());parity.append(ref(r.path))
        if value['status']!='pass':failure={'parity':name,'blockers':value['blockers']};break
    if failure is None:
        for config in design['cases']:
            scope=[config['resolved']['profile'],config['recipe']['method'],config['recipe']['quantizer'],config['recipe']['rotation'],config['recipe']['backward'],config['packing']]
            label='case-'+str(len(runs));case=output/label;case.mkdir()
            state=create_trial_state(config);initial=capture_snapshot(state);row={'scope':scope,'steps':[], 'cost':[]}
            expected=[]
            # Independent full replay from same authenticated initial snapshot;
            # second arm additionally strict-restores its mid-horizon state.
            for arm in range(2):
                state=restore_snapshot(initial,'mps');armstart=time.perf_counter()
                with (case/f'arm-{arm}.jsonl').open('x') as log:
                    for step in range(design['steps']):
                        if arm==1 and step==design['steps']//2:
                            snap=capture_snapshot(state);checkpoint=save_snapshot(snap,case/'resume');state=restore_snapshot(snap,'mps')
                        from ralup.optim.base import SchedulePoint
                        state.schedule['point']=SchedulePoint(config['recipe']['learning_rate'],config['recipe'].get('decay',0.),step/max(1,design['steps']-1))
                        batch=select_logical_batch(state,8192);inputs=_offsets(batch)
                        try:result=train_step(state,batch)
                        except Exception as error:
                            failure={'scope':scope,'arm':arm,'step':step,'error':str(error)};break
                        digest=exact_cpu_state_hash(state)
                        event={'step':step+1,'state_sha256':digest,'inputs':inputs,'events':state.ledger.events[-1:]}
                        log.write(canonical_bytes(event).decode()+'\n');log.flush()
                        if arm==0:expected.append(digest)
                        elif digest!=expected[step]:
                            save_snapshot(capture_snapshot(state),case/'mismatch')
                            failure={'scope':scope,'arm':arm,'step':step+1,'error':'exact duplicate/resume mismatch','expected':expected[step],'actual':digest};break
                row['cost'].append({'arm':arm,'complete_seconds':time.perf_counter()-armstart,'stream':state.ledger.stream,'passes':state.ledger.passes,'elapsed':state.ledger.elapsed})
                save_snapshot(capture_snapshot(state),case/f'final-{arm}')
                if failure:break
            runs.append(row)
            if failure:break
            scopes.append(scope)
    report={'status':'blocked' if failure else 'pass','source_identity':source_identity(),'scopes':scopes,
        'maximum_steps':design['steps'],'contract':design['contract'],'failure':failure,'runs':runs,'parity':parity,
        'created_utc':datetime.now(timezone.utc).isoformat(),'design':ref(Path(sys.argv[2])),
        'complete_seconds':time.perf_counter()-started,'common_unknown_engineering_seconds':None}
    write('admission.json',report);print(json.dumps({'admission':ref(output/'admission.json'),'status':report['status'],'failure':failure}))
    raise SystemExit(1 if failure else 0)
raise ValueError('Unknown explicit development stage')
