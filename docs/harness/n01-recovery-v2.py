"""Root-authorized administrative recovery; no completed numerical work rerun."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

import torch
from ralup.calibrate import _create,import_warmstart,_offsets
from ralup.config import canonical_bytes
from ralup.records import ArtifactRef
from ralup.state import _validated,capture_snapshot,load_snapshot
from ralup.train import select_logical_batch
spec=importlib.util.spec_from_file_location('original_frozen_n01','docs/harness/n01-verification.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
DESIGN=Path('docs/harness/n01-administrative-recovery-v2.json')
ROOT=core.OUTPUT


def freeze():
    original,resolved=core.design()
    value={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'version':2,'prior_administrative_design':core.ref(Path('docs/harness/n01-administrative-recovery-v1.json')),'authority':'root ADMINONLY recovery accepted; exact same72-step numerical scope and original source/child/comparison/bounds, remaining work only',
           'runner':core.ref(Path(__file__)),'original_runner':core.ref(Path('docs/harness/n01-verification.py')),'original_design':core.ref(core.DESIGN),
           'producer':original['producer'],'parent':original['parent'],'timeout_record':core.ref(ROOT/'controller-timeout.json'),
           'hold_release':core.ref(ROOT/'resource-release.json'),'storage_supplement':core.ref(Path('docs/harness/n01-storage-supplement.json')),
           'completed_quest_u0':core.ref(ROOT/'quest_fp4_row_reference/U0/arm.json'),
           'authenticate_before_reuse':'exact producer/design, reconstructed initial scientificidentity, all8raw complete ownerpayload+perownerhashes+input hashes+scientific workcounters/rawNLL, independently authenticated4/8 checkpoints matching same-step complete scientificpayload',
           'remaining_launches':['ONLYoriginal child questU1 full8','ONLYoriginal child questR first4','ONLYoriginal child questR second4 fresh-interpreterstrictrestore'],
           'comparison':'original helper compare_arms at every step exact all scientificowners/scalars/NLL; no ceilings/envelope changed',
           'process_status':'original controller FAILED administrativeTimeoutExpired; completed U0 scientificpublication/ledger status known, cleanOSexit0/stdout capture unavailable and never claimed',
           'elapsed':'record original controller failure/hold/recovery clocks and union original shellredirect start through recoverycompletion once, including hold and rulingwait. Hold receipt finalelapsed supersedes estimates; total originalwave2400-second envelope reported honestly, no reset.',
           'failure_policy':'any source/owner/input/checkpoint/cost authentication or numeric comparison failure closesquestgate; preservefailure/artifacts and askroot, no rerun',
           'storage':'existing root-approved canonicalization only after exactcompletedPASS; same2GiBcap/free1GiB and prelaunch forecasts; allU0/checkpoints/failureprocessmetadata retained'}
    core.publish(DESIGN,value)
    print(json.dumps(core.ref(DESIGN)))


def authenticate():
    start=time.perf_counter();original,resolved=core.design()
    frozen=json.loads(DESIGN.read_bytes());assert core.ref(Path(__file__))==frozen['runner']
    assert core.ref(core.DESIGN)==frozen['original_design']
    assert core.ref(Path('docs/harness/n01-verification.py'))==frozen['original_runner']
    arm_path=core.helper.auth(frozen['completed_quest_u0']);arm=json.loads(arm_path.read_bytes())
    assert arm['source']==original['producer'] and arm['design']==core.ref(core.DESIGN)
    assert arm['case']==original['cases'][2] and len(arm['steps'])==8 and set(arm['states'])==set(map(str,range(1,9)))
    state=import_warmstart(ArtifactRef(**original['parent']),_create(resolved,'target_train',65536,808),partition='target_train',continuation_targets=65536)
    state.microbatch=64;state.model.configure_quantization({**state.model._recipe,'name':'quest_fp4_row_reference'})
    expected_initial=hashlib.sha256(core.helper.encoded(*core.scientific(capture_snapshot(state)))).hexdigest()
    assert expected_initial==arm['initial_science_sha256']
    parent=json.loads(core.helper.auth(original['parent']).read_bytes())
    verified=[]
    for step_record in arm['steps']:
        step=step_record['step'];assert step in range(1,9)
        batch=select_logical_batch(state)
        digest=hashlib.sha256(canonical_bytes({'offsets':_offsets(batch)}))
        for block in batch.block_slices:
            digest.update(canonical_bytes({'inputs':list(block.inputs),'positions':list(block.positions),'labels':list(block.labels),'mask':list(block.score_mask)}))
        assert digest.hexdigest()==step_record['input_sha256'] and step_record['targets']==batch.target_count==8192
        value=arm['states'][str(step)];tensors,scalars=core.helper.read_science(value)
        assert set(tensors)==set(value['owners'])
        assert hashlib.sha256(canonical_bytes(scalars)).hexdigest()==value['scalar_sha256']
        for name,tensor in tensors.items():
            assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest()==value['owners'][name]['sha256']
        work=scalars['scientific_work']
        assert work['stream']['targets']==parent['cost']['stream']['targets']+step*8192
        assert work['passes']['backward']==parent['cost']['passes']['backward']+step*8192
        assert work['passes']['loss_targets']==parent['cost']['passes']['loss_targets']+step*8192
        assert work['stream']['updates']==parent['cost']['stream']['updates']+step
        assert step_record['nll_sum']==next(e for e in arm['cost']['events'][len(parent['cost']['events']):] if e['status']=='completed_step' and e['step']==step-1)['nll_sum']
        verified.append({'step':step,'input_sha256':digest.hexdigest(),'payload_sha256':value['uncompressed_sha256'],'complete_owner_count':len(tensors),'raw_nll_sum':step_record['nll_sum'],'scientific_work':work})
    checkpoints=[]
    for step in (4,8):
        checkpoint=load_snapshot(ArtifactRef(**arm['checkpoints'][str(step)]));payload=_validated(checkpoint)
        assert {k:v.sha256 for k,v in payload['identities']['automatic'].items()}==original['producer']
        assert payload['schedule']['step']==step and payload['microbatch']==64
        encoded=core.helper.encoded(*core.scientific(checkpoint))
        assert hashlib.sha256(encoded).hexdigest()==arm['states'][str(step)]['uncompressed_sha256']
        checkpoints.append({'step':step,'checkpoint':arm['checkpoints'][str(step)],'complete_scientific_payload_matches':True})
    # Actual process exit cannot be inferred from this completed record.
    report={'recovery_design':core.ref(DESIGN),'original_u0':frozen['completed_quest_u0'],'initial_science_sha256':expected_initial,
            'verified_steps':verified,'checkpoints':checkpoints,'producer':original['producer'],'reuse_authenticated':True,
            'child_scientific_publication_and_status':'completed','child_actual_os_exit_status':'unavailable','child_stdout_terminal_capture':'unavailable',
            'new_forward_backward_passes':0,'new_setup_snapshot_authentication_seconds':time.perf_counter()-start}
    artifact=core.publish(ROOT/'recovery-authentication.json',report);print(json.dumps({'authentication':{'path':str(artifact.path),'sha256':artifact.sha256}}))


def run():
    start=time.perf_counter();original,resolved=core.design()
    frozen=json.loads(DESIGN.read_bytes());assert core.ref(Path(__file__))==frozen['runner']
    auth=json.loads((ROOT/'recovery-authentication.json').read_bytes());assert auth['reuse_authenticated']
    assert auth['recovery_design']==core.ref(DESIGN) and auth['producer']==original['producer']
    directory=ROOT/'quest_fp4_row_reference'
    u0=json.loads(core.helper.auth(frozen['completed_quest_u0']).read_bytes())
    u1=core.launch(2,'U1');first=core.launch(2,'R','first');second=core.launch(2,'R','second',first['checkpoints']['4'])
    replay={**second,'states':{**first['states'],**second['states']},'steps':first['steps']+second['steps'],
            'checkpoints':{**first['checkpoints'],**second['checkpoints']},'project_seconds':first['project_seconds']+second['project_seconds']}
    core.publish(directory/'R/combined.json',replay)
    failures=[];checks=[]
    if len({u0['initial_science_sha256'],u1['initial_science_sha256'],replay['initial_science_sha256']})!=1:failures.append({'owner':'initial_science'})
    for label,other in [('duplicate',u1),('strict_resume',replay)]:
        comparisons=core.helper.compare_arms(u0,other,directory,label)
        for step,check in comparisons.items():
            if check['discrete_mismatches']:failures.append({'comparison':label,'step':step,'owners':check['discrete_mismatches']})
            for owner,row in check['owners'].items():
                if not row['exact']:failures.append({'comparison':label,'step':step,'owner':owner,'observed':row})
        checks.append({'label':label,'steps':comparisons})
    result={'case':original['cases'][2],'comparisons':checks,'failures':failures,'accepted':not failures,
            'arms':[core.ref(directory/a/p) for a,p in [('U0','arm.json'),('U1','arm.json'),('R','combined.json')]],
            'administrative_recovery':core.ref(DESIGN),'reuse_authentication':core.ref(ROOT/'recovery-authentication.json'),
            'original_controller':'FAILED administrative hold-induced timeout; never claimed clean'}
    core.publish(directory/'result.json',result)
    started=Path('artifacts/checks/n01-wave.txt').stat().st_birthtime
    total=time.time()-started
    outcomes=[{'recipe':r,'accepted':json.loads((ROOT/r/'result.json').read_bytes())['accepted'],'result':core.ref(ROOT/r/'result.json')} for r in original['recipes']]
    core.publish(ROOT/'wave.json',{'design':core.ref(core.DESIGN),'recovery_design':core.ref(DESIGN),'results':outcomes,'status':'PASS bounded enumerated scope only' if all(x['accepted'] for x in outcomes) else 'FAILED scope',
                                 'new_continuation_steps':72,'new_scored_bytes':589824,'recovery_project_seconds':time.perf_counter()-start,
                                 'full_project_union_seconds_original_start_through_recovery':total,'original_wave_envelope_seconds':original['envelopes']['wave_wall_seconds'],
                                 'total_envelope_exceeded':total>original['envelopes']['wave_wall_seconds'],
                                 'cost_contract':'full chronological union includes original startup/failedcontroller/recovery/hold/rulingwait/observerIO once; separately reported overlapping arm/component clocks not summed; all completedwork reused/no reruns',
                                 'hold_release':core.ref(ROOT/'resource-release.json'),'controller_timeout':core.ref(ROOT/'controller-timeout.json')})
    print(json.dumps({'accepted':not failures,'failures':len(failures),'total_union_seconds':total,'total_envelope_exceeded':total>original['envelopes']['wave_wall_seconds']}))
    if failures:raise RuntimeError('Required exact quest scope CLOSED; no enlargement')


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='authenticate':authenticate()
    elif sys.argv[1]=='run':run()
