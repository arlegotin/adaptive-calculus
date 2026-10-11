"""Index completed immutable M10 attempts without rewriting producer evidence."""
import hashlib,json,os
from pathlib import Path
store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m10-20261011')
active=Path(os.environ['M10_ATTEMPT_DIR'])
def ref(path):
    data=path.read_bytes()
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
attempts=[]
for base in sorted(store.iterdir()):
    if base==active or not (base/'finish.json').exists():continue
    startup=json.loads((base/'startup.json').read_bytes());finish=json.loads((base/'finish.json').read_bytes())
    for name,expected in startup['source'].items():
        actual=ref(base/'producer'/name)
        assert actual['sha256']==expected['sha256'] and actual['bytes']==expected['bytes']
    for name in ('stdout.txt','stderr.txt','startup.json','runtime-observation.json'):
        actual=ref(base/name);expected=next(o for o in finish['completed_objects'] if o['path']==name)
        assert (actual['bytes'],actual['sha256'])==(expected['bytes'],expected['sha256'])
    attempts.append({'name':base.name,'command':startup['command'],'utc':startup['utc'],'exit_code':finish['exit_code'],
        'elapsed_seconds':finish['elapsed_seconds'],'child_cpu_user_seconds':finish['child_cpu_user_seconds'],
        'child_cpu_system_seconds':finish['child_cpu_system_seconds'],'child_maxrss':finish['child_maxrss'],
        'startup':ref(base/'startup.json'),'stdout':ref(base/'stdout.txt'),'stderr':ref(base/'stderr.txt'),'finish':ref(base/'finish.json'),
        'runtime':ref(base/'runtime-observation.json'),'storage_before':{k:v for k,v in startup['storage'].items() if k!='inventory'},
        'storage_after':{k:finish[k] for k in ('git_free_after','data_free_after')}})
native=store/'native-functional-v1';report=json.loads((native/'metaquant.json').read_bytes())
assert report['status']=='pass' and report['scientific_gate'] is None and len(report['variants'])==4
current={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('src/ralup').rglob('*.py'))}
assert current==report['source_hashes']
native_start=json.loads((native/'startup.json').read_bytes())
cpu_start=json.loads((store/'full-cpu-regression-3/startup.json').read_bytes())
for name,digest in current.items():assert native_start['source'][name]['sha256']==digest==cpu_start['source'][name]['sha256']
work=[w for v in report['variants'] for w in v['work']+v['replay_work']]
summary={'executed_training_steps':len(work),'training_target_bytes':sum(w['stream'] for w in work),
    'training_backward_target_bytes':sum(w['backward'] for w in work),'extra_loss_target_bytes':sum(w['additional_loss'] for w in work),
    'readonly_functional_and_score_target_bytes':sum(16 for v in report['variants']),
    'total_LM_loss_target_exposures':sum(v['loss_target_exposures'] for v in report['variants']),
    'training_loss_executions':sum(len(w['metaquant_work']['loss_pass_events']) for w in work),
    'readonly_functional_loss_executions':4,'readonly_score_executions':4,
    'original_fixture_scalar_training_losses':report['original_mechanism']['training_losses'],
    'original_fixture_scalar_backwards':report['original_mechanism']['backward_passes'],
    'factor_forward_derivative_audit_gradient_executions':4,
    'classification_overlap':'ordinary/priming/fast/meta share execution identity; replay is separately executed and charged; extra read-only passes add loss but no backward',
    'scope':'native synthetic engineering only; not corpus bytes/scientific budget/performance evidence'}
index={'schema':'ralup.m10.completed_evidence.v1','base':'33c8352a13babde50303a4ed072c05befe83243f','attempts':attempts,
    'native_report':ref(native/'metaquant.json'),'native_contract':ref(Path('docs/harness/m10-native-contract.json')),
    'final_source_hashes':current,'native_actual_work':summary,'scientific_gate':None,
    'source_copy_status':'blocked_missing_grant','release_parity_claimed':False,'blocked_is_defeated_baseline':False,
    'CPU_regression_scope':'explicit cached-function availability override plus lazy subprocess bootstrap; physical MPS stays exclusively in native-functional-v1 and seed-isolation checks',
    'cost_scope':'each completed launcher interval includes child; sum is scoped invocation work, not task wall union or complete engineering/project cost',
    'completed_invocation_elapsed_seconds':sum(a['elapsed_seconds'] for a in attempts),
    'common_enclosing_engineering_and_project_cost':'unknown, not zero',
    'active_administration':'this index launcher is excluded; its finished stdout/finish receipt is a subsequent immutable observation'}
content=(json.dumps(index,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
Path('docs/harness/m10-evidence.json').write_bytes(content)
with (active/'index.json').open('xb') as stream:stream.write(content)
print(json.dumps({'attempts':len(attempts),'pointer_bytes':len(content),'pointer_sha256':hashlib.sha256(content).hexdigest(),'native_work':summary},sort_keys=True))
