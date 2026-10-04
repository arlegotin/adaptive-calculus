"""Administrative M04 receipt/pointer summary; no numerical experiment launches."""
import hashlib,json,shutil,sys
from pathlib import Path
root=Path.cwd();store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03');task=store/'m04-20261004'
names=['red','green01','additive-mutant','microbatch-mutant','matrix-json-red','covering','helper-exit','verify-cpu','verify-mps']
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
rows=[];wall=cpu=child=0.
for name in names:
    base=task/name;start=json.loads((base/'startup.json').read_bytes());finish=json.loads((base/'finish.json').read_bytes())
    stdout=(base/'stdout.txt').read_text()
    wall+=finish['elapsed_seconds'];cpu+=finish['cpu_user_seconds']+finish['cpu_system_seconds'];child+=finish['child_cpu_user_seconds']+finish['child_cpu_system_seconds']
    assert hashlib.sha256((base/'stdout.txt').read_bytes()).hexdigest()==finish['stdout_sha256']
    # Authenticate every exact pre-start producer object, not merely its label.
    for path,declared in start['source_inventory'].items():
        content=(base/'producer'/path).read_bytes();assert len(content)==declared['bytes'] and hashlib.sha256(content).hexdigest()==declared['sha256']
    row={'name':name,'startup':ref(base/'startup.json'),'finish':ref(base/'finish.json'),'stdout':ref(base/'stdout.txt'),'observed_helper_exit_code':finish['exit_code'],'module':start['module'],'arguments':start['arguments'],'cost':{k:finish[k] for k in ('elapsed_seconds','cpu_user_seconds','cpu_system_seconds','child_cpu_user_seconds','child_cpu_system_seconds','maxrss')},'stdout_summary':stdout.strip().splitlines()[-1]}
    if name=='covering':row['status_explanation']='165 tests passed in raw pytest stdout; original helper exited 1 because successful pytest ExitCode.OK IntEnum was misclassified. Original receipt stays unchanged; helper-only IntEnum regression later passed. No covering rerun, no source/bound relabel.'
    if name.startswith('verify-'):row['report']=ref(base/'report.json')
    rows.append(row)
scopes={name:{p:d['sha256'] for p,d in json.loads((task/name/'startup.json').read_bytes())['source_inventory'].items() if p.startswith(('src/ralup/','tests/'))} for name in ('covering','verify-cpu','verify-mps')}
assert scopes['covering']==scopes['verify-cpu']==scopes['verify-mps']
files=[ref(p) for p in sorted(task.rglob('*')) if p.is_file()]
forecast=256*1024**2;git_forecast=16*1024**2;git_free=shutil.disk_usage(root).free;data_free=shutil.disk_usage(store).free
assert git_free-git_forecast>=512*1024**2
assert sum(p.stat().st_size for p in store.rglob('*') if p.is_file())+forecast<=32*1024**3 and data_free-forecast>=64*1024**3
value={'schema':'m04.trs_evidence.v1','base_commit':'097a314','task_store':str(task),'store':ref(store/'store.json'),'attempts':rows,'complete_task_inventory_at_summary_start':files,
    'source_identity':{'product_test_hashes_same_for_covering_and_both_cli':True,'product_test_sha256':scopes['covering']},
    'cost':{'complete_recorded_nine_attempts_wall_seconds':wall,'self_cpu_seconds':cpu,'child_cpu_seconds':child,'scope':'these nine complete helper/module processes only; nested child wall already inside helper wall and not added; peak RSS macOS bytes is per-process, not sustained/MPS capacity','unknown':'earlier task reading/editing/inspection, outer tool overhead, surrounding project work; not zero or a complete project total'},
    'storage':{'git_free_before_summary':git_free,'data_free_before_summary':data_free,'task_existing_logical_bytes':sum(r['bytes'] for r in files),'remaining_task_forecast_bytes':forecast,'source_config_pointer_git_transaction_forecast_bytes':git_forecast,'root_store_ceiling_bytes':32*1024**3,'data_free_floor_bytes':64*1024**3,'future_small_source_git_reserve_bytes':512*1024**2,'historical_frozen_artifact_git_guard_unchanged':True,'existing_artifacts_deleted_moved_or_overwritten':False},
    'producing_summary_startup':ref(task/'inventory/startup.json'),
    'claims':{'protected_data_used':False,'scientific_gate':None,'author_GPL_release_executed_or_copied':False,'backbone_tuning_winner_selected':False,'supported_families':['AdamW','Muon','Muon+'],'unsupported_families':['SOAP','Madam','arbitrary unregistered BaseUpdateRule'],'unsupported_is_defeated_baseline':False}}
output=root/'docs/harness/m04-evidence.json'
with output.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
assert output.stat().st_size<git_forecast and shutil.disk_usage(root).free>=512*1024**2
print(json.dumps({'evidence':ref(output),'git_free_after_summary':shutil.disk_usage(root).free,'data_free_after_summary':shutil.disk_usage(store).free,'scope':'administrative authenticated metadata only; original logs and statuses remain immutable'}))
