"""Finite root-authorized commit/closure; no Data writes or numerical work."""
from datetime import datetime,timezone
import importlib.util,json,os,shutil,subprocess,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('capture',Path(__file__).with_name('e07-capture.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=Path.cwd();base=m.GIT_COMPLETION;out=base/'closure-v1'
assert not out.exists() and not (base/'completion.json').exists()
start=time.perf_counter();utc=datetime.now(timezone.utc).isoformat()
data_before=m.inventory(m.STORE);git_before=m.inventory(base);e06=m.inventory(base.parent/'e06-v1')
allowance=1024**2
assert git_before['allocated_bytes']+allowance<=160*m.MIB
assert shutil.disk_usage(root).free-allowance-16*m.MIB>=m.GIB
assert max(data_before['logical_bytes']+git_before['logical_bytes']+e06['logical_bytes'],data_before['allocated_bytes']+git_before['allocated_bytes']+e06['allocated_bytes'])+allowance<=32*m.GIB
out.mkdir();(out/'producer.py').write_bytes(Path(__file__).read_bytes())
startup={'utc':utc,'forecast_physical_bytes':allowance,'separate_source_git_allowance_bytes':16*m.MIB,
 'data_before':data_before,'git_before':git_before,'raw_data_free_before':shutil.disk_usage(m.STORE).free,
 'git_free_before':shutil.disk_usage(root).free,'producer':m.ref(out/'producer.py'),'operation':'git diff --check; explicit git add; required local commit; exact closure refs',
 'zero_owned_Data_writes':True,'raw_Data_floor_exception':'finite final administration only; no all-bounds PASS',
 'head_before':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'final_admin_finish':m.ref(base/'final-admin-v1/finish.json')}
(out/'startup.json').write_bytes(m.encoded(startup))
subprocess.run(['git','diff','--check'],check=True)
paths=['README.md','decisions.md','progress.json','src/ralup/report.py','src/ralup/__main__.py','reproduce.sh','tests/integration/test_report_reproduction.py',
 'docs/superpowers/plans/2026-10-03-ralup.md','docs/superpowers/plans/2026-10-03-ralup-evaluation.md']
paths+=sorted(str(p.relative_to(root)) for p in (root/'docs/harness').glob('e07*') if p.is_file())
subprocess.run(['git','add','--',*paths],check=True)
with (out/'commit-stdout.txt').open('x') as stdout,(out/'commit-stderr.txt').open('x') as stderr:
 result=subprocess.run(['git','commit','-m','feat: add reproducible research report and outcome ledger'],stdout=stdout,stderr=stderr)
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
status=subprocess.check_output(['git','status','--short'],text=True)
(out/'git-status.txt').write_text(status)
after=m.inventory(m.STORE);git_after=m.inventory(base)
refs={}
for name,p in [('evidence_index',root/'docs/harness/e07-evidence.json'),('full_report',root/'.superpowers/sdd/2026-10-03-ralup/E07-report.md'),
 ('durable_full_report',base/'final-admin-v1/history-delta/E07-report.md'),('final_admin_finish',base/'final-admin-v1/finish.json'),
 ('history_delta',base/'final-admin-v1/history-delta.json'),('handoff_mapping',base/'final-admin-v1/handoff-mapping.json'),
 ('native_storage_supplement',base/'report-v3/native-storage-supplement.json'),('research_report',base/'report-v3/report/report.json'),
 ('report_finish',base/'report-v3/finish.json'),('plots_finish',base/'plots-v2/finish.json')]:refs[name]=m.ref(p)
assert refs['full_report']['sha256']==refs['durable_full_report']['sha256']
resource_ok=(after==data_before and git_after['allocated_bytes']-git_before['allocated_bytes']+65536<=allowance
 and git_after['allocated_bytes']+65536<=160*m.MIB and shutil.disk_usage(root).free-16*m.MIB>=m.GIB
 and max(after['logical_bytes']+git_after['logical_bytes']+e06['logical_bytes'],after['allocated_bytes']+git_after['allocated_bytes']+e06['allocated_bytes'])+65536<=32*m.GIB)
finish={'schema':'ralup.e07_completion.v1','utc':datetime.now(timezone.utc).isoformat(),'commit':head,'commit_exit_code':result.returncode,
 'git_status':status,'startup':m.ref(out/'startup.json'),'producer':m.ref(out/'producer.py'),'refs':refs,
 'elapsed_seconds':time.perf_counter()-start,'cost_scope':'closure including inventory and commit only; editing/review/project costs unknown',
 'Data_inventory_unchanged':after==data_before,'Data_after':after,'Git_completion_after_before_final_small_receipt':git_after,
 'remaining_receipt_reserved_bytes':65536,'raw_data_free_after':shutil.disk_usage(m.STORE).free,'git_free_after':shutil.disk_usage(root).free,
 **m.route_disposition(resource_ok,shutil.disk_usage(m.STORE).free>=64*m.GIB,True,result.returncode),
 'late_root_delta_obligation':'Preserve later E07 reviews/fixes, root whole-branch review, final chronological rulings and finishing before only-own-plan cleanup; current capture is not final root record collection.'}
(base/'completion.json').write_bytes(m.encoded(finish));print(json.dumps(finish))
assert resource_ok and result.returncode==0 and not status
