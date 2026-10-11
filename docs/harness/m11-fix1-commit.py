"""Bounded I1 evidence/source commit after focused archived selector verification."""
import ast,json,os,shutil,subprocess,time
from pathlib import Path
from ralup.policy.checks import _ref,_write
base=Path(os.environ['M11_ATTEMPT_DIR']);store=base.parent;started=time.perf_counter()
fixbase='e25a96c4c316129abc193f3715493a3fdbfe08ca'
files=['src/ralup/meta/selection.py','tests/unit/test_es.py','tests/unit/test_meta_selection.py','docs/harness/m11-selection-authority.md','docs/harness/m11-fix1-verification-design.json','docs/harness/m11-fix1-commit.py','docs/harness/m11-fix1-commit-design.json','docs/harness/m11-index.py','docs/harness/m11-evidence.json','progress.json']
size=sum(Path(p).stat().st_size for p in files);assert size<=16*1024**2
before=shutil.disk_usage(Path.cwd()).free;assert before>=512*1024**2+16*1024**2
assert subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip()==''
changed=subprocess.check_output(['git','diff','--name-only',fixbase,'--','src'],text=True).splitlines();assert changed==['src/ralup/meta/selection.py']
verify=store/'fix1-verify-selector-v1';finish=json.loads((verify/'finish.json').read_bytes());startup=json.loads((verify/'startup.json').read_bytes());assert finish['exit_code']==0
assert '27 passed' in (verify/'stdout.txt').read_text() and not (verify/'stderr.txt').read_bytes()
for p in files[:3]:assert _ref(Path(p))['sha256']==startup['source'][p]['sha256']
checks={'python_ast':0,'json':0,'indexed_refs':0}
for name in files:
 p=Path(name)
 if p.suffix=='.py':ast.parse(p.read_text());checks['python_ast']+=1
 if p.suffix=='.json':json.loads(p.read_bytes());checks['json']+=1
index=json.loads(Path('docs/harness/m11-evidence.json').read_bytes())
for attempt in index['attempts']:
 for ref in attempt['files'].values():assert _ref(Path(ref['path']))['sha256']==ref['sha256'];checks['indexed_refs']+=1
archived=[]
for source in sorted(Path('/private/tmp').glob('m11-fix1-*.txt')):
 target=base/'local-cpu-stdout'/source.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target);archived.append({'source':str(source),'archive':_ref(target),'scope':'Local stdout only; original whole-process/source startup not retroactively claimed. Edit/tool/enclosing cost unknown.'})
for name in ('M11-review.md','M11-fix1-findings.md'):
 shutil.copyfile(Path('.superpowers/sdd/2026-10-03-ralup')/name,base/name)
normalizer=store/'native-reference-v1'/'final-reference-normalizer.json'
assert _ref(normalizer)['sha256']=='d75faa2001817b0d2add0e0abe1bea48530eb37196f925c2ce00bbc696f53c9e'
applicability={'schema':'ralup.m11.fix1_applicability.v1','fix_base':fixbase,'changed_runtime_files':changed,'selector_source':_ref(Path(changed[0])),'native_assessment_report':_ref(store/'native-final-calibration-v1'/'report.json'),'normalizer_data':_ref(normalizer),'scope':'Only future selector admission changed. No rollout/model/optimizer/quantizer/feature/history/normalizer math changes; strict complete-state runtime identity changes, no old full-state import. Historical35outcomes/normalizer DATA remain original producer evidence. No final scientific campaign or selection authority exists.','WP06':'NEEDS_REVISION','new_optimizer_updates':0,'new_query_targets':0}
_write(base/'applicability.json',applicability)
commands=[]
for command in [['git','diff','--check'],['git','add','--',*files],['git','commit','-m','fix: require eligible training and authenticated selection endpoints']]:
 result=subprocess.run(command,text=True,capture_output=True);commands.append({'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr});assert result.returncode==0
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();status=subprocess.check_output(['git','status','--short'],text=True)
after=shutil.disk_usage(Path.cwd()).free;assert after>=512*1024**2
report=Path('.superpowers/sdd/2026-10-03-ralup/M11-report.md')
with report.open('a') as stream:stream.write('\nFix1 final commit `'+commit+'`; scoped review range '+fixbase+'..'+commit+'. Static checks '+json.dumps(checks,sort_keys=True)+', diffcheck passed, source/test hashes equal archived27-pass producer. Source transaction '+str(size)+'B, Gitfree'+str(before)+'→'+str(after)+'B. Durable final report/commands/outputs/applicability/local RED/GREEN stdout at `'+str(base)+'`. Working tree after commit: '+repr(status)+'. All scientific gates and exact historical scopes remain unchanged.\n')
shutil.copyfile(report,base/'M11-report.md')
_write(base/'report.json',{'status':'fix1_committed_pending_scoped_rereview','fix_base':fixbase,'commit':commit,'commands':commands,'checks':checks,'source_input_bytes':size,'git_free_before':before,'git_free_after':after,'working_tree_status':status,'verification':_ref(verify/'finish.json'),'historical_local_cpu_outputs':archived,'applicability':_ref(base/'applicability.json'),'authoritative_report':_ref(base/'M11-report.md'),'elapsed_seconds':time.perf_counter()-started,'scientific_status':'WP06_NEEDS_REVISION','new_optimizer_updates':0,'new_query_targets':0})
print(json.dumps({'commit':commit,'report':_ref(base/'report.json'),'authoritative_report':_ref(base/'M11-report.md'),'working_tree_status':status}),flush=True)
