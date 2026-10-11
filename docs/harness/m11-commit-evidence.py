"""Validate and commit the bounded M11 evidence/handoff transaction."""
import ast,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from ralup.policy.checks import _ref,_write
base=Path(os.environ['M11_ATTEMPT_DIR']);started=time.perf_counter()
files=sorted([str(p) for p in Path('docs/harness').glob('m11-*') if p.suffix in ('.py','.json')]+['progress.json'])
assert sum(Path(p).stat().st_size for p in files)<=16*1024**2
free_before=shutil.disk_usage(Path.cwd()).free;assert free_before>=512*1024**2+16*1024**2
assert subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip()==''
counts={'python_ast':0,'json':0,'authenticated_index_references':0}
for name in files:
 p=Path(name)
 if p.suffix=='.py':ast.parse(p.read_text());counts['python_ast']+=1
 elif p.suffix=='.json':json.loads(p.read_bytes());counts['json']+=1
index=json.loads(Path('docs/harness/m11-evidence.json').read_bytes())
assert index['status']=='needs_revision_pending_independent_review'
for attempt in index['attempts']:
 for ref in attempt['files'].values():
  assert _ref(Path(ref['path']))['sha256']==ref['sha256'];counts['authenticated_index_references']+=1
progress=json.loads(Path('progress.json').read_bytes())
assert progress['M11']['selected_controller'] is None and progress['M11']['WP06']=='NEEDS_REVISION'
commands=[]
for command in [['git','diff','--check'],['git','add','--',*files],['git','commit','-m','docs: record M11 measured horizon reversal and revision gates']]:
 result=subprocess.run(command,text=True,capture_output=True);commands.append({'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr});assert result.returncode==0
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();status=subprocess.check_output(['git','status','--short'],text=True)
free_after=shutil.disk_usage(Path.cwd()).free;assert free_after>=512*1024**2
report=Path('.superpowers/sdd/2026-10-03-ralup/M11-report.md')
with report.open('a') as stream:
 stream.write('\n## Final evidence transaction and review handoff\n\nEvidence commit `'+commit+'` records the helper/source-design/index/progress handoff. Static checks: '+json.dumps(counts,sort_keys=True)+'. `git diff --check` passed; all indexed references authenticated; scientific status remains NEEDS_REVISION. No fresh native or full-suite rerun. Source input '+str(sum(Path(p).stat().st_size for p in files))+'B; Gitfree '+str(free_before)+'→'+str(free_after)+'B. Exact transaction commands/output and final durable report: `'+str(base)+'`. Final implementation review remains c2552c0722110d0c7ed5e5e0e28df11d3d680eda..'+commit+'. Working-tree status after commit: '+repr(status)+'.\n')
shutil.copyfile(report,base/'M11-report.md')
_write(base/'report.json',{'status':'evidence_committed_needs_revision_pending_independent_review','commit':commit,'base':'c2552c0722110d0c7ed5e5e0e28df11d3d680eda','commands':commands,'checks':counts,'source_input_bytes':sum(Path(p).stat().st_size for p in files),'git_free_before':free_before,'git_free_after':free_after,'working_tree_status':status,'authoritative_report':_ref(base/'M11-report.md'),'elapsed_seconds':time.perf_counter()-started,'scientific_status':'WP06_NEEDS_REVISION','new_optimizer_updates':0,'new_query_targets':0})
print(json.dumps({'commit':commit,'checks':counts,'report':_ref(base/'report.json'),'authoritative_report':_ref(base/'M11-report.md'),'working_tree_status':status}),flush=True)
