"""Authorized small local M09 commit; capture every Git command stream."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path.cwd();started=time.perf_counter();index=json.loads((root/'docs/harness/m09-controller-evidence.json').read_bytes())
for name,digest in index['source_identity'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='feat/ralup-execution'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==index['task_base']
paths=sorted(p for p in subprocess.check_output(['git','ls-files','--modified','--others','--exclude-standard','-z']).decode().split('\0') if p)
allowed={'src/ralup/__main__.py','src/ralup/train.py','src/ralup/state.py','src/ralup/policy/checks.py','src/ralup/policy/controllers.py','src/ralup/policy/adapter.py','tests/unit/test_controllers.py','tests/integration/test_policy_identity.py'}
assert paths and all(p in allowed or p.startswith('docs/harness/m09') for p in paths)
selected_bytes=sum((root/p).stat().st_size for p in paths);before=shutil.disk_usage(root).free
assert selected_bytes<=16*1024**2 and before-16*1024**2>=512*1024**2
base=Path(os.environ['M09_ATTEMPT_DIR']);report=root/'.superpowers/sdd/2026-10-03-ralup/M09-report.md';(base/'report-before-commit.md').write_bytes(report.read_bytes())
commands=[['git','diff','--check'],['git','add','--',*paths],['git','diff','--cached','--check'],['git','commit','-m','feat: add RALUP and information-matched policy controls'],['git','rev-parse','HEAD'],['git','status','--short']]
results=[]
for command in commands:
 r=subprocess.run(command,capture_output=True,text=True);results.append({'command':command,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr});print(r.stdout,end='');print(r.stderr,end='')
 if r.returncode:break
after=shutil.disk_usage(root).free;assert after>=512*1024**2
record={'selected_paths':paths,'selected_bytes':selected_bytes,'git_free_before':before,'git_free_after':after,'complete_seconds':time.perf_counter()-started,'commands':results,'source_matches_terminal_native':True,'report_before_commit_sha256':hashlib.sha256(report.read_bytes()).hexdigest()}
with (base/'commit.json').open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
raise SystemExit(results[-1]['exit_code'])
