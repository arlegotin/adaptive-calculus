"""Small local M07 closing commit with native output and storage observations."""
import json,os,shutil,subprocess,time
from pathlib import Path
root=Path.cwd();start=time.perf_counter()
all_paths=subprocess.check_output(['git','ls-files','--modified','--others','--exclude-standard','-z']).decode().split('\0')
paths=sorted(set(p for p in all_paths if p and (p.startswith('docs/harness/m07') or p in {'src/ralup/search.py','tests/unit/test_headroom_selection.py','tests/unit/test_m07_supervision.py','tests/unit/test_m07_view_helper.py'})))
assert paths and set(p for p in all_paths if p)==set(paths),'Unrelated working tree change: do not absorb it'
forecast=sum((root/p).stat().st_size for p in paths);before=shutil.disk_usage(root).free
assert forecast<=16*1024**2 and before-16*1024**2>=512*1024**2
commands=[['git','add','--',*paths],['git','diff','--cached','--check'],
    ['git','commit','-m','fix: validate recipe provenance and record M07 disposition'],['git','rev-parse','HEAD'],['git','status','--short']]
results=[]
for command in commands:
    result=subprocess.run(command,capture_output=True,text=True)
    results.append({'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    print(result.stdout,end='');print(result.stderr,end='')
    if result.returncode:break
after=shutil.disk_usage(root).free;assert after>=512*1024**2
record={'selected_paths':paths,'selected_bytes':forecast,'git_free_before':before,'git_free_after':after,
    'complete_seconds':time.perf_counter()-start,'commands':results,
    'task_status':'Initial-selection disposition complete pending fresh task review; B0 null / WP04 inconclusive; dated exploratory policy-development amendment only'}
with (Path(os.environ['M07_ATTEMPT_DIR'])/'commit.json').open('x') as f:json.dump(record,f,sort_keys=True,indent=2)
raise SystemExit(results[-1]['exit_code'])
