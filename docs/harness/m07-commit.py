"""Small measured local M07 commit transaction; task remains scientifically open."""
import json,os,shutil,subprocess,time
from pathlib import Path
root=Path.cwd();start=time.perf_counter()
paths=[*root.glob('docs/harness/m07*'),*root.glob('configs/m07*'),root/'configs/proxy.headroom.json']
paths += [root/p for p in ('src/ralup/__main__.py','src/ralup/calibrate.py','src/ralup/evaluate.py',
    'src/ralup/state.py','src/ralup/train.py','src/ralup/search.py','src/ralup/warmstart.py',
    'tests/unit/test_headroom_selection.py','tests/unit/test_derived_warmstart.py','tests/integration/test_scoring.py')]
paths=sorted({str(p.relative_to(root)) for p in paths if p.is_file()})
forecast=sum((root/p).stat().st_size for p in paths);before=shutil.disk_usage(root).free
assert forecast<=16*1024**2 and before-16*1024**2>=512*1024**2
commands=[['git','add','--',*paths],['git','diff','--cached','--check'],
    ['git','commit','-m','feat: add development headroom and backbone selection'],['git','rev-parse','HEAD']]
results=[]
for command in commands:
    result=subprocess.run(command,capture_output=True,text=True)
    results.append({'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    print(result.stdout,end='');print(result.stderr,end='')
    if result.returncode:break
after=shutil.disk_usage(root).free;assert after>=512*1024**2
record={'selected_paths':paths,'selected_bytes':forecast,'git_free_before':before,'git_free_after':after,
    'complete_seconds':time.perf_counter()-start,'commands':results,
    'task_status':'OPEN; intermediate reviewed-by-tests implementation, actual ordinary coverage/selection remains; no B0/final WP04 claim'}
with (Path(os.environ['M07_ATTEMPT_DIR'])/'commit.json').open('x') as f:json.dump(record,f,sort_keys=True,indent=2)
raise SystemExit(results[-1]['exit_code'])
