"""Publish completed consolidation identities; omit active own stdout."""
import hashlib,json,os,shutil
from pathlib import Path
root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m06-20261004/consolidate')
def ref(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
assert json.loads((base/'finish.json').read_bytes())['exit_code']==0
value={'task':'M06','evidence':ref(base/'evidence.json'),'completed_consolidation':{n:ref(base/n) for n in ('startup.json','stdout.txt','stderr.txt','finish.json')},
    'active_publication_stdout':'omitted; subsequent persistent task report names completed publication/commit refs',
    'scope':'final source-stable task+affected-boundary covering; exact same-backend learned physical cache, no scientific claims',
    'history':'RED, mutants, original MPS validator failure and observed exp diagnostic immutable; no old moves/deletes/overwrites'}
forecast=16*1024**2;before=shutil.disk_usage(root).free;assert before-forecast>=512*1024**2
with (root/'docs/harness/m06-evidence.json').open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
after=shutil.disk_usage(root).free;assert after>=512*1024**2 and before-after<=forecast
print(json.dumps({'pointer':ref(root/'docs/harness/m06-evidence.json'),'git_free_before':before,'git_free_after':after,'forecast_bytes':forecast}))
