"""New FIX1 supplement; authenticate completed tests and omit active stdout."""
import hashlib,json,os,shutil
from pathlib import Path
root=Path.cwd();store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m06-fix1-20261011')
def ref(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
attempts=[]
for name,expected in (('red',1),('green',0)):
    base=store/name;startup=json.loads((base/'startup.json').read_bytes());finish=json.loads((base/'finish.json').read_bytes())
    assert finish['exit_code']==expected
    for entry in finish['completed_objects']:
        path=base/entry['path'];assert ref(path)['sha256']==entry['sha256'] and path.stat().st_size==entry['bytes']
    attempts.append({'name':name,'command':startup['command'],'exit_code':finish['exit_code'],
        'objects':{n:ref(base/n) for n in ('startup.json','runtime-observation.json','stdout.txt','stderr.txt','finish.json')},
        'complete_scoped_cost':{k:v for k,v in finish.items() if 'seconds' in k},
        'scope':'current-runtime CPU engineering regression; classifications may overlap; enclosing/editing/tool costs unknown'})
green=json.loads((store/'green/startup.json').read_bytes())
for path in sorted(root.glob('src/ralup/**/*.py')):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==green['source'][str(path.relative_to(root))]['sha256']
assert hashlib.sha256((root/'tests/unit/test_learned_scale.py').read_bytes()).hexdigest()==green['source']['tests/unit/test_learned_scale.py']['sha256']
value={'task':'M06','round':'FIX1','fix_base':'439eef1','task_base':'b0b471e','attempts':attempts,
    'result':'3 expected RED failures; 18 focused GREEN passes in 3.25s; no warnings/skips/stderr',
    'fix':'Bind complete prepared method-owned Parameter inventory/identities before backward and verify them before successful publication; preserve existing owners and first-step construction inside original rollback transaction',
    'runtime':'11 October 2026 observations macOS 26.7.1 arm64; historical F02/4 October MPS receipts are not current native admission; no old-runtime restore or new scientific/native claim',
    'source_identity':'all current product bytes and affected test authenticated against GREEN startup; fixed requirements.lock preserved',
    'historical_evidence':'docs/harness/m06-evidence.json and all earlier attempts unchanged; research cutoff 3 October unchanged',
    'active_publication_stdout':'omitted; exact completed publication/commit stream and finish identities appear in subsequent M06-report.md FIX1 appendix'}
forecast=16*1024**2;before=shutil.disk_usage(root).free;assert before-forecast>=512*1024**2
target=root/'docs/harness/m06-evidence-fix1.json'
with target.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
after=shutil.disk_usage(root).free;assert after>=512*1024**2 and before-after<=forecast
print(json.dumps({'pointer':ref(target),'git_free_before':before,'git_free_after':after,'forecast_bytes':forecast}))
