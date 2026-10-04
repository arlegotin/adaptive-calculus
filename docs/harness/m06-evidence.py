"""Consolidate completed immutable M06 attempts; omit this active stream."""
import hashlib,json,os
from pathlib import Path
root=Path.cwd();store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m06-20261004')
output=Path(os.environ['M06_ATTEMPT_DIR'])/'evidence.json'
def ref(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
rows=[]
for base in sorted(store.iterdir()):
    if base==output.parent or not (base/'finish.json').exists():continue
    f=json.loads((base/'finish.json').read_bytes());s=json.loads((base/'startup.json').read_bytes())
    for obj in f['completed_objects']:
        p=base/obj['path'];assert p.stat().st_size==obj['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==obj['sha256']
    rows.append({'name':base.name,'command':s['command'],'exit_code':f['exit_code'],
        'startup':ref(base/'startup.json'),'stdout':ref(base/'stdout.txt'),'stderr':ref(base/'stderr.txt'),'finish':ref(base/'finish.json'),
        'scoped_complete_cost':{k:v for k,v in f.items() if 'seconds' in k},'scope':'complete launcher/child only; enclosing/editing/tool costs unknown'})
final=json.loads((store/'final-verification/startup.json').read_bytes())
for p in sorted(root.glob('src/ralup/**/*.py')):
    assert hashlib.sha256(p.read_bytes()).hexdigest()==final['sources'][str(p.relative_to(root))]['sha256']
reports=[ref(p) for base in sorted(store.iterdir()) for p in sorted(base.glob('*.json')) if p.name in ('results.json','cpu-results.json','mps-results.json','diagnostic.json','checks.json')]
value={'task':'M06','task_base':'b0b471e','attempts':rows,'reports':reports,'source_stable_covering':'262 passed in40.39s; task plus named affected boundaries; CPU/MPS public scale-noise gates pass',
    'active_administrative_stdout':'omitted; completed consolidator streams/finish published by subsequent pointer',
    'source_identity':'each startup archives exact product/test/helper/lock/F02 bytes; final source verified against final-verification startup',
    'history':'RED/mutants and original CPU-exp/MPS-cache validator failure retained; no old rewrite or tolerance adjustment',
    'cost_scope':'known scoped complete attempts including failures; no overlapping pass-classification sum or whole-project cost total',
    'claim_scope':'bounded clean equations, exact same-backend learned-cache/state replay; cross-backend restore unsupported; no protected/scientific gate'}
with output.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'evidence':ref(output)}))
