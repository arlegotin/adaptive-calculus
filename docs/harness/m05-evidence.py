"""Consolidate only completed attempts; this active stream is omitted."""
import hashlib,json,os,sys
from pathlib import Path
store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m05-20261004');output=Path(os.environ['M05_ATTEMPT_DIR'])/'evidence.json'
def ref(path):return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
rows=[]
for base in sorted(store.iterdir()):
    if base==output.parent or not (base/'finish.json').exists():continue
    finish=json.loads((base/'finish.json').read_bytes());startup=json.loads((base/'startup.json').read_bytes())
    for obj in finish['completed_objects']:
        p=base/obj['path'];assert p.stat().st_size==obj['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==obj['sha256']
    rows.append({'name':base.name,'command':startup['command'],'exit_code':finish['exit_code'],'startup':ref(base/'startup.json'),'stdout':ref(base/'stdout.txt'),'stderr':ref(base/'stderr.txt'),'finish':ref(base/'finish.json'),
        'scoped_cost':{k:v for k,v in finish.items() if 'seconds' in k},'scope':'complete launcher/child only; enclosing costs unknown'})
reports=[ref(p) for base in store.iterdir() for p in base.glob('results.json')]
value={'task':'M05','task_base':'215c1d6','attempts':rows,'reports':reports,'active_administrative_stdout':'omitted; publish exact complete stream identity in a subsequent pointer','cost_scope':'scoped complete attempts, including expected red and negative failures; not a whole-project cost total','source_identity':'each startup maps exact copied producing bytes; older cold-parent attempts remain historical; final covering includes corrected warm-parent initialization branch'}
with output.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'evidence':ref(output)}))
