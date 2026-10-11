"""Query-boundary control applicability; preserve interrupted old producer unchanged."""
import json,os,difflib
from pathlib import Path
from datetime import datetime,timezone
from ralup.search import source_identity
from ralup.records import sha256_file
from ralup.calibrate import _write
root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
old=json.loads((base/'applicability-v3/admission-applicability-256.json').read_bytes())['source_identity'];new=source_identity()
changed={k:{'before':old.get(k),'after':new.get(k)} for k in sorted(set(old)|set(new)) if old.get(k)!=new.get(k)}
assert set(changed)=={'src/ralup/evaluate.py','src/ralup/search.py'}
checks=base/'green-public-query-cap/finish.json';assert json.loads(checks.read_bytes())['exit_code']==0
for horizon in (32,128,256):
    sources=[base/f'applicability-v3/admission-applicability-{horizon}.json']
    if horizon==256:sources.append(base/'native-meta-parent/admission.json')
    scopes=[]
    for source in sources:
        before=json.loads(source.read_bytes());assert before['source_identity']==old and before['status']=='pass'
        for scope in before['scopes']:
            if scope not in scopes:scopes.append(scope)
    report={'status':'pass','source_identity':new,'scopes':scopes,'maximum_steps':horizon,
        'record_type':'separate_execution_control_applicability_not_new_numerical_replay',
        'created_utc':datetime.now(timezone.utc).isoformat(),'source_evidence':[ref(p) for p in sources],'changed_modules':changed,
        'exact_diffs':{name:''.join(difflib.unified_diff((base/'applicability-v3/producer'/name).read_text().splitlines(keepends=True),
            (root/name).read_text().splitlines(keepends=True),fromfile='old/'+name,tofile='new/'+name)) for name in changed},
        'numerical_applicability':'model/quantizer/optimizer/train/state bytes unchanged; evaluator retains identical context construction/forward/host fsum arithmetic, optional progress metadata and callbacks added; CPU exact complete scores/read-only owners tested',
        'execution_delta':'callback before/after contexts and after each completed block; public search enforces finite whole envelope there and charges known completed prefix; uninstrumented in-flight failure remains explicitly unknown; capped partial score never enters curve',
        'limits':'callback observes block boundaries, not a hard interrupt inside one operation; future helper must use callbacks and external monotonic supervision where needed; old query overrun remains noncompliant',
        'checks':ref(checks),'stdout':ref(base/'green-public-query-cap/stdout.txt'),
        'original_interrupted_attempt':ref(base/'query-ordinary-views/finish.json'),
        'supervision':[ref(base/'query-ordinary-views-cap/observation.json'),ref(base/'query-ordinary-views-cap-v2/observation.json')],
        'strict_restore':'old whole-runtime states incompatible; fresh current states or authenticated typed/ordinary model-only import, no old full-state waiver',
        'new_training_steps':0,'new_query_targets':0}
    p=Path(os.environ['M07_ATTEMPT_DIR'])/f'admission-applicability-{horizon}.json';_write(p,report);print(json.dumps(ref(p)))
