"""Separate metadata-guard applicability; no numerical replay or source relabel."""
import difflib,json,os
from pathlib import Path
from datetime import datetime,timezone
from ralup.search import source_identity
from ralup.records import sha256_file
from ralup.calibrate import _write
root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
old=json.loads((base/'applicability-v4/admission-applicability-256.json').read_bytes())['source_identity'];new=source_identity()
changed={k:{'before':old.get(k),'after':new.get(k)} for k in sorted(set(old)|set(new)) if old.get(k)!=new.get(k)}
assert set(changed)=={'src/ralup/search.py'}
checks=base/'green-recipe-metadata/finish.json';assert json.loads(checks.read_bytes())['exit_code']==0
correction=base/'authenticate-recipe-metadata-v2/recipe-correction.json'
assert json.loads(correction.read_bytes())['source_identity']==old
for horizon in (16,32,128,256):
    sources=([base/'native-ordinary-geometry/admission.json',base/'native-quantizer-geometry/admission.json'] if horizon==16 else [base/f'applicability-v4/admission-applicability-{horizon}.json'])
    scopes=[]
    for source in sources:
        before=json.loads(source.read_bytes());assert before['source_identity']==old and before['status']=='pass'
        for scope in before['scopes']:
            if scope not in scopes:scopes.append(scope)
    report={'status':'pass','source_identity':new,'scopes':scopes,'maximum_steps':horizon,
        'record_type':'separate_metadata_guard_applicability_not_new_numerical_replay',
        'created_utc':datetime.now(timezone.utc).isoformat(),'source_evidence':[ref(p) for p in sources],'changed_modules':changed,
        'exact_diffs':{name:''.join(difflib.unified_diff((base/'authenticate-recipe-metadata-v2/producer'/name).read_text().splitlines(keepends=True),(root/name).read_text().splitlines(keepends=True),fromfile='old/'+name,tofile='new/'+name)) for name in changed},
        'numerical_applicability':'model/quantizer/optimizer/train/state/evaluate bytes unchanged; correctly labelled configs execute the identical named recipe; source-scoped primitive and paired proofs remain historical',
        'execution_delta':'reject contradictory grouping/normalization/moment-coordinate declarations before public construction or final selection; disposition retains historical evidence; missing optional annotations remain legacy-compatible',
        'annotation_correction':ref(correction),'checks':ref(checks),'stdout':ref(base/'green-recipe-metadata/stdout.txt'),
        'strict_restore':'old whole-runtime states incompatible; fresh current states or authenticated typed/ordinary model-only import, no full-state waiver',
        'limits':'only listed exact model/method/quantizer/rotation/backward/packing and bounded horizon; no SOAP/Madam/QuEST/Hadamard horizon widening',
        'new_training_steps':0,'new_query_targets':0}
    p=Path(os.environ['M07_ATTEMPT_DIR'])/f'admission-applicability-{horizon}.json';_write(p,report);print(json.dumps(ref(p)))
