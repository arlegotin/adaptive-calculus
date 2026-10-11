"""Separate post-derived-parent source applicability, no replay or full-state waiver."""
import json,os,difflib
from pathlib import Path
from datetime import datetime,timezone
from ralup.search import source_identity
from ralup.records import sha256_file
from ralup.calibrate import _write
root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
old=json.loads((base/'applicability-v2/admission-applicability-256.json').read_bytes())['source_identity'];new=source_identity()
changed={k:{'before':old.get(k),'after':new.get(k)} for k in sorted(set(old)|set(new)) if old.get(k)!=new.get(k)}
assert set(changed)=={'src/ralup/calibrate.py','src/ralup/search.py','src/ralup/warmstart.py'}
prior=base/'applicability-v2/producer/src/ralup/calibrate.py';current=root/'src/ralup/calibrate.py'
marker="    if partition not in PARTITIONS or payload['configuration'].get('source_partition') != partition:"
assert prior.read_text().split(marker)[1]==current.read_text().split(marker)[1]
proof=base/'native-derived-parent/import-results.json';assert json.loads(proof.read_bytes())['status']=='pass'
checks=base/'green-derived-final/finish.json';assert json.loads(checks.read_bytes())['exit_code']==0
for horizon in (32,128,256):
    source=base/f'applicability-v2/admission-applicability-{horizon}.json';before=json.loads(source.read_bytes())
    report={'status':'pass','source_identity':new,'scopes':before['scopes'],'maximum_steps':horizon,
        'record_type':'source_applicability_analysis_with_separate_new_derived_import_proof',
        'created_utc':datetime.now(timezone.utc).isoformat(),'source_evidence':ref(source),'changed_modules':changed,
        'exact_diffs':{name:''.join(difflib.unified_diff((base/'applicability-v2/producer'/name).read_text().splitlines(keepends=True) if name in old else [],
            (root/name).read_text().splitlines(keepends=True),fromfile='old/'+name,tofile='new/'+name)) for name in changed},
        'unchanged_math':'all model/train/optimizer/quantizer/evaluate/state modules unchanged; original private model-only body byte-identical, original ordinary role/cost guard retained',
        'added_scope':'typed derived-record authentication and model-only/reset import only; no numerical horizon or method scope expansion',
        'native_import_proof':ref(proof),'native_import_complete_attempt':ref(base/'native-derived-parent/finish.json'),
        'checks':ref(checks),'stdout':ref(base/'green-derived-final/stdout.txt'),
        'strict_restore':'fresh current-source states or explicit tested model-only importer; old full-state restore remains incompatible',
        'new_training_steps':0,'new_query_targets':0}
    p=Path(os.environ['M07_ATTEMPT_DIR'])/f'admission-applicability-{horizon}.json';_write(p,report);print(json.dumps(ref(p)))
