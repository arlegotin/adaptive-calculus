"""Versioned valid-state numerical applicability, never a producer relabel."""
import json,difflib,os
from pathlib import Path
from datetime import datetime,timezone
from ralup.search import source_identity
from ralup.records import sha256_file
from ralup.calibrate import _write
root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
oldpath=base/'native-admission-v1/admission.json';old=json.loads(oldpath.read_bytes());new=source_identity()
assert old['status']=='pass'
changed={k:{'before':old['source_identity'][k],'after':new[k]} for k in new if new[k]!=old['source_identity'][k]}
assert set(changed)=={'src/ralup/search.py','src/ralup/state.py'}
checks=base/'green-consumer-validation/finish.json';assert json.loads(checks.read_bytes())['exit_code']==0
diffs={k:''.join(difflib.unified_diff((base/'native-admission-v1/producer'/k).read_text().splitlines(keepends=True),(root/k).read_text().splitlines(keepends=True),fromfile='original/'+k,tofile='current/'+k)) for k in changed}
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
report={'status':'pass','source_identity':new,'scopes':old['scopes'],'maximum_steps':old['maximum_steps'],
 'record_type':'separate_source_applicability_analysis; not new numerical execution or producer relabel',
 'numerical_producer_source_identity':old['source_identity'],'numerical_evidence':ref(oldpath),
 'completed_native_attempt':ref(base/'native-admission-v1/finish.json'),'changed_modules':changed,'exact_diffs':diffs,
 'covered_corrections':['H08 cost ledger schema publication','matched model/partition/budget screening','explicit real-data-only launch guard','hash-bound prospective template sidecar','reject malformed initial FP32 history/nonmapping empty cache'],
 'unchanged_numerical_path':'all model/embedding/optimizer/quantization/train/evaluate/calibrate source bytes unchanged; valid numerical states unchanged; state.py changes only reject malformed initial history/cache',
 'new_engineering_checks':ref(checks),'new_check_stdout':ref(base/'green-consumer-validation/stdout.txt'),
 'restore_contract':'no cross-source full-state restore; current-source fresh states or explicit authenticated model-only import preserving cursor/RNG/cost and declared resets',
 'controller_ruling':'finish frozen admission, narrow corrections+consumer tests, separate exact source applicability; no whole native repeat solely for metadata; received11October',
 'created_utc':datetime.now(timezone.utc).isoformat(),'analysis_process':'this current helper has no loss/optimization/query passes'}
p=Path(os.environ['M07_ATTEMPT_DIR'])/'admission-applicability.json';r=_write(p,report);print(json.dumps(ref(p)))
