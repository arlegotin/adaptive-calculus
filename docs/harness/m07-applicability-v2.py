"""Exact source applicability after selection/validation/failure-accounting fixes."""
import json,difflib,os
from pathlib import Path
from datetime import datetime,timezone
from ralup.search import source_identity
from ralup.records import sha256_file
from ralup.calibrate import _write
root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
old_path=base/'native-extension256/admission.json';old=json.loads(old_path.read_bytes());new=source_identity()
changed={k:{'before':old['source_identity'][k],'after':new[k]} for k in new if new[k]!=old['source_identity'][k]}
assert set(changed)=={'src/ralup/search.py'}
checks=base/'green-final-contracts/finish.json';assert json.loads(checks.read_bytes())['exit_code']==0
prior32=base/'applicability-v1/admission-applicability.json'
h128=base/'native-horizon128/admission.json';warm256=base/'native-warm-extension256/admission.json'
sources={32:[prior32],128:[h128],256:[old_path,warm256]}
output=Path(os.environ['M07_ATTEMPT_DIR']);results=[]
for horizon,paths in sources.items():
    scopes=[]
    for p in paths:
        evidence=json.loads(p.read_bytes())
        assert evidence['source_identity']==old['source_identity']
        # The capped128 report lists only its five completed exact scopes; its
        # failed/unattempted QuEST recipes never enter this applicability list.
        for scope in evidence['scopes']:
            if scope not in scopes:scopes.append(scope)
    report={'status':'pass','source_identity':new,'scopes':scopes,'maximum_steps':horizon,
        'record_type':'separate_source_applicability_analysis_not_numerical_execution',
        'created_utc':datetime.now(timezone.utc).isoformat(),'numerical_producer_source_identity':old['source_identity'],
        'source_evidence':[ref(p) for p in paths],'changed_modules':changed,
        'exact_diff':''.join(difflib.unified_diff((base/'native-extension256/producer/src/ralup/search.py').read_text().splitlines(keepends=True),
            (root/'src/ralup/search.py').read_text().splitlines(keepends=True),fromfile='original/search.py',tofile='current/search.py')),
        'covered_changes':['quantized-only ranking retains FP32 reference evidence','reject unsupported nonempty AdamW method_config before construction',
            'failed training raw records contain only new events','failed query counts remain known-completed lower bounds with explicit unknown partial work'],
        'unchanged_numerical_path':'all model/embedding/optimizer/quantizer/train/evaluate/state/calibrate bytes unchanged; successful existing recipes use empty AdamW method_config and identical resolved betas/epsilon; new query_attempts are report metadata only',
        'limits':'exact old/new source distinct; no cross-source full-state restore, no gate extension for failed QuEST or unknown horizons; each scope has only the listed horizon',
        'restore_contract':'fresh current-source states or separately authenticated explicit model-only import with cursor/RNG/full cost preservation and declared resets',
        'checks':ref(checks),'check_stdout':ref(base/'green-final-contracts/stdout.txt'),
        'original_strong_parent_authentication':ref(base/'authenticate-strong-parent/authenticated-original.json'),
        'new_training_steps':0,'new_query_targets':0}
    p=output/f'admission-applicability-{horizon}.json';_write(p,report);results.append(ref(p))
print(json.dumps({'applicability':results}))
