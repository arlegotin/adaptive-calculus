"""Authenticate the original continuation parent before ending its source boundary."""
import json,os,time
from pathlib import Path
from datetime import datetime,timezone
from ralup.records import ArtifactRef,sha256_file
from ralup.state import load_snapshot,_validated
from ralup.search import source_identity
from ralup.calibrate import _write

start=time.perf_counter();root=Path.cwd()
base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
designpath=root/'docs/harness/m07-strong-conversion-design-v1.json'
d=json.loads(designpath.read_bytes());assert d['source_identity']==source_identity()
for r in (d['checkpoint'],d['admission'],d['completed_attempt'],d['ancestry']['warm_parent']):
    assert sha256_file(r['path'])==r['sha256']
snapshot=load_snapshot(ArtifactRef(**d['checkpoint']));payload=_validated(snapshot)
assert payload['configuration']['artifact_role']=='continuation'
assert payload['configuration']['source_partition']=='target_train'
assert payload['cursor']['cumulative_target_count']==2359296
q=base/'query-strong-conversion/conversion-results.json';query=json.loads(q.read_bytes())
assert query['status']=='completed' and all(r['before_state_sha256']==r['after_state_sha256']==d['expected_state_sha256'] for r in query['rows'])
report={'record_type':'authenticated_original_continuation_for_future_explicit_derived_model_only_contract',
    'created_utc':datetime.now(timezone.utc).isoformat(),'checkpoint':d['checkpoint'],'unchanged_original_role':'continuation',
    'configuration_sha256':payload['configuration_sha256'],'source_identity':source_identity(),
    'producer_archive':json.loads(Path(d['checkpoint']['path']).read_bytes())['producer_files'],
    'parameter_manifest':[{'name':n,'shape':list(w.shape),'dtype':str(w.dtype)} for n,w in payload['weights'].items()],
    'source_partition':'target_train','source_data':payload['cursor']['source'],'mixed_ancestry':d['ancestry'],
    'stream':payload['ledger']['stream'],'passes':payload['ledger']['passes'],'full_state_read_only_proof':ref(q),
    'complete_query_attempt':ref(base/'query-strong-conversion/finish.json'),'design':ref(designpath),
    'cost_contract':'authentication CPU/I/O only, zero model passes; future derived creation/import/selection charged separately, full original ancestry once globally',
    'new_training_steps':0,'new_query_targets':0,'complete_seconds':time.perf_counter()-start}
p=Path(os.environ['M07_ATTEMPT_DIR'])/'authenticated-original.json';_write(p,report);print(json.dumps(ref(p)))
