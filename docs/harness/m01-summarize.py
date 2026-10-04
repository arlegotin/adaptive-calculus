"""Read-only M01 proof inventory, with explicitly bounded timing claims."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

from ralup.config import canonical_bytes
from ralup.records import sha256_file
from ralup.calibrate import _write

ROOT=Path.cwd();ART=ROOT/'artifacts/checks/m01-v1'
def ref(path):return {'path':str(path),'sha256':sha256_file(path)}
wave=json.loads((ART/'proxy-wave-v3.json').read_bytes())
audit=json.loads((ART/'archive-audit-v3.json').read_bytes())
assert wave['status']==audit['status']=='pass' and len(audit['snapshots'])==30
attempts=[]
for path in sorted(ART.glob('*-finish.json')):
    prefix=path.name.removesuffix('-finish.json')
    start=ART/(prefix+'-start.json');output=ART/(prefix+'.txt')
    finish=json.loads(path.read_bytes());entry=json.loads(start.read_bytes())
    assert sha256_file(output)==finish['output_sha256']
    attempts.append({'name':prefix,'start':ref(start),'finish':ref(path),'output':ref(output),
                     'exit_code':finish['exit_code'],'recorded_clock_seconds':finish['full_process_seconds'],
                     'clock_scope':'wrapper entry before source hashing and workload import through workload stdout fsync; outer shell/Python bootstrap and terminal receipt publication/exit are outside this clock',
                     'nested_in_recovery_wave':prefix.startswith('proxy-v3-'),
                     'nested_in_failed_wave':prefix=='proxy-muon_project_fp32-U-full'})
# Every published input/cost/mapping journal is an exact durable copy. This
# rechecks the destination, preserving historical original path/hash identities.
for entry in wave['journal_mappings']:
    assert sha256_file(entry['durable']['path'])==entry['durable']['sha256']
scored=set();target_events=0
for path in sorted((ART/'proxy-journals-v3').rglob('input-*.json')):
    value=json.loads(path.read_bytes());count=0
    for group in value['offsets']:
        for offset in group['offsets']:scored.add((group['document_id'],offset));count+=1
    assert count==8192
    target_events+=count
assert target_events==229376
union={'scope':'M01 new workload only; full-project incremental novelty is not inferred',
       'offsets':[[doc,offset] for doc,offset in sorted(scored)]}
union_ref=_write(ART/'task-offset-union.json',union)
evidence={'schema':'m01.evidence_inventory.v1','recorded_utc':datetime.now(timezone.utc).isoformat(),
    'status':'DONE_WITH_CONCERNS','product_commit':'0d3bb2f','task_base':'ce27204',
    'design':ref(ROOT/'docs/harness/m01-design-v4.json'),'wave':ref(ART/'proxy-wave-v3.json'),
    'complete_archive_audit':ref(ART/'archive-audit-v3.json'),
    'cpu_cli':ref(ART/'muon-cpu-final.json'),'mps_cli':ref(ART/'muon-mps-final.json'),
    'tdd_and_verification_attempts':attempts,'indexed_completed_attempts':len(attempts),
    'this_inventory_attempt':'separate summarize-start/finish/output receipts publish around this operation; no future receipt hash invented',
    'scientific_work':{'new_updates':28,'scored_update_bytes':229376,'query_bytes':0,'protected_bytes':0,
                       'task_distinct_target_bytes':len(scored),'offset_union':{'path':str(union_ref.path),'sha256':union_ref.sha256},
                       'full_project_incremental_novelty':None},
    'cost':{'chronological_original_plus_recovery_wave_seconds':wave['chronological_wave_seconds'],
            'recovery_process_seconds_to_wave_publication':wave['recovery_process_seconds'],
            'wave_envelope_seconds':1200,'archive_audit_workload_seconds':audit['project_seconds'],
            'workload_process_clocks':'all invoked new validation/probe attempts have durable stdout/startup source hashes and elapsed/OS CPU receipts; these clocks are explicitly scoped, not exact whole-shell/OS-launch-to-exit wall',
            'complete_engineering_session_cost':None,'complete_engineering_session_cost_reason':'editing/review/outer interpreter and shell bootstrap/terminal publication clocks are not covered by attempt wrapper clocks; no fabricated zero/total',
            'nesting':'proxy-v3 child receipts are inside recovery-wave; original child receipt is inside failed-wave; do not add nested clocks or overlapping classifications',
            'fixture_work':'synthetic engineering, never converted to corpus/scientific byte counts',
            'shared_derivative':'all model transactions include shared OrderedEmbedding selector/transfer/GEMM; source and costs retained'},
    'preservation':{'durable_directory':str(ART),'temporary_originals':['/private/tmp/ralup-m01-20261004-v1','/private/tmp/ralup-m01-20261004-v3'],
        'complete_snapshot_generations':30,'essential_raw_owners_contexts_sources':'exact complete manifest/container reconstruction from durable object archive; no temp dependency',
        'originals_disposition':'preserved; no old/user/failure artifact removed or relocated',
        'artifact_refs':'historical original refs remain immutable; wave journal mappings and audit snapshot entries give explicit durable counterparts',
        'bulk_git_status':'Git-ignored; only small inventory/design/helpers/report committed',
        'external_destination':'none authorized or used'},
    'limits':['Muon+ software/release reuse grant unresolved; source-reference eligibility blocked, never defeated',
              'independent equations/portable FP32, not upstream BF16/release reproduction',
              'four-step two-method replay plus named single-step order composition only; no core/long-horizon/universal exactness or winner claim',
              'original N01 early engineering uncertainty remains inherited, not repaired or zeroed',
              'scientific budgets null and protected scoring denied'],
    'retained_durable_bytes_before_inventory':sum(p.stat().st_size for p in ART.rglob('*') if p.is_file()),
    'git_free_bytes_before_inventory':shutil.disk_usage(ROOT).free}
assert evidence['git_free_bytes_before_inventory']>=2**30
reference=_write(ROOT/'docs/harness/m01-evidence.json',evidence)
print(json.dumps({'evidence':ref(reference.path),'attempts':len(attempts),'distinct_task_targets':len(scored),
                  'retained_durable_bytes':evidence['retained_durable_bytes_before_inventory'],'git_free_bytes':evidence['git_free_bytes_before_inventory']}))
