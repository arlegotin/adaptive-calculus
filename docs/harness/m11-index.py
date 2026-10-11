"""Small durable M11 evidence index; immutable owned attempt files stay in place."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m11-20261011')
def ref(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
rows=[]
for folder in sorted(store.iterdir()):
 if not folder.is_dir():continue
 row={'attempt':folder.name,'files':{p.name:ref(p) for p in folder.glob('*.json') if p.name in ('startup.json','finish.json','report.json','design.json','generation-1.json','supervision.json','duplicate-0.json','duplicate-1.json','reference-normalizer.json','final-reference-normalizer.json','pilot-generation.json','vectors.json','nominees.json','selected-exploratory-controller.json','disposition.json','scored-source-offset-intervals.json')}}
 if (folder/'finish.json').exists():
  d=json.loads((folder/'finish.json').read_bytes());row['exit_code']=d['exit_code'];row['complete_launcher_seconds']=d['elapsed_seconds']
 if (folder/'report.json').exists():
  d=json.loads((folder/'report.json').read_bytes());row['status']=d['status']
  if 'failure' in d:row['failure']=d['failure']
  if 'unknown_partial_work' in d:row['historical_unknown_partial_work']=d['unknown_partial_work']
 rows.append(row)
value={'schema':'ralup.m11_evidence_index.v1','updated_utc':datetime.now(timezone.utc).isoformat(),'status':'needs_revision_pending_independent_review','attempts':rows,
 'finalization_receipt_location':str(store/'fix1-commit-v1'/'report.json'),
 'final_report_location':str(store/'fix1-commit-v1'/'M11-report.md'),
 'fix_round_1':{'finding':'I1','status':'implemented_pending_scoped_rereview','fix_base':'e25a96c4c316129abc193f3715493a3fdbfe08ca','verification':'fix1-verify-selector-v1:27passed1.54s','source_applicability':'Selector-only strict runtime identity change; no old complete-state import, normalizer DATA unchanged; no new model work'},
 'scope':'Exploratory B0null/WP04inconclusive/confirmation-ineligible. No selected final controller or learned gain claim. Current source/noise evidence does not qualify all later horizons.',
 'v1_completed_work_bound':{'updates':64,'training_targets':524288,'query_forward_targets':0,'basis':'raw completed_step events plus pre-scoring query-auth validation traceback and failure-only duplicate journal; historical generic unknown_partial_work=true remains unchanged'},
 'ancestor_cost':'Original typed META parent707 288updates2359296targets shared ancestry, globally deduplicate; never count cloned ledger ancestry as fresh independent work.',
 'unknown_costs':'Early unreceipted CPU/editing/tool/enclosing work unknown, never zero or represented by native launcher walls.',
 'telemetry_scope':'Old bootstrap/calibration producer dropped completed last_actions on rebind: no pre-update complete-owner proof. Corrected5ed817e reference/affected-assessment preserves full-owner clones but rollout step receipts omit policy_actions. No invented old action attribution. Logging-only correction committed16fe874; focused23passed. Exact src delta is meta/tasks.py; feature/history/normalization math unchanged. No old complete-state import. Any future learning requires fresh strict pools; authenticated META-only normalizer DATA retains original provenance.',
 'next':'Full M11 independent review BASEc2552c0..finalHEAD, then mandatoryM10 beforeE01. WP06 reopening requires justified longer-horizon/objective calibration, matched candidate/LR/static own-policy-refresh learning, both development selection stages and B0 applicability; final budgets/selected controller remain null.'}
Path('docs/harness/m11-evidence.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
