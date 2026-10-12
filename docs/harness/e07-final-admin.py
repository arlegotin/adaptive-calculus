"""Bounded E07 report/index and authenticated scratch delta; no cleanup authority."""
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,subprocess
from pathlib import Path
spec=importlib.util.spec_from_file_location('capture',Path(__file__).with_name('e07-capture.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path.cwd();BASE=Path(os.environ['E07_BASE']);GIT=Path(os.environ['E07_COMPLETION_ROOT']);OUT=Path(os.environ['E07_OPERATION'])
def write(path,v):path.write_bytes(m.encoded(v))
def ref(path):return m.ref(Path(path))
subprocess.run([str(BASE/'environment-v1/core/bin/python'),'-B','docs/harness/e07-bound-check.py'],check=True)
ops=[]
for root in (BASE,GIT):
 for p in sorted(root.glob('*/finish.json')):
  f=json.loads(p.read_bytes());s=json.loads((p.parent/'startup.json').read_bytes())
  ops.append({'name':p.parent.name,'startup':ref(p.parent/'startup.json'),'finish':ref(p),
   'stdout':ref(p.parent/'stdout.txt'),'stderr':ref(p.parent/'stderr.txt'),
   'child_exit_code':f['exit_code'],'original_bounds_pass':f['bounds_pass'],
   'capture_exit_code':f.get('capture_exit_code',f['exit_code'] if f['bounds_pass'] else 3),
   'child_component_bound_pass':False if p.parent.name=='native-replay-v2' else f.get('child_component_bound_pass'),
   'elapsed_seconds':f['elapsed_seconds'],'child_user_seconds':f['child_user_seconds'],'child_system_seconds':f['child_system_seconds'],
   'forecast_physical_bytes':s['forecast_physical_bytes'],'remaining_reserve_bytes':s['remaining_reserve_bytes'],
   'actual_allocated_bytes':f['owned']['allocated_bytes'],'started_utc':s['utc'],'finished_utc':f['utc']})
history_path=BASE/'preservation-v1/history-mapping.json';history=json.loads(history_path.read_bytes());originals={r['original_path']:r for r in history['files']}
for r in originals.values():assert ref(r['durable_path'])['sha256']==r['sha256']
proof=BASE/'preservation-proof-v2/m01-extraction-proof.json'
evidence=GIT/'report-v3/evidence.json';report=GIT/'report-v3/report/report.json'
report_value=json.loads(report.read_bytes());supplement=GIT/'report-v3/native-storage-supplement.json'
assets={}
for label,path in [('raw_evidence',evidence),('research_report',report),('tables',report.parent/'tables.json'),('report_markdown',report.parent/'report.md'),
 ('native_storage_supplement',supplement),('paired_replay',GIT/'native-replay-v2/paired-replay.json'),
 ('source_union',GIT/'report-v3/source-union.json'),('environment_audit',BASE/'preservation-proof-v2/environment-audit.json'),
 ('analysis_lock',ROOT/'docs/harness/e07-analysis.lock'),('M01_required_member_proof',proof),('E06_dual_root_authentication',BASE/'e06-proof-v3/e06-proof.json'),
 ('initial_history_mapping',history_path),('report_CLI_stdout',GIT/'report-v3/cli-0-stdout.txt'),('reproduce_CLI_stdout',GIT/'report-v3/cli-1-stdout.txt'),
 ('shell_reproduce_stdout',GIT/'report-v3/shell-stdout.txt')]:assets[label]=ref(path)
for path in sorted((GIT/'plots-v2').glob('*')):
 if path.suffix in ('.svg','.pdf','.png') or path.name=='plot-provenance.json':assets[path.name]=ref(path)
assets['root-ruling-index-v8.md']=ref(ROOT/'.superpowers/sdd/2026-10-03-ralup/root-ruling-index-v8.md')
for name in ('baseline-sources.md','numerics-report.md','dataset-card.md','root-ruling-index-v7.md','root-whole-branch-review-requirements.md','E07-review-context.md'):
 r=next(r for r in originals.values() if Path(r['original_path']).name==name);assets[name]=ref(r['durable_path'])
index={'schema':'ralup.e07_evidence_index.v1','status':'implemented_verified_pending_fresh_task_review',
 'research_status':report_value['scientific_status'],'evidence_class':'exploratory engineering; no confirmation or protected scoring',
 'attempts':ops,'assets':assets,'scope_notes':{'CPU':'1250 behavior passes; v2 enclosing reserve fails/outer3',
 'native':'numerical exact replay passes; child binding bound fails; original complete-stage bounds and exit0 preserved',
 'historical_source':'never acquire current restore compatibility through migration or index',
 'cost':'launcher walls are scoped; nested clocks not added; complete project/engineering cost unknown'},
 'final_administration_startup':ref(OUT/'startup.json') if (OUT/'startup.json').exists() else None,
 'final_administration_finish_path':str(OUT/'finish.json'),
 'late_root_delta_required':'Capture hash-linked changed/new review/ruling/finishing records after this boundary and before only-own-plan scratch cleanup; no cleanup authorization here',
 'durable_roots':[str(BASE),str(GIT),str(GIT.parent/'e06-v1')],
 'environments':{'core':str(BASE/'environment-v1/core/bin/python'),'analysis':str(BASE/'analysis-environment-v1/analysis/bin/python')}}
indexpath=ROOT/'docs/harness/e07-evidence.json';write(indexpath,index)
lines=['# E07 implementation report','',
'Implemented and verified within the admitted exploratory-only terminal scope; awaiting fresh task review. WP04 remains inconclusive/B0 null and WP06 remains NEEDS_REVISION. Final C, LR, B*, budgets, n and confirmation are absent; protected data were never scored. This is not an unqualified operational PASS: the CPU reserve failure and native child-bound failure remain explicit below.','',
'## Changes and runnable handoff','',
'Added deterministic `build_report`, `reproduce` and `research_disposition`, CLI report/reproduce commands, explicit immutable-input shell entry point, synthetic contract tests, independent core/analysis environment reproduction, finite native paired replay, authenticated raw-table regeneration and six standalone figures. README, decisions, current execution links and progress now distinguish 31 completed prior engineering tasks plus E07 pending review from the unchanged scientific gates. Original progress is preserved from the base Git object; drafting checklist history is not rewritten as scientific success.','',
'The report contains all eleven S39 sections, per-run rows including failures, distinct deployment/search/project ledgers, source deviations, null missing costs, analytical precision and break-even eligibility rules. Nineteen genuine M11 META rollout endpoints are query NLL, not validation BPB or independent final seeds. Same-parent held panels do not establish recipe replication. No final-arm regression/ablation/power/efficiency result is invented. The source union merges authenticated H08/N01/M01/M07/M11/E07 document offsets; incomplete historical coverage keeps the complete PROJECT unique total null. Shared ancestry is counted once and nested walls are not summed.','',
'Explicit report command (fresh destination required):','',
'```sh',f'{BASE}/environment-v1/core/bin/python -B -m ralup report --evidence {evidence} --sha256 {ref(evidence)["sha256"]} --output NEW_REPORT_DIRECTORY',
f'sh reproduce.sh {BASE}/environment-v1/core/bin/python {evidence} {ref(evidence)["sha256"]} NEW_REPRODUCTION_DIRECTORY','```','',
'Run from this checkout with `PYTHONPATH=src` for the direct module command. The shell sets the source path. The recorded report-v3 operation actually executed report CLI, reproduce CLI and shell reproduction successfully; reconstruction compares two complete reports exactly. Confirmed-study manifests are explicitly blocked/reopened until a validated adapter and actual authorized study exist. No hooks, pickle execution or hidden protected scorer are accepted. Numerical reproduction is separately source-bound in the recorded replay recipe and is not silently rerun by the reporting CLI.','',
'## RED/GREEN and final verification','',
'Initial RED produced eleven missing-report-API failures. First GREEN exposed six fixture serialization errors (JSON list incorrectly passed to the object-only canonical helper); correcting only fixture serialization yielded eleven passing report integration tests. Tests reject omitted failed trials, altered hashes, missing costs, stream-only efficiency, and unsupported meta-recipe claims; they cover distinct research dispositions, reproducibility and eligible break-even.','',
'First full current-tree CPU wrapper run: 16 failed, 1236 passed, 76 skipped. Fourteen failures traced to wrapper import-root omission and two to actual-MPS-specific assertions under the explicit CPU availability override. The wrapper was corrected to include the project import root and deselect the actual-MPS export composition tests. No numerical/product/test expectation was changed. Corrected genuine full current-tree CPU run: 1250 passed, 76 skipped, 2 deselected in 134.36 s; child exit0. Its enclosing future-reserve guard failed and returned outer exit3. Thus behavior passes and operational admission fails are independent. No repeat was made solely to relocate administration.','',
'Fresh actual M3/fallback0 replay: seed607, paired fixed/dynamic max quantizers with full Hadamard rotation, one-layer width8 model, four128-target updates per arm. Each arm repeats with interruption after two updates and a fresh interpreter continuation. Every per-step scientific-state hash matches uninterrupted execution exactly; inputs are paired. Sixteen physical updates and2048 training/loss targets were executed, with512 unique source targets; no development/protected score, selection or pilot. First native attempt failed on an incorrect helper BatchSlice field before training; corrected helper uses block_slices. No source product change was needed. Full CPU includes the finite numerical reference fixtures; native is the separately admitted composition, not a Cartesian native-gate repeat.','',
'Native-v2 original child/capture exit0 and original complete-stage bounds_pass=true are preserved. Its declared child_allocation_bound_bytes=33,554,432 was binding. Checkpoint trees alone allocate36,470,784, exceeding it by2,916,352. The new hash-linked supplement records numerical_pass=true, complete_stage_bounds_pass=true and child_component_bound_pass=false; it also measures the complete child inventory. Complete stage38,608,896 fits50,331,648, which does not repair the child failure. Report-v1/v2 historical interpretation is superseded by versioned report-v3, never overwritten.','',
'After plots-v1 completed, scoped recorder RED demonstrated the missing bound checker. GREEN checks exact-bound, child-overrun/aggregate-fit and administration-overrun/aggregate-fit cases. New receipts enforce all declared child/administration bounds; report-v3 and plots-v2 pass these checks. All78 src Python files remain byte-identical to the CPU/native producing revisions, authenticated by the supplement. Administrative helpers have their own later identities. H07 source-strict complete-state compatibility is never weakened; historical MatrixHistory snapshots remain rejected by the current semantic loader. Model-only import still requires its explicit reset contract.','',
'## Actual environment, provenance and figures','',
'Core requirements.lock SHA488142a35a4b1a6abe132f08b5895621ca30bb535b0326e82516446903fcb32b is unchanged. A separate real venv was recreated with hash-required installation of all20 locked distributions. environment-v1 terminal failed only on equality with the original editable ralup0.1.0 metadata, which is not a locked distribution. The subsequent audit authenticates installed RECORD file hashes, wheel install report, exact executable and explicit current-source import root; no metadata-only reinstall concealed the failure. Actual runtime/OS26.7.1/UTC/commands/source archives are in each startup; old operating-system and Linux prototype claims retain their historical scopes.','',
'Plotting uses a separately installed, wheel-hash-locked analysis environment (matplotlib3.10.8, analysis numpy2.5.4 versus core2.5.3), separate process and caches, with the exact lock saved. Cost/quality plots use19 raw M11 META endpoints versus training targets and measured nested rollout seconds, with warm ancestry/tuning and final-quality limits explicit. Precision uses the analytical two-sided97.5% Student-t half-width in units of unknown paired SD for hypothetical n; it is not an empirical pilot, power estimate or locked design. Both figures are standalone SVG/PDF/PNG. The plots were visually inspected for readable labels and scope caveats; plots-v2 binds final report-v3.','',
'## Storage, costs and failure accounting','',
'Every captured command has exact startup source bytes/UTC/runtime, logical and physical forecast, actual stdout/stderr, child status, cost and postflight. Tables below list scoped launcher walls; nested subprocess clocks are not summed as project time. Editing, tool, review and missing historical costs remain unknown. No inherited reserve was asserted restored. The earlier planned RED admission with2300MiB remaining reserve was refused before writing an operation directory; the remaining estimate was recalibrated after installation.','',
'The Data volume lost the planned reserve after CPU-v2. New Data writes were held. Root separately authorized only the new original-checkout ignored E07 Git completion root, with <=160MiB aggregate, >=1GiB Git free, combined evidence <=32GiB, raw Data >=64GiB and zero owned Data writes. The older E06 exception was not reused implicitly. All new tmp/cache/bytecode/output stays in that owned Git root. Pre/post inventories show Data unchanged. No external erosion is attributed to this project without evidence. The final-admin-v1 preflight was initially refused before creating its directory when fresh raw Data free68,673,179,648 was46,297,088 bytes below the hard floor; this observation is not retroactively relabeled. Every preserved bound failure remains visible, and no existing artifacts were deleted to make room. Root subsequently authorized a strictly finite zero-Data-write final-administration exception. Data-floor status remains false when below64GiB, even if the zero-write route is admitted; Git/component/aggregate bounds remain binding. No model, query, CPU/MPS suite or dependency installation is permitted under this exception. Focused administrative RED/GREEN checks verify the distinction. Final administration has its own forecast/finish; any later finalization records have a small separately measured allowance.','',
'| Operation | Child / capture | Original complete bounds | Child bound | Scoped seconds | Allocated bytes |','|---|---:|---|---|---:|---:|']
for r in ops:lines.append(f"| {r['name']} | {r['child_exit_code']} / {r['capture_exit_code']} | {r['original_bounds_pass']} | {r['child_component_bound_pass']} | {r['elapsed_seconds']:.6f} | {r['actual_allocated_bytes']} |")
lines+=['','Earlier receipts without a separately checked child field show null above, not an inferred pass. Report normalization wrappers preserve failures and original source indexes; reporter/plot operations are engineering/admin work, not new scientific target passes.','',
'## Mandatory migration mapping and late delta','',
'No existing item was moved, deleted, overwritten or relabeled. The initial scratch mapping authenticates every copied file and its original path/hash. The final-administration history-delta mapping selects changed/new bytes against that mapping, preserves prior hashes, captures its own UTC boundary, and reauthenticates unchanged essential records. The current copies are not a claim that future root records already exist. Thirteen worktree-bound raw-report inputs are also preserved exactly with the handoff mapping. Actual report-v3 retains original absolute references; after checkout removal, reproduce from an authenticated checkout at those paths or create a separately authenticated relocated evidence root using that mapping. No automatic path substitution or historical source identity waiver is claimed.','',
'| Essential scope | Durable preservation / proof | Coverage and limits |','|---|---|---|',
f'| All captured task briefs/reports/reviews/findings/Declined dispositions and source/numerics/data ledgers | {history_path} | Exact original-to-durable file mappings; permissions/publication rights unchanged |',
f'| Changed root progress/state/current context/final-review instructions plus E07 report | {OUT}/history-delta.json | Current boundary only; later root delta mandatory |',
f'| Missing M01 temporary tree | {proof} |30 mandatory generations recovered from authenticated relocated archive, with original manifest/container hashes, payload members and exact reconstruction proof; not the whole approximately882MiB tree |',
f'| M01 archive source material | {BASE}/preservation-v1/m01-archive | Original archive members retained; current-loader MatrixHistory rejection preserved as semantic incompatibility |',
f'| E06 dual-root admin bundle | {BASE}/e06-proof-v3/e06-proof.json | Exact gzip and uncompressed inventory, complete-report/postflight/original refs authenticated separately; original-checkout e06-v1 root survives cleanup |',
f'| E07 numerical/CLI/plots/receipts | {GIT} and {BASE} | Both roots remain durable; source/UTC/runtime identities historical |','',
'M01 temporary original was observed absent on12October; attribution is unknown. Thirty required snapshot generations have authenticated member-level reconstruction coverage, including6574 tensors and1440 producer entries. Optional/unarchived files and unknown whole-tree coverage cannot be claimed recovered. Missing historical operands/costs are not reconstructed by copying. The first preservation operation failed semantic recovery after successful copies; preservation-proof-v2 proved exact members but then failed relative E06-ref resolution; e06-proof-v3 corrected only that resolver. All three terminals remain independently visible.','',
'Root must capture a later hash-linked delta covering E07 fresh review/fixes, ONE final whole-branch review, final chronological rulings and finishing records before deleting only this plan scratch. Preserve the current interim root-ruling-index-v8 (all80 literal rulings), root-ruling-index-v7 and all older indexes, root-final-review-context, root-whole-branch-review-requirements and E07-review-context exactly. E07 does no cleanup, merge, push or publication.','',
'## Exact evidence pointers','',f'Index: `{indexpath}`. Every pointer below includes its SHA256; operation startup/finish/stdout/stderr refs and costs are in the index. Final administration finish: `{OUT}/finish.json`; later completion.json binds this index/report to the local commit.','']
for label,r in assets.items():lines += [f'- {label}: `{r["path"]}` — `{r["sha256"]}` ({r["bytes"]} bytes).']
lines+=['','## Self-review and remaining limitations','',
'Checked API completeness/error paths, source authentication, null cost preservation, protected-data closure, source identity applicability, separate actual child/complete resource statuses, no historical authority escalation, standalone plots, exact core versus analysis environments, all required preserved ledgers and explicit late-root capture obligation. This self-review does not replace coordinator fresh task review or root whole-branch review.','',
'R41 access remains unresolved; no author/existence/priority inference. QuEST/comparator release fidelity and software reuse rights remain unresolved. Dataset-card CC3 versus prose CC4/GFDL discrepancy remains; local evidence preservation does not authorize corpus redistribution. Supplied Linux prototype evidence is historical. WP04/WP06 and later selection/design/confirmation gates remain unresolved; no final effect estimate, H1/H2 result, efficiency percentage or laptop runtime is presumed.','']
reportpath=ROOT/'.superpowers/sdd/2026-10-03-ralup/E07-report.md';reportpath.write_text('\n'.join(lines))
if os.environ.get('E07_DRAFT_ONLY')=='1':
 print(json.dumps({'draft_index':ref(indexpath),'draft_report':ref(reportpath),'administration_capture':'not admitted; no copy claimed'}));raise SystemExit(0)
# Copy exact current records at one boundary; never overwrite initial history.
started=datetime.now(timezone.utc).isoformat();pending=[]
for p in sorted((ROOT/'.superpowers/sdd/2026-10-03-ralup').rglob('*')):
 if not p.is_file():continue
 raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();old=originals.get(str(p))
 if old is None or old['sha256']!=digest:pending.append((p,raw,digest,old))
raw_source_paths=[Path(r['path']) for r in json.loads(evidence.read_bytes())['sources'] if r['path'].startswith(str(ROOT)+'/')]
forecast=sum(((len(raw)+4095)//4096)*4096 for _,raw,_,_ in pending)+sum(((p.stat().st_size+4095)//4096)*4096 for p in raw_source_paths)+512*1024
assert forecast<4*1024**2,('Changed scratch exceeds forecast',forecast)
write(OUT/'delta-forecast.json',{'utc':started,'new_files':len(pending),'prospective_physical_bytes':forecast,'scope':'scratch delta plus13 worktree-bound raw source refs plus512KiB handoff/index/report margin'})
changed=[]
for p,raw,digest,old in pending:
 dest=OUT/'history-delta'/p.relative_to(ROOT/'.superpowers/sdd/2026-10-03-ralup');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 changed.append({'original_path':str(p),'prior_sha256':old['sha256'] if old else None,'durable':ref(dest)})
write(OUT/'history-delta.json',{'schema':'ralup.e07_history_delta.v1','previous_mapping':ref(history_path),
 'capture_started_utc':started,'capture_finished_utc':datetime.now(timezone.utc).isoformat(),'files':changed,
 'unchanged_original_mapping_authenticated':len(originals),'later_root_delta_required':index['late_root_delta_required']})
# Save exact tracked handoff documents and shell, independently of mutable checkout.
raw_source_paths=[Path(r['path']) for r in json.loads(evidence.read_bytes())['sources'] if r['path'].startswith(str(ROOT)+'/')]
paths=raw_source_paths+[ROOT/'README.md',ROOT/'decisions.md',ROOT/'progress.json',indexpath,ROOT/'reproduce.sh',ROOT/'docs/harness/e07-contract.md',ROOT/'docs/superpowers/plans/2026-10-03-ralup.md',ROOT/'docs/superpowers/plans/2026-10-03-ralup-evaluation.md']
preserved=[]
for p in paths:
 dest=OUT/'handoff'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());preserved.append({'original':str(p),'durable':ref(dest)})
write(OUT/'handoff-mapping.json',{'files':preserved,'report':ref(OUT/'history-delta/E07-report.md')})
subprocess.run(['git','diff','--check'],check=True)
print(json.dumps({'index':ref(indexpath),'report':ref(reportpath),'delta':ref(OUT/'history-delta.json'),
 'handoff':ref(OUT/'handoff-mapping.json'),'changed_history_files':len(changed),'operations':len(ops),'git_diff_check':'pass'}))
