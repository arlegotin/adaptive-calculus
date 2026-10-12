"""Fix1 exact report/index/delta publication within the finite admin route."""
from datetime import datetime,timezone
import importlib.util,json,os,subprocess
from pathlib import Path
spec=importlib.util.spec_from_file_location('capture',Path(__file__).with_name('e07-capture.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path.cwd();OUT=Path(os.environ['E07_OPERATION']);BASE=Path(os.environ['E07_BASE']);GIT=Path(os.environ['E07_COMPLETION_ROOT'])
ref=m.ref
ops=[]
for name in ('red','green','report','plots'):
 p=GIT/f'fix1-{name}-v1';f=json.loads((p/'finish.json').read_bytes());s=json.loads((p/'startup.json').read_bytes())
 ops.append({'name':p.name,'startup':ref(p/'startup.json'),'finish':ref(p/'finish.json'),'stdout':ref(p/'stdout.txt'),'stderr':ref(p/'stderr.txt'),
  'child_exit_code':f['exit_code'],'capture_exit_code':f['capture_exit_code'],'data_floor_pass':f['data_floor_pass'],
  'route_resource_bounds_pass':f['route_resource_bounds_pass'],'zero_data_write_route_admitted':f['zero_data_write_route_admitted'],
  'child_component_bound_pass':f['child_component_bound_pass'],'administration_bound_pass':f['administration_bound_pass'],
  'elapsed_seconds':f['elapsed_seconds'],'child_user_seconds':f['child_user_seconds'],'child_system_seconds':f['child_system_seconds'],
  'forecast_physical_bytes':s['forecast_physical_bytes'],'remaining_reserve_bytes':s['remaining_reserve_bytes'],
  'actual_allocated_bytes':f['owned']['allocated_bytes'],'started_utc':s['utc'],'finished_utc':f['utc']})
reportdir=GIT/'fix1-report-v1';assets={}
for name,path in [('raw_evidence',reportdir/'evidence.json'),('research_report',reportdir/'report/report.json'),('tables',reportdir/'report/tables.json'),
 ('report_markdown',reportdir/'report/report.md'),('source_union',reportdir/'source-union.json'),('post_fix_source_applicability',reportdir/'source-applicability.json'),
 ('report_CLI_stdout',reportdir/'cli-0-stdout.txt'),('reproduce_CLI_stdout',reportdir/'cli-1-stdout.txt'),('shell_reproduce_stdout',reportdir/'shell-stdout.txt')]:assets[name]=ref(path)
for path in (GIT/'fix1-plots-v1').iterdir():
 if path.suffix in ('.svg','.pdf','.png') or path.name=='plot-provenance.json':assets[path.name]=ref(path)
identical_png={n:ref(GIT/'fix1-plots-v1'/n)['sha256']==ref(GIT/'plots-v2'/n)['sha256'] for n in ('cost-quality.png','precision.png')}
assert all(identical_png.values()) # Same plotted values/labels; prior visual inspection applies to exact PNG bytes.
indexpath=ROOT/'docs/harness/e07-evidence.json';index=json.loads(indexpath.read_bytes());prior_index=ref(GIT/'final-admin-v1/handoff/docs/harness/e07-evidence.json')
assert prior_index['sha256']=='6ee31d1c46a0b8b572946d8aae487739bcb2e148bae2aafd1761ae8bd5f50921' and prior_index['bytes']==40921
index['initial_assets_before_fix1']=index['assets'].copy();index['assets'].update(assets);index['attempts']+=ops
index['status']='fix1_implemented_pending_fresh_scoped_review';index['fix1']={'base':'532370fa2d9936cef0d336f12d580fbc46fc815b','findings':['I1','I2'],
 'prior_index':prior_index,'prior_index_original_location':{'kind':'historical_location_only','original_path':str(indexpath)},'assets':assets,'plots_identical_PNG_bytes':identical_png,
 'source_applicability':'Only focused report-contract checks cover changed report.py. Original CPU1250/native proof and ALL78 equality are historical; H07 strict cross-source restoration remains rejected.',
 'admin_startup':ref(OUT/'startup.json'),'admin_finish_path':str(OUT/'finish.json'),'late_root_delta_required':True}
indexpath.write_bytes(m.encoded(index))
reportpath=ROOT/'.superpowers/sdd/2026-10-03-ralup/E07-report.md'
lines=['','## E07 fix round1 — I1/I2 (supersedes current-source applicability above)','',
'Base532370fa2d9936cef0d336f12d580fbc46fc815b. Read the FULL E07 review: spec Issues found and quality Needs fixes, two Important findings, no Critical/Minor. Both findings are addressed; pending fresh scoped review. Earlier report text/receipts remain historical. In particular every pre-fix ALL78-current equality statement describes only its original capture; current report.py differs from both historical CPU/native producers. No old source hashes or snapshots were changed.','',
'I1: removed the blanket rejection of every positive report containing any unknown row cost. Required cost fields and nonnegative finite known-value checks remain. The public report now retains positive quality but slower, positive quality with unresolved deployment cost, and measured lower-deployment-cost dispositions while project costs remain null. Comparable measured baseline/candidate costs gate deployment efficiency; known meta/tuning numerators separately gate break-even. Unknown costs never become zero or invented totals.','',
'I2: validate each declared operational primary-ledger path/hash directly before any report output. Missing or altered bytes yield incomplete evidence and blocked reproduction. The normalizer also registers all three operational durable ledgers in sources. Historical original locations inside mapping documents remain identities, not recursively followed operational references. No archive/path-compatibility framework or original ArtifactRef rewrite was introduced.','',
'RED: five new failures plus11 existing passes in0.07s. GREEN:16 passed in0.04s. Three public build_report quality fixtures cover slower, unresolved deployment, and lower measured deployment with null project/meta/tuning costs; the latter has no unsupported break-even numerator. Two primary-ledger fixtures first reproduce successfully, then alter/remove the ledger with unchanged immutable evidence-root and mapping hashes, requiring build_report rejection and blocked reproduce. Historical absent original-location paths in mapping remain unopened. Existing omission/hash/required-cost/stream-only and disposition controls continue passing. Focused allocation/route administrative checks also pass; no model/training/query/MPS/dependency work occurred.','',
'Actual new raw report, report CLI, reproduce CLI, shell entry point and six standalone figures were regenerated in unique fix1 paths. The two PNGs are byte-identical to the previously inspected plots, and new plot provenance binds the newly authenticated report. Research remains exploratory/inconclusive: WP04/B0 and WP06 blocks, no final C/LR/B*/budgets/n/confirmation/protected scores. Prior report-v1/v2/v3 and numerical/storage failures remain immutable.','',
'Prospective all-remaining physical forecast at admission:47MiB (RED6 + GREEN6 + raw report20 + plots6 + administration8 + closure1). Starting E07 Git allocation89,788,416B; total forecast139,071,488B <160MiB. Data store28,942,725,120 allocated B and prior E06 Git inventory are included in the combined32GiB cap. Git free6,760,222,720B retains1GiB plus separate16MiB source allowance. Raw Data68,102,668,288B was below64GiB; data_floor_pass=false is never relabeled. Each stage uses separately declared child and4MiB administration bounds, owned Git temp/cache/test-output, and exact unchanged Data inventories. Allocation/Git/combined statuses pass and the specifically authorized zero-Data route is admitted; all-bounds-pass remains false because Data floor fails. No move/delete/cleanup or experiment was used to make room.','',
'| Stage | Child/capture | Scoped seconds | Physical forecast | Actual allocation |','|---|---:|---:|---:|---:|']
for r in ops:lines.append(f"| {r['name']} | {r['child_exit_code']}/{r['capture_exit_code']} | {r['elapsed_seconds']:.6f} | {r['forecast_physical_bytes']} | {r['actual_allocated_bytes']} |")
lines+=['','Costs above are separately timed launcher regions; child CPU is retained in exact finishes/index. Enclosing editing/review/project costs remain unknown and nested clocks are not summed. Fix administration/closure receipts supply their own later boundaries.','',
'New exact-source audit authenticates both original CPU/native startup refs and finds only src/ralup/report.py changed among78 src files. The current module has focused administrative verification, not a claimed current-tree1250 pass or native rerun. H07 complete-state cross-source rejection and explicit model-only reset contracts remain unchanged. Root owns any genuinely covering final integration after all fixes under a separate admission.','',
'Exact new assets:','']
for k,r in assets.items():lines.append(f'- {k}: `{r["path"]}` SHA256 `{r["sha256"]}` ({r["bytes"]}B).')
lines+=['',f'Fix administration: `{OUT}/finish.json`; source/UTC/runtime/full forecast in `{OUT}/startup.json`. Exact amended report/index and changed scratch are preserved under `{OUT}`. Closure binds their final hashes and commit in `fix1-completion.json` at the E07 Git root.','',
'Migration delta: authenticate initial history and prior final-admin delta, then preserve changed/new current scratch against that boundary, including FULL E07 review and current root rulings. Prior13 raw-report worktree input copies are retained and reauthenticated without copying unchanged bytes again; their original identities remain in the prior handoff mapping. New reports reference the same exact primary durable ledger bytes. No missing M01 whole-tree coverage or historical operands are inferred.','',
'Self-review checked the two traced branches, public fixtures, null preservation, operational versus historical refs, actual CLI/shell/figures, source applicability and independent resource-status dimensions. Fresh scoped review is still required. All11 Declined boundaries retain future owners; none authorizes additional work here. A later hash-linked root delta must still capture scoped rereview/fixes, whole-branch review, final rulings and finishing before only-own-plan scratch cleanup.','']
reportpath.write_text(reportpath.read_text()+'\n'.join(lines))
# Prior exact copies remain immutable; only changed/new scratch is copied.
initial=BASE/'preservation-v1/history-mapping.json';previous=GIT/'final-admin-v1/history-delta.json'
old={r['original_path']:r['sha256'] for r in json.loads(initial.read_bytes())['files']}
for r in json.loads(previous.read_bytes())['files']:old[r['original_path']]=r['durable']['sha256']
pending=[]
for p in sorted((ROOT/'.superpowers/sdd/2026-10-03-ralup').rglob('*')):
 if p.is_file() and old.get(str(p))!=ref(p)['sha256']:pending.append(p)
forecast=sum(((p.stat().st_size+4095)//4096)*4096 for p in pending)+512*1024
assert forecast<4*m.MIB
(OUT/'delta-forecast.json').write_bytes(m.encoded({'physical_bytes':forecast,'changed_files':len(pending),'scope':'scratch delta plus512KiB report/index/handoff margin'}))
changed=[]
for p in pending:
 dest=OUT/'history-delta'/p.relative_to(ROOT/'.superpowers/sdd/2026-10-03-ralup');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
 changed.append({'original_path':str(p),'prior_sha256':old.get(str(p)),'durable':ref(dest)})
(OUT/'history-delta.json').write_bytes(m.encoded({'previous_mapping':ref(previous),'initial_mapping':ref(initial),'capture_finished_utc':datetime.now(timezone.utc).isoformat(),'files':changed,'later_root_delta_required':True}))
old_handoff=GIT/'final-admin-v1/handoff-mapping.json'
for r in json.loads(old_handoff.read_bytes())['files']:assert ref(Path(r['durable']['path']))['sha256']==r['durable']['sha256']
files=[]
for p in [indexpath,ROOT/'README.md',ROOT/'docs/harness/e07-contract.md',ROOT/'progress.json',ROOT/'decisions.md']:
 dest=OUT/'handoff'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());files.append({'original_path':str(p),'durable':ref(dest)})
(OUT/'handoff-mapping.json').write_bytes(m.encoded({'prior_handoff_authenticated':ref(old_handoff),'files':files,'report':ref(OUT/'history-delta/E07-report.md')}))
subprocess.run(['git','diff','--check'],check=True)
print(json.dumps({'index':ref(indexpath),'report':ref(reportpath),'history_delta':ref(OUT/'history-delta.json'),'changed_scratch_files':len(changed),'git_diff_check':'pass'}))
