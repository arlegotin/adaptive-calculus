"""Repair one current operational index ref; historical files are never rewritten."""
from datetime import datetime,timezone
import importlib.util,json,os,subprocess
from pathlib import Path
spec=importlib.util.spec_from_file_location('capture',Path(__file__).with_name('e07-capture.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path.cwd();OUT=Path(os.environ['E07_OPERATION']);BASE=Path(os.environ['E07_BASE']);GIT=Path(os.environ['E07_COMPLETION_ROOT']);ref=m.ref

def authenticate_index_refs(value):
    """Check complete operational refs contained in THIS index only, not referenced file contents."""
    checked=[];bad=[]
    def walk(v,where='$'):
        if isinstance(v,dict):
            if isinstance(v.get('path'),str) and isinstance(v.get('sha256'),str):
                try:
                    actual=ref(Path(v['path']));valid=actual['sha256']==v['sha256'] and ('bytes' not in v or actual['bytes']==v['bytes'])
                except OSError as error:actual={'error':str(error)};valid=False
                row={'location':where,'expected':v,'actual':actual,'valid':valid};checked.append(row)
                if not valid:bad.append(row)
            for k,x in v.items():walk(x,where+'.'+k)
        elif isinstance(v,list):
            for i,x in enumerate(v):walk(x,where+f'[{i}]')
    walk(value)
    return {'unique_path_sha_pairs':len({(r['expected']['path'],r['expected']['sha256']) for r in checked}),'checked_count':len(checked),'valid_count':len(checked)-len(bad),'failures':bad,'checked':checked,
        'scope':'complete path/sha256 refs contained in the current index; no recursive opening of referenced JSON or historical archive paths'}

indexpath=ROOT/'docs/harness/e07-evidence.json';index=json.loads(indexpath.read_bytes())
prior_current=ref(indexpath);assert prior_current['sha256']=='86a6c2b04740d0291825ac7a790b0f38b6119b4b682964a89f3e9e76d5a4daf5'
red=authenticate_index_refs(index);assert len(red['failures'])==1 and red['failures'][0]['location']=='$.fix1.prior_index',red['failures']
(OUT/'red-authentication.json').write_bytes(m.encoded(red))
original=index['fix1']['prior_index'];durable=ref(GIT/'final-admin-v1/handoff/docs/harness/e07-evidence.json')
assert durable['sha256']==original['sha256']=='6ee31d1c46a0b8b572946d8aae487739bcb2e148bae2aafd1761ae8bd5f50921'
assert durable['bytes']==original['bytes']==40921
index['fix1']['prior_index']=durable
index['fix1']['prior_index_original_location']={'kind':'historical_location_only','original_path':original['path'],
 'original_sha256':original['sha256'],'original_bytes':original['bytes'],'meaning':'location at initial capture, not a current operational ArtifactRef'}
# The previous index bytes are already immutable in this capture's source archive.
index['fix2']={'base':'af305faf0a8c31a6a7fcffbb517cb088f653c917','status':'implemented_pending_fresh_scoped_review',
 'previous_index':ref(OUT/'producer/docs/harness/e07-evidence.json'),
 'failed_first_verifier_attempt':{'startup':ref(GIT/'fix2-admin-v1/startup.json'),'finish':ref(GIT/'fix2-admin-v1/finish.json'),'stderr':ref(GIT/'fix2-admin-v1/stderr.txt'),'scope':'guard incorrectly equated occurrence count with root unique-ref count; one intended mismatch, no publication'},'original_index_authenticated':durable,
 'admin_startup':ref(OUT/'startup.json'),'admin_finish_path':str(OUT/'finish.json'),
 'verification':'actual current index RED then corrected complete operational refs GREEN; referenced historical JSON contents are not recursively reinterpreted',
 'source_applicability':'metadata/administrative helpers only; src/report.py and all product code unchanged from fix1; prior CPU/native scope remains historical',
 'late_root_delta_required':True}
index['status']='fix2_implemented_pending_fresh_scoped_review'
green=authenticate_index_refs(index);assert not green['failures'],green['failures']
(OUT/'green-authentication.json').write_bytes(m.encoded(green))
# Regression: reintroducing precisely the original mutable-path operand must fail.
regression=json.loads(json.dumps(index));regression['fix1']['prior_index']=original
rejected=authenticate_index_refs(regression);assert len(rejected['failures'])==1 and rejected['failures'][0]['location']=='$.fix1.prior_index'
(OUT/'regression.json').write_bytes(m.encoded({'reintroduced_mutable_path_rejected':True,'failure':rejected['failures'][0],
 'durable_original':durable,'red_counts':[red['checked_count'],red['valid_count']],'green_counts':[green['checked_count'],green['valid_count']]}))
indexpath.write_bytes(m.encoded(index))
# Authenticate publication too; no self-ref is inserted into the index.
published=authenticate_index_refs(json.loads(indexpath.read_bytes()));assert not published['failures']
assert ref(Path(durable['path']))==durable
p=ROOT/'progress.json';progress=json.loads(p.read_bytes());progress['engineering_status']['E07']='fix2_implemented_pending_fresh_scoped_review';progress['engineering_status']['E07_fix2']={'finding':'mutable operational prior_index','base':'af305faf0a8c31a6a7fcffbb517cb088f653c917','product_source_changed':False,'verification':'current index ref authentication and actual-path regression only'};p.write_bytes(m.encoded(progress))
reportpath=ROOT/'.superpowers/sdd/2026-10-03-ralup/E07-report.md';startup=json.loads((OUT/'startup.json').read_bytes())
lines=['','## E07 fix round2 — operational prior_index repair','',
'BASE af305faf0a8c31a6a7fcffbb517cb088f653c917. Root reopened E07 after the clean fix1 scoped verdict when current-index authentication found one operational reference mismatch. The earlier complete entry is historical/superseded; engineering count remains31/32 until repair and fresh review. Read E07-fix2-findings.md and the full progress/state reopening.','',
'The fix1 administration producer had captured the old index hash at the mutable worktree path and then overwritten that current index. Current $.fix1.prior_index therefore advertised40921B/SHA6ee31d1c46a0b8b572946d8aae487739bcb2e148bae2aafd1761ae8bd5f50921 at a path holding62457B/SHA86a6c2b04740d0291825ac7a790b0f38b6119b4b682964a89f3e9e76d5a4daf5. This was an introduced metadata/operational-ref defect, not a numerical failure.','',
'Authenticated the exact original immutable final-admin-v1/handoff/docs/harness/e07-evidence.json bytes and bound the CURRENT operational prior_index to that copy. The original worktree location/hash/size are explicit historical_location_only metadata using original_path/original_sha256/original_bytes, not a pretending current ArtifactRef. Corrected the administration producer to use that exact immutable copy. No old JSON, ArtifactRef, receipt, source hash, report version, figure or claimed identity was rewritten.','',
f'Actual RED authenticated{red["checked_count"]} complete current-index ref occurrences: {red["valid_count"]} valid and exactly $.fix1.prior_index mismatched. After repair and fix2 metadata, GREEN authenticated{green["checked_count"]}/{green["valid_count"]} refs, zero failures; publication was checked again. Regression reintroduces exactly the original mutable path and must fail only that operand. The verifier walks complete operational refs embedded in the current index, not the contents of referenced historical JSON; it is not a general relocation or recursive archive adapter.','',
'The first fix2-admin-v1 capture failed before publication because its verifier hardcoded the root140 count while walking duplicate embedded occurrences. The actual sole mismatch was still prior_index. Its original child/capture1, traceback,2,240,512B allocation and exact cost remain retained. The corrected verifier reports occurrence and unique path/hash counts separately, requires exactly the named RED mismatch, and requires every published ref to authenticate. New v2 was admitted with a fresh full9MiB remaining forecast; no v1 output was overwritten.', '',
'Product report.py and all src code are unchanged from fix1. Existing16 focused report tests, accepted raw report/CLI/shell/figures and old1250/native evidence were not rerun or regenerated. Their prior source scopes remain intact. No model/training/query/numerical/MPS/dependency work occurred. Root separately owns final-current-product integration after the actual fixes.','',
f'Fresh prospective admission: full FIX2 remaining footprint9MiB (4MiB child+4MiB administration+1MiB closure), starting E07 Git allocation{startup["git_completion_before"]["allocated_bytes"]}B, planned total{startup["git_completion_before"]["allocated_bytes"]+9*m.MIB}B <160MiB. Git free{startup["git_free_before"]}B retains1GiB plus16MiB source. Combined Data/E07/E06 evidence stays within32GiB. Raw Data{startup["raw_data_free_before"]}B remains below64GiB and is separately false; zero-owned-Data administration is explicitly authorized. All temp/cache/output belongs to the new Git stage; exact postflight enforces child/admin/Git/combined bounds and unchanged Data inventory. No deletion/move or inferred floor waiver.','',
'Exact proof pointers:','']
for name in ('red-authentication.json','green-authentication.json','regression.json'):
 r=ref(OUT/name);lines.append(f'- `{r["path"]}` SHA256 `{r["sha256"]}` ({r["bytes"]}B).')
lines += [f'- Immutable original index: `{durable["path"]}` SHA256 `{durable["sha256"]}` ({durable["bytes"]}B).',
 f'- Capture source/UTC/runtime/forecast `{OUT}/startup.json`; actual status/allocation/cost `{OUT}/finish.json`; final commit binding `fix2-completion.json` in the E07 Git root.','',
'Prior complete-index mismatch and clean-review/reopening chronology remain preserved. New late-record delta preserves changed scratch, including fix2 finding, current root contexts, interim v9 all82 rulings and updated whole-branch requirements where present. All older copies remain. Later root scoped review/fixes, whole-branch review, final rulings and finishing still require a subsequent hash-linked delta before only-own-plan cleanup. No scope from the11 Declined boundaries is consumed here.','',
'Self-review: checked exact original byte identity, explicit historical metadata typing, every complete current operational ref, publication and concrete regression, unchanged product source, retained failures and separate resource dimensions. Fresh scoped review remains required; this is not a self-accepted task closure. Launcher/child costs are in the new finish; enclosing editing/review/project costs remain unknown.','']
reportpath.write_text(reportpath.read_text()+'\n'.join(lines))
# Capture only changed/new scratch against the three existing durable boundaries.
initial=BASE/'preservation-v1/history-mapping.json';boundaries=[GIT/'final-admin-v1/history-delta.json',GIT/'fix1-admin-v1/history-delta.json']
old={r['original_path']:r['sha256'] for r in json.loads(initial.read_bytes())['files']}
for boundary in boundaries:
 for r in json.loads(boundary.read_bytes())['files']:old[r['original_path']]=r['durable']['sha256']
pending=[p for p in sorted((ROOT/'.superpowers/sdd/2026-10-03-ralup').rglob('*')) if p.is_file() and old.get(str(p))!=ref(p)['sha256']]
forecast=sum(((p.stat().st_size+4095)//4096)*4096 for p in pending)+1024**2
assert forecast<4*m.MIB,forecast
(OUT/'delta-forecast.json').write_bytes(m.encoded({'physical_bytes':forecast,'changed_files':len(pending),'scope':'scratch delta plus1MiB verifier/index/handoff margin'}))
changed=[]
for p in pending:
 dest=OUT/'history-delta'/p.relative_to(ROOT/'.superpowers/sdd/2026-10-03-ralup');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());changed.append({'original_path':str(p),'prior_sha256':old.get(str(p)),'durable':ref(dest)})
(OUT/'history-delta.json').write_bytes(m.encoded({'initial_mapping':ref(initial),'previous_mapping':ref(boundaries[-1]),'capture_finished_utc':datetime.now(timezone.utc).isoformat(),'files':changed,'later_root_delta_required':True}))
files=[]
for p in [indexpath,ROOT/'progress.json']:
 dest=OUT/'handoff'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());files.append({'original_path':str(p),'durable':ref(dest)})
(OUT/'handoff-mapping.json').write_bytes(m.encoded({'files':files,'report':ref(OUT/'history-delta/E07-report.md'),'immutable_prior_index':durable}))
subprocess.run(['git','diff','--check'],check=True)
print(json.dumps({'index':ref(indexpath),'report':ref(reportpath),'red':[red['checked_count'],red['valid_count']],'green':[green['checked_count'],green['valid_count']],'regression':'pass','delta':ref(OUT/'history-delta.json')}))
