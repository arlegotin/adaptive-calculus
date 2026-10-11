"""Finalize only the bounded E01 I1/I2 review fix; no tests or science here."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path.cwd();attempt=Path(os.environ['E01_ATTEMPT_DIR']);base=attempt.parent
fixbase='e2e8ad71f7291f2bb908532191eca0227ba7e90a'
subject='fix: authenticate finalist seed and parent lineage'


def ref(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def git(*args):
    result=subprocess.run(['git',*args],capture_output=True,text=True,check=True)
    print(json.dumps({'command':['git',*args],'stdout':result.stdout,'stderr':result.stderr}))
    return result.stdout.strip()


assert git('rev-parse','HEAD')==fixbase
assert git('branch','--show-current')=='feat/ralup-execution'
records=[]
for directory in sorted(base.glob('fix1-*')):
    if directory==attempt:continue
    startup=json.loads((directory/'startup.json').read_bytes());finish=json.loads((directory/'finish.json').read_bytes())
    objects={row['path']:row for row in finish['completed_objects']};files={}
    for name in ('startup.json','finish.json','stdout.txt','stderr.txt','runtime-observation.json'):
        files[name]=ref(directory/name)
        if name!='finish.json':assert objects[name]['sha256']==files[name]['sha256']
    for name,row in startup['source'].items():
        assert ref(directory/'producer'/name)['sha256']==row['sha256']
    output=(directory/'stdout.txt').read_text()
    records.append({'attempt':directory.name,'utc':startup['utc'],'files':files,'command':startup['command'],
        'exit_code':finish['exit_code'],'summary':output.splitlines()[-1],'elapsed_seconds':finish['elapsed_seconds'],
        'forecast_bytes':startup['storage']['complete_remaining_fix_forecast_bytes'],
        'storage_before':{k:v for k,v in startup['storage'].items() if k!='inventory'},
        'git_free_after':finish['git_free_after'],'data_free_after':finish['data_free_after']})
records.sort(key=lambda row:row['utc'])
assert len(records)==6
final=json.loads((base/'fix1-focused-final-v2/startup.json').read_bytes())
for path in root.glob('src/ralup/**/*.py'):
    name=str(path.relative_to(root))
    assert ref(path)['sha256']==final['source'][name]['sha256']
    ast.parse(path.read_text(),filename=name)
for name in ('test_finalist_selection.py','test_headroom_selection.py','test_meta_selection.py'):
    relative='tests/unit/'+name
    assert ref(root/relative)['sha256']==final['source'][relative]['sha256']
selection=ref(base/'select-finalists-v1/selection.json')
assert selection['sha256']=='00f6bb750d3c6602ee1eb00afcaabdeda3ec7f2b6f08b740bd04ac45e87be1f3'
report_path=root/'.superpowers/sdd/2026-10-03-ralup/E01-report.md'
old_report=report_path.read_text()
assert old_report==(base/'finalize-v1/report.md').read_text()
git('diff','--check')
table='\n'.join(f"| {r['attempt']} | {r['exit_code']} | {r['summary']} | {r['elapsed_seconds']:.6f} |" for r in records)
commands='\n\n'.join('```text\nPYTORCH_ENABLE_MPS_FALLBACK=0 E01_FORECAST_BYTES='+str(r['forecast_bytes'])+
    ' .venv/bin/python docs/harness/e01-attempt.py '+r['attempt']+' pytest '+
    ' '.join(r['command'][3:])+'\n```' for r in records)
seconds=sum(r['elapsed_seconds'] for r in records)
observed=datetime.now(timezone.utc).isoformat()
appendix=f'''

## E01 review fix round 1 — I1 and I2

Observed UTC: {observed}; research cutoff remains 2026-10-03. Fix base `{fixbase}`. Status: **DONE_WITH_CONCERNS**, pending coordinator fresh scoped rereview. Reviewed FULL E01-review.md and the verbatim two Important findings in E01-fix1-findings.md under the v7 receiving-code-review/TDD/debugging/verification workflow. Sole worker; no child or reviewer spawned.

I1 was confirmed: the selector compared declared tuning/fresh lists, then read tuning results without deriving their used seeds. I2 was confirmed: fresh construction's seed was overwritten by the exact imported parent W/cursor/all-RNG trajectory; distinct labels could count duplicate complete continuations as independent units.

The design now freezes every retained tuning result/config and source-authenticated producer receipt. All tuning assessment references must exhaust that exact frozen inventory. Used tuning seeds come from authenticated results/configs and their actual seed-specific parent lineages, must equal the frozen tuning list, and must be disjoint from every replication seed. Failed retained tuning attempts remain in this used-seed inventory rather than becoming fresh opportunities.

Every tuning/replication seed now requires a complete origin and ordered warm/training parent inventory. The origin checkpoint has actual step-zero W, cursor and named/global RNG arrays. Validation reproduces the existing CPU initialization (no forward/update), compares actual weights and the post-initialization RNG, reconstructs the initial indexed cursor/permutation/data RNG, and verifies the quantizer/meta/timing states. Producer source archives and actual snapshot tensor bytes are authenticated. Typed subsequent derived parents must link their actual previous checkpoint/artifact and prospective trial config; every method within one seed must use that seed's exact terminal parent. Actual origin and terminal-state fingerprints must differ across independent seeds; different filenames, report hashes or seed labels cannot manufacture independent units.

No import, reset, owner, cursor, RNG, optimizer, training or cost implementation changed. MODELONLY still preserves exact W/aux/cursor/named/global RNG/parent cost and resets local moments/history/schedule. No compatibility framework was introduced. Historical origins without this complete current-source producer inventory remain ineligible. The existing typed derived schema's aggregate-only parent cost remains unknown; lineage admission cannot supply missing physical execution IDs.

### Focused RED/GREEN and the retained derived-cost finding

The prerequisite-shaped finite fixture uses real typed snapshots, source archives, source/config/result/producer authentication, indexed local cursor bytes, actual development query-byte validation and the real cost converter. Only DATA_INDEX points to a finite local corpus. Tiny CPU warmstart steps produce engineering checkpoint states; these are not a genuine scientific campaign, native gate, controller, full-budget experiment or finalist. The root positive case admits two distinct matched units across two methods.

The first RED retained six missing denials: actual tuning seed71 while declaring10, a tuning result absent from the frozen inventory, shared parent for seeds71/72, repackaged identical checkpoint, relabeled origin configuration, and mismatched parents across methods within one seed. After implementing these guards, all seven initial cases passed. Two additional origin-state RED cases showed that correct labels plus altered initial weights or quantizer RNG still needed a real seeded-state check; both now reject.

| Attempt | Exit | Actual terminal output | Enclosing seconds |
| --- | ---: | --- | ---: |
{table}

The `fix1-derived-lineage` attempt expected the complete authority to admit a typed derived parent. Its actual output was one failed/one passed: lineage validation succeeded, then the unchanged complete-physical-parent-cost check rejected the aggregate-only derived schema. That was an overbroad test expectation, not evidence to waive the cost requirement. The final test separately asserts successful typed lineage validation and the correct specific cost denial. It actually imports the warmstart through the existing MODELONLY path, executes a tiny continuation, publishes the typed derived record and authenticates its complete chain.

Final focused verification is **98 passed in 10.67s**, covering E01 finalist, existing M07 headroom and M11 selection files. All current src hashes and those test files equal this final startup archive. Cost implementation did not change, so no extra cost test or full/native suite was run. The original 929-pass/76-skip/3-fail covering snapshot and original 105-pass focused result remain immutable earlier-source evidence; no current full-suite pass is claimed. Final static AST parsing and `git diff --check` passed. Exact commands:

{commands}

Complete raw stdout/stderr, source archives, runtime observations, startup and completed finish refs for all six attempts are in `docs/harness/e01-fix1-evidence.json`. Attempts whose names contain green/final are reported by their actual exit/results; failed evidence was not overwritten or relabeled.

### Unchanged science, limitations and costs

The actual original selection remains SHA256 `{selection['sha256']}` at `{selection['path']}`. It was authenticated read-only, not rewritten on the changed source. WP04 stays inconclusive, WP06 stays NEEDS_REVISION, B0/B* and all finalists stay null. No stronger scientific conclusion, family defeat, completed tournament, global unaffordability or confirmation eligibility follows. E01/reopened WP06 retains the genuine learning, robust backbone/tuning/full-budget fresh replication and augmented ordinary work already listed above. No scientific/native/protected run occurred during this fix.

The new successful fixture establishes only the guarded admission boundary. No genuine current-source complete origin inventory or eligible campaign is supplied. Future native quantizer state validation uses the recorded generator representation; no imported RNG is converted or reseeded, and this CPU fixture does not grant native numerical admission. Actual selector construction/authentication time must remain in its enclosing receipt.

Minor M1 remains deferred to E07/root final review as instructed. Root's binding tie ruling is lower MEAN complete DEPLOYMENT end-to-end time per fresh full-budget run, including attributable common warm start and all inside-run work. Offline META/hyperparameter search is excluded from that tie but retained in separate search/full-project ledgers; unknown cost stays unknown. This fix makes no overall-efficiency claim and does not add all tuning to the tie.

The six non-overlapping fix-test launcher receipts total **{seconds:.9f} seconds** before final administration. Nested pytest durations are not added. Overall engineering/project time and aggregate fixture exposure counts remain unknown, not zero. The tests' tiny CPU training and initialization work belongs to engineering verification, not scientific development measurement. This administration has its separate complete wall/CPU/IO receipt. All writes use new task-owned paths; every wrapper remeasured the complete store plus forecast against 32 GiB, Data free minus forecast against 64 GiB, and a 16 MiB small-source forecast retaining at least 512 MiB on Git. Exact before/after observations are indexed. Runtime remains macOS26.7.1/build25G241 arm64, Python3.14.6/Torch2.14.1, fallback0 and unchanged pinned lock.

Scoped self-review covered the two reported bypasses, source-authenticated origin/parent traversal, tuning inventory completeness, immutable cost denial, no global RNG mutation during CPU origin reconstruction, and unchanged import semantics. Parent coordinator owns fresh scoped `{fixbase}..HEAD` rereview; root retains final whole-branch review after E07. No merge/push/publication occurred.
'''
report=old_report+appendix
report_path.write_text(report);(attempt/'report-precommit.md').write_text(report)
index={'schema':'ralup.e01_fix1_evidence.v1','created_utc':observed,'research_cutoff':'2026-10-03','fix_base':fixbase,
       'findings':['I1','I2'],'deferred_minor':'M1: E07/root final','attempts':records,'unchanged_original_selection':selection,
       'report_precommit':ref(attempt/'report-precommit.md'),
       'final_administration_receipt_location':str(attempt/'finish.json'),
       'post_commit_report_location':str(attempt/'report.md'),'administration_locations_are_not_yet_completed_evidence':True,
       'final_verified_source':{k:v['sha256'] for k,v in final['source'].items() if k.startswith('src/')},
       'verification':{'final_focused':'98 passed in 10.67s','current_full_suite_pass_claimed':False,'native_scientific_admission_performed':False},
       'cost':{'enumerated_test_launchers_seconds':seconds,'administration_included':False,
               'common_engineering_seconds':None,'total_project_seconds':None,'all_cpu_fixture_target_counts':None},
       'scientific_status':{'WP04':'inconclusive','WP06':'NEEDS_REVISION','B0':None,'B_star':None,'completed_tournament':False}}
write(root/'docs/harness/e01-fix1-evidence.json',index)
files=['src/ralup/search.py','tests/unit/test_finalist_selection.py','docs/harness/e01-contract.md',
       'docs/harness/e01-fix1-finalize.py','docs/harness/e01-fix1-evidence.json']
assert sum((root/name).stat().st_size for name in files)<16*1024**2
git('diff','--check');git('add','--',*files);git('diff','--cached','--check')
assert set(git('diff','--cached','--name-only').splitlines())==set(files)
git('commit','-m',subject);head=git('rev-parse','HEAD');status=git('status','--short');assert not status
report+=f'\n### Fix1 local commit\n\n`{head}` — `{subject}`. Working tree clean; fresh scoped rereview required.\n'
report_path.write_text(report);(attempt/'report.md').write_text(report)
write(attempt/'commit.json',{'fix_base':fixbase,'head':head,'subject':subject,'status':status,
                           'report':ref(attempt/'report.md'),'index':ref(root/'docs/harness/e01-fix1-evidence.json')})
print(json.dumps({'status':'DONE_WITH_CONCERNS','head':head,'report':str(report_path),'permanent_report':str(attempt/'report.md')}))
