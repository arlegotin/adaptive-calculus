"""Authenticate E01 receipts, record the supported gate, and commit local sources.

Administrative only: run once through e01-attempt.py in a new owned directory.
This does not rerun tests, admit numerics, or perform scientific scoring.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

root = Path.cwd()
attempt = Path(os.environ['E01_ATTEMPT_DIR'])
base = attempt.parent
baseline = 'f2d9a40e05d400919f907b9311c0adff340d50c6'
subject = 'feat: add fair finalist selection and full cost accounting'


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def git(*args):
    result = subprocess.run(['git', *args], capture_output=True, text=True, check=True)
    print(json.dumps({'command': ['git', *args], 'stdout': result.stdout, 'stderr': result.stderr}))
    return result.stdout.strip()


assert git('rev-parse', 'HEAD') == baseline
assert git('branch', '--show-current') == 'feat/ralup-execution'
records = []
for directory in sorted(base.iterdir()):
    if directory == attempt:
        continue
    startup = json.loads((directory/'startup.json').read_bytes())
    finish = json.loads((directory/'finish.json').read_bytes())
    objects = {row['path']: row for row in finish['completed_objects']}
    files = {}
    for name in ('startup.json', 'finish.json', 'stdout.txt', 'stderr.txt', 'runtime-observation.json'):
        files[name] = ref(directory/name)
        if name != 'finish.json':
            assert objects[name]['sha256'] == files[name]['sha256']
    for relative, identity in startup['source'].items():
        assert ref(directory/'producer'/relative)['sha256'] == identity['sha256']
    output = (directory/'stdout.txt').read_text()
    records.append({'attempt': directory.name, 'utc': startup['utc'], 'files': files,
                    'command': startup['command'], 'exit_code': finish['exit_code'],
                    'summary': output.splitlines()[-1:] or [], 'elapsed_seconds': finish['elapsed_seconds'],
                    'storage_before': {key: value for key, value in startup['storage'].items() if key != 'inventory'},
                    'git_free_after': finish['git_free_after'], 'data_free_after': finish['data_free_after']})
records.sort(key=lambda row: row['utc'])
focused = json.loads((base/'focused-final/startup.json').read_bytes())
cli = json.loads((base/'select-finalists-v1/startup.json').read_bytes())
for path in sorted(root.glob('src/ralup/**/*.py')):
    key = str(path.relative_to(root))
    assert ref(path)['sha256'] == focused['source'][key]['sha256'] == cli['source'][key]['sha256']
    ast.parse(path.read_text(), filename=key)
for name in ('test_finalist_selection.py', 'test_cost_ledgers.py', 'test_headroom_selection.py', 'test_meta_selection.py'):
    key = 'tests/unit/' + name
    assert ref(root/key)['sha256'] == focused['source'][key]['sha256']
selection_ref = ref(base/'select-finalists-v1/selection.json')
selection = json.loads(Path(selection_ref['path']).read_bytes())
assert selection['B_star'] is None and selection['B0'] is None and selection['WP06'] == 'NEEDS_REVISION'
assert selection['completed_tournament'] is False and selection['defeated_methods'] == []
git('diff', '--check')
observed = datetime.now(timezone.utc).isoformat()
launcher_seconds = sum(row['elapsed_seconds'] for row in records)
table = '\n'.join('| ' + ' | '.join((row['attempt'], str(row['exit_code']), row['summary'][0], f"{row['elapsed_seconds']:.6f}")) + ' |' for row in records)
commands = '\n\n'.join('`' + row['attempt'] + '`\n\n```text\n' + ' '.join(row['command']) + '\n```' for row in records)
report = f'''# E01 implementation and scientific disposition

Status: **DONE_WITH_CONCERNS** for the authorized implementation and supported prerequisite disposition. The scientific tournament is **not completed**. WP04 is inconclusive, WP06 is NEEDS_REVISION, and strongest B0/B* remain unresolved. No candidate, LR-only or static finalist was selected; no family was defeated.

Observed UTC: {observed}. Research cutoff: 2026-10-03. The e01-20261011 directory is a historical task label; no receipt date is backdated. Base: `{baseline}`. Branch: `feat/ralup-execution`. Requested local commit subject: `{subject}`. No merge, push or publication is part of this task.

## Implemented behavior

The exact `allocate_search`, `select_finalists` and `cost_quality_frontier` APIs, ordinary search orchestration, four explicit CLI actions, immutable finalist exports, authenticated prerequisite disposition and engineering fixtures are implemented. Equal initial seconds reserve includes blocked families; variable trial counts follow finite forecasts. Extensions require authenticated comparison curves and specific edge, uncertainty, unsettled or missing-mechanism reasons. Generic improvements and development information are shared. Candidate/LR/static allocation requires full fourteen-feature/history/EMA/meta information and each family's own-current-policy refresh; allocation labels do not confer learned-policy eligibility.

Ordinary factory routes use existing matrix/EWGS/TRS/QuEST/learned-scale/damping/masking/QAR/stochastic/online-MetaQuant implementations. Ten tiny CPU adapter cases take two actual H06 steps. Initialized frozen MetaQuant fails with a specific missing genuine trained-theta prerequisite. Constructor support does not confer native parity or scientific competitiveness. The M11 policy campaign remains under its own APIs and two-stage authority; ordinary search cannot substitute for it.

Fresh full-budget selection requires a prospective source-bound replication design, exact frozen configs, identical fresh seeds disjoint from tuning, complete family disposition inventory, retained failed results, startup design binding and completed producer receipts, full official-development query coverage, authentic tuning judgments and complete parent/work accounting. Bootstrap booleans cannot promote old rows. B* ranking is independent of B0; mean development BPB wins, then complete end-to-end seconds, then recipe hash. Existing M11 authority is exploratory/null-B0, so any future nomination under this contract remains confirmation-ineligible. The successful measured full-budget path has not been exercised because its prerequisites are absent.

H06 now records each actually launched loss microbatch, its offsets and forward/backward status without changing training mathematics. Physical loss, gradient-only and HVP event IDs are distinct; query/additional-loss/backward classifications can overlap one physical event. EWGS forward and backward roles share one loss execution. Cost totals deduplicate ancestry and event IDs, union global raw-source offsets and enclosing wall intervals, preserve failed/partial work, and retain unknown historical counts rather than invent old IDs. Curves retain all observed milestones and require ordered execution prefixes. Unknown ancestor work propagates to totals; common engineering time remains unknown. Raw ordinary methods pay their actual execution costs without unused policy features.

Strict H07 all-source identity remains intact. No old complete pool was restored after edits. No state compatibility framework, pickle, dependency change, protected scoring, normalizer mathematics change, historical evidence overwrite or native calibration wave was introduced. Model-only import remains typed/authenticated, retaining exact parameters, auxiliary tensors, cursor, named RNG and parent costs with fresh local moments/history/schedule. The historical normalizer is DATA only.

## Scientific judgment and required owners

The actual CLI authenticates indexed M07 B0-null/WP04-inconclusive evidence and indexed M11 NEEDS_REVISION plus its diagnostic inputs. Its immutable output is `{selection_ref['path']}`, SHA256 `{selection_ref['sha256']}`. It records no completed tournament, measured non-advantage or global unaffordability. All sixteen named ordinary/policy families remain unresolved. Frozen MetaQuant lacks genuine META training/selection; its Muon/Muon+ composition is unverified and SOAP/Madam composition unsupported. This does not exclude those ordinary backbones from B0/B* or call them defeated.

No new scientific measurement was run. Existing M11 evidence concerns two provisional one-step vectors whose short-horizon ranking reverses at full remaining horizon; they are not eligible learned-family campaigns. Another arbitrary short ES retry would not resolve final K/m/sigma/iterations/refresh or matched learning. A bounded ordinary bootstrap comparison would not identify those missing budgets or establish the strongest full-budget backbone. The minimally scoped decision was an authenticated prerequisite exit, not an invented replacement campaign. Cost if wrong: genuinely informative longer-horizon calibration could be delayed; reopening WP06 must prospectively justify that design from the retained evidence. A resource cap was not interpreted as global unaffordability.

E01/reopened WP06 owns prospective longer-horizon/objective calibration and evidence-based learning budgets; matched candidate45/LR49/static3 learning with each own-policy refresh; META-development then full official-development selection; accepted B0/recipe applicability or retraining. E01 retains independent robust B0 and family backbone-LR tuning, every locally competitive ordinary family, LR/betas/decay/schedule brackets, genuine frozen MetaQuant training/selection, full-budget fresh development replication, shared generic improvements and data/compute-augmented ordinary training. E05 retains shared-range learned-LR sensitivity and mechanism/removal contrasts. E07 owns global ancestry/source-offset/cost reconciliation. None of these required measurements is marked passed by this report.

## Verification, including failures

The v7 TDD, debugging and verification workflow was applied. Initial RED was 14 genuine missing-interface failures; later REDs exercised runner accounting, missing converter/extra-loss identities, bootstrap authority, adapter routing, slow crossing milestones, unknown ancestor costs and missing earlier checkpoint executions. Some attempts named green still failed; their actual exit/result is retained below.

| Attempt | Exit | Observed pytest/CLI summary | Enclosing seconds |
| --- | ---: | --- | ---: |
{table}

The full CPU covering snapshot is **929 passed, 76 skipped, 3 failed**, on its archived source. It first observed actual MPS availability and then used a test-only CPU override; it is not native admission. All three failures were the existing M07 partial-work cases for training/query/query-cap. Newly serialized query blocks retained tuple fields from `asdict`, violating the existing strict JSON contract before those cases could reach their intended failures. The fix emits explicit LISTS for public block shape/offset metadata; no validator was weakened.

Final current-source focused verification is **105 passed in 2.81s**, including those three cases and new ancestor/prefix regressions. Exact final src hashes equal the focused and actual CLI archives. The full suite was not repeated after those targeted fixes, so this report does not claim a current full-suite pass. Earlier failures also exposed a fake test clock missing `time`, a newline mismatch in exported content hashes, incomplete explicit fixture EWGS/RNG setup, and a duplicate default event list; each was corrected narrowly and covered by final focused tests.

Self-review checked search allocation/authority, H06 actual loss logging, CLI action boundaries, all cost logic and retained evidence applicability. Static AST parsing and `git diff --check` passed at finalization. Parent coordinator must still perform fresh full `{baseline}..HEAD` review; no child reviewer was spawned. The unexecuted genuine positive full-budget authority route and unqualified native adapter compositions remain explicit limitations, not inferred passes.

Every command below ran through `PYTORCH_ENABLE_MPS_FALLBACK=0 E01_FORECAST_BYTES=67108864 .venv/bin/python docs/harness/e01-attempt.py ATTEMPT MODULE [ARGS]`. The startup records contain exact argv, full archived source, Git diff and runtime; the complete stdout/stderr and finish receipts are retained. Commands:

{commands}

## Runtime, resources and accounting scope

Observed native host: macOS 26.7.1 build 25G241 arm64, Python 3.14.6, Torch distribution 2.14.1; fallback=0. Pinned lock SHA256 `488142a35a4b1a6abe132f08b5895621ca30bb535b0326e82516446903fcb32b`. Historical F02 bytes are archived context, not current admission. Tests include actual tiny CPU model work; no new native scientific model run occurred.

All nineteen completed attempt receipts were preserved and authenticated. Their non-overlapping enclosing launcher durations sum to **{launcher_seconds:.9f} seconds**, before this administration. Nested pytest/scorer/step regions are not added. This is only the enumerated launcher scope, not total project or engineering time. Editing, tools, coordinator/review, earlier tasks and uninstrumented CPU fixture target counts remain unknown; no aggregate target count was fabricated. No new scientific stream/query/gradient/HVP measurements occurred. Existing prerequisite work remains in its original receipts for E07 global reconciliation.

Before each attempt, the complete external store plus prospective forecast was checked against 32 GiB and Data free space minus forecast against 64 GiB. The small-source/Git forecast was 16 MiB with at least 512 MiB retained. Every attempt retained before/after free-space observations; startup source archives preceded child imports and finish records followed flushed stdout/stderr. The index records exact per-attempt storage values. The administration receipt records its own additional wall/CPU/IO and complete-store inventory, without adding itself twice. No old evidence was moved, deleted, overwritten or relabeled.

Tracked evidence index: `docs/harness/e01-evidence.json`. Contract/applicability: `docs/harness/e01-contract.md`. This report is also copied into a new immutable task-owned administration path. The commit receipt and post-commit copy are covered by its completed finish receipt; they are not predeclared scientific evidence.
'''
report_path = root/'.superpowers/sdd/2026-10-03-ralup/E01-report.md'
report_path.write_text(report)
(attempt/'report-precommit.md').write_text(report)
index = {'schema': 'ralup.e01_evidence.v1', 'created_utc': observed, 'research_cutoff': '2026-10-03',
         'base_commit': baseline, 'attempts': records, 'selection': selection_ref,
         'report_precommit': ref(attempt/'report-precommit.md'),
         'final_administration_receipt_location': str(attempt/'finish.json'),
         'post_commit_report_location': str(attempt/'report.md'),
         'administration_locations_are_not_yet_completed_evidence': True,
         'final_verified_source': {key: value['sha256'] for key, value in focused['source'].items() if key.startswith('src/')},
         'verification': {'full_cpu_archived_snapshot': '929 passed, 76 skipped, 3 failed',
                          'final_focused': '105 passed in 2.81s', 'current_full_suite_pass_claimed': False,
                          'native_scientific_admission_performed': False},
         'scientific_status': {'WP04': 'inconclusive', 'WP06': 'NEEDS_REVISION', 'B0': None, 'B_star': None,
                               'completed_tournament': False, 'confirmation_eligible': False},
         'cost': {'enumerated_completed_launchers_seconds': launcher_seconds, 'common_engineering_seconds': None,
                  'total_project_seconds': None, 'all_cpu_fixture_target_counts': None,
                  'administration_included_in_sum': False, 'nested_regions_added': False}}
write(root/'docs/harness/e01-evidence.json', index)
files = ['src/ralup/__main__.py', 'src/ralup/search.py', 'src/ralup/train.py',
         'src/ralup/analysis/__init__.py', 'src/ralup/analysis/cost.py',
         'tests/unit/test_headroom_selection.py', 'tests/unit/test_finalist_selection.py',
         'tests/unit/test_cost_ledgers.py', 'configs/e01-gated-selection.json',
         'docs/harness/e01-attempt.py', 'docs/harness/e01-contract.md',
         'docs/harness/e01-cpu-sitecustomize.py', 'docs/harness/e01-pytest-cpu.py',
         'docs/harness/e01-finalize.py', 'docs/harness/e01-evidence.json']
assert sum((root/name).stat().st_size for name in files) < 16*1024**2
git('diff', '--check')
git('add', '--', *files)
git('diff', '--cached', '--check')
assert set(git('diff', '--cached', '--name-only').splitlines()) == set(files)
git('commit', '-m', subject)
head = git('rev-parse', 'HEAD')
status = git('status', '--short')
assert not status
report += f'\n## Local commit\n\n`{head}` — `{subject}`. Working tree clean after commit. Parent fresh full-base review remains required.\n'
report_path.write_text(report)
(attempt/'report.md').write_text(report)
write(attempt/'commit.json', {'base': baseline, 'head': head, 'subject': subject,
                             'status': status, 'report': ref(attempt/'report.md'),
                             'index': ref(root/'docs/harness/e01-evidence.json'),
                             'observed_utc': datetime.now(timezone.utc).isoformat()})
print(json.dumps({'status': 'DONE_WITH_CONCERNS', 'head': head, 'report': str(report_path),
                  'permanent_report': str(attempt/'report.md'), 'selection': selection_ref}))
