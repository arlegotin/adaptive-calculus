"""Guard and record the authorized local M10 transaction; no model execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path.cwd()
STORE = Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03')
TASK = STORE / 'm10-20261011'
ACTIVE = Path(os.environ['M10_ATTEMPT_DIR'])
BASE = '33c8352a13babde50303a4ed072c05befe83243f'
PATHS = [
    'src/ralup/__main__.py', 'src/ralup/baselines/registry.py',
    'src/ralup/state.py', 'src/ralup/baselines/metaquant.py',
    'tests/integration/test_metaquant_adapter.py',
    'tests/unit/test_metaquant_fast_weights.py',
    'docs/harness/m10-attempt.py', 'docs/harness/m10-cpu-sitecustomize.py',
    'docs/harness/m10-evidence.json', 'docs/harness/m10-index.py',
    'docs/harness/m10-native-contract.json', 'docs/harness/m10-pytest-cpu.py',
    'docs/harness/m10-replay-diagnostic.py', 'docs/harness/m10-finalize.py',
]
commands = []


def ref(path):
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    record = {'command': args, 'exit_code': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr}
    commands.append(record)
    print(json.dumps(record, sort_keys=True), flush=True)
    if result.returncode:
        write_new(ACTIVE / 'failed-git-commands.json', commands)
        raise RuntimeError(f'Command failed: {args!r}')
    return result.stdout


def authenticate_attempt(name):
    base = TASK / name
    startup = json.loads((base / 'startup.json').read_bytes())
    finish = json.loads((base / 'finish.json').read_bytes())
    for relative, expected in startup['source'].items():
        actual = ref(base / 'producer' / relative)
        assert (actual['bytes'], actual['sha256']) == (expected['bytes'], expected['sha256'])
    for expected in finish['completed_objects']:
        actual = ref(base / expected['path'])
        assert (actual['bytes'], actual['sha256']) == (expected['bytes'], expected['sha256'])
    assert finish['exit_code'] == 0
    return startup, finish, {key: ref(base / f'{key}.json' if key in ('startup', 'finish')
                                     else base / f'{key}.txt')
                             for key in ('startup', 'finish', 'stdout', 'stderr')}


assert ROOT == Path('/Volumes/git/legotin/adaptive-calculus/.worktrees/ralup-execution')
assert run(['git', 'branch', '--show-current']).strip() == 'feat/ralup-execution'
assert run(['git', 'rev-parse', 'HEAD']).strip() == BASE
status = run(['git', 'status', '--short'])
assert {line[3:] for line in status.splitlines()} == set(PATHS)
run(['git', 'diff', '--check'])

native_start, native_finish, native_refs = authenticate_attempt('native-functional-v1')
cpu_start, cpu_finish, cpu_refs = authenticate_attempt('full-cpu-regression-3')
focused_start, _, _ = authenticate_attempt('focused-final-2')
index_start, index_finish, index_refs = authenticate_attempt('index-v1')
current = {str(p): ref(p)['sha256'] for p in sorted(Path('src/ralup').rglob('*.py'))}
native = json.loads((TASK / 'native-functional-v1/metaquant.json').read_bytes())
assert native['source_hashes'] == current
for name, digest in current.items():
    assert digest == native_start['source'][name]['sha256'] == cpu_start['source'][name]['sha256']
for name in ('tests/integration/test_metaquant_adapter.py', 'tests/unit/test_metaquant_fast_weights.py',
             'tests/unit/test_method_registry.py'):
    assert ref(Path(name))['sha256'] == focused_start['source'][name]['sha256']
pointer_path = Path('docs/harness/m10-evidence.json')
assert pointer_path.read_bytes() == (TASK / 'index-v1/index.json').read_bytes()
pointer = json.loads(pointer_path.read_bytes())
pointer['completed_administration'] = [{
    'name': 'index-v1', 'command': index_start['command'],
    'elapsed_seconds': index_finish['elapsed_seconds'], 'receipts': index_refs,
    'immutable_index': ref(TASK / 'index-v1/index.json'),
}]
pointer['paper_attribution'] = {
    'authors': ['Shangyu Chen', 'Wenya Wang', 'Sinno Jialin Pan'],
    'title': 'MetaQuant: Learning to Quantize by Learning to Penetrate Non-differentiable Quantization',
    'venue': 'NeurIPS 2019',
    'primary_url': 'https://papers.nips.cc/paper_files/paper/2019/file/f8e59f4b2fe7c5705bf878bbd494ccdf-Paper.pdf',
    'attribution_checked': 'primary paper first page; equations independently implemented',
    'project': 'csyhhu/MetaQuant@50694404fa010958aa24c96b4c870d5e9442e849',
    'source_copy_status': 'blocked_missing_grant',
}
physical = []
logical = []
for variant in native['variants']:
    for route, key in (('original', 'work'), ('replay', 'replay_work')):
        for work in variant[key]:
            prefix = '/'.join(('native-functional-v1', variant['variant'], variant['rotation'], route))
            logical.append({'id': f'{prefix}/step-{work["step"]}',
                            'observed_step_elapsed_seconds': work['elapsed_seconds'],
                            'timing_scope': 'nested in enclosing native launcher; do not add again'})
            for event in work['metaquant_work']['loss_pass_events']:
                physical.append({'physical_id': f'{prefix}/{event["execution_id"]}',
                                 'raw_event': event,
                                 'producer': pointer['native_report'],
                                 'per_pass_elapsed_seconds': None})
assert len(physical) == len({event['physical_id'] for event in physical}) == 40
assert sum(event['raw_event']['target_bytes'] for event in physical) == 160
assert len(logical) == 20
pointer['native_physical_training_events'] = physical
pointer['native_logical_step_timing'] = logical
pointer['administration_cost_scope'] = 'index receipt separate; active commit launcher omitted until its immutable finish exists; engineering/project total unknown'
data = (json.dumps(pointer, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
pointer_path.write_bytes(data)
with (ACTIVE / 'final-evidence.json').open('xb') as stream:
    stream.write(data)

transaction_bytes = sum(Path(name).stat().st_size for name in PATHS)
git_before = shutil.disk_usage(ROOT).free
data_before = shutil.disk_usage(STORE).free
store_before = sum(p.stat().st_size for p in STORE.rglob('*') if p.is_file())
forecast = int(os.environ['M10_FORECAST_BYTES'])
assert transaction_bytes <= 16 * 1024**2 and git_before - 16 * 1024**2 >= 512 * 1024**2
assert store_before + forecast <= 32 * 1024**3 and data_before - forecast >= 64 * 1024**3
run(['git', 'add', '--', *PATHS])
run(['git', 'diff', '--cached', '--check'])
staged = run(['git', 'diff', '--cached', '--name-only']).splitlines()
assert set(staged) == set(PATHS)
run(['git', 'commit', '-m', 'feat: add validated MetaQuant learning mechanism and LM adaptation'])
sha = run(['git', 'rev-parse', 'HEAD']).strip()
clean = run(['git', 'status', '--short'])
assert clean == ''
assert run(['git', 'branch', '--show-current']).strip() == 'feat/ralup-execution'
git_after = shutil.disk_usage(ROOT).free
data_after = shutil.disk_usage(STORE).free
assert git_after >= 512 * 1024**2 and data_after >= 64 * 1024**3
observation = {'commit': sha, 'subject': 'feat: add validated MetaQuant learning mechanism and LM adaptation',
               'branch': 'feat/ralup-execution', 'clean_git_status': clean,
               'transaction_bytes': transaction_bytes, 'git_free_before': git_before,
               'git_free_after': git_after, 'data_free_before': data_before,
               'data_free_after': data_after, 'retained_store_bytes_before': store_before,
               'remaining_forecast_bytes': forecast, 'commands': commands,
               'evidence_pointer': ref(pointer_path), 'native_receipts': native_refs,
               'CPU_receipts': cpu_refs, 'completed_index_receipts': index_refs,
               'commit_receipt_scope': 'startup exists before transaction; stdout/stderr/finish finalized by enclosing recorder after return'}
write_new(ACTIVE / 'commit.json', observation)
report_path = Path('.superpowers/sdd/2026-10-03-ralup/M10-report.md')
with report_path.open('a') as stream:
    stream.write(f'\nActual local commit `{sha}` — `{observation["subject"]}`. Branch `feat/ralup-execution`; exact post-commit `git status --short` output is empty. Explicit transaction of{len(PATHS)}files totals{transaction_bytes}B≤16MiB. Git free before{git_before}B / after{git_after}B≥512MiB. Total retained store before{store_before}B + remaining administration forecast{forecast}B≤32GiB; Data free before{data_before}B / after{data_after}B≥64GiB. Exact captured Git commands/stdout/stderr and source/test authentication are in `{ACTIVE}/commit.json`; recorder finalizes immutable startup/stdout/stderr/finish after the helper returns.\n')
    stream.write(f'\nCompleted `index-v1` receipt wall{index_finish["elapsed_seconds"]}s is separate administration; active commit wall is not invented. The tracked pointer additionally qualifies40 physical training events and20 logical-step nested timings from preserved native rows without launching work. No per-microbatch seconds are available. Paper full authors/title verified against primary first page; product source remains the exact native/full-CPU3 producer identity. A permanent complete report is `{ACTIVE}/final-report.md`.\n')
with (ACTIVE / 'final-report.md').open('xb') as stream:
    stream.write(report_path.read_bytes())
print(json.dumps({'commit': sha, 'clean': clean == '', 'transaction_bytes': transaction_bytes,
                  'report': str(report_path.resolve()), 'permanent_report': str(ACTIVE / 'final-report.md')}, sort_keys=True), flush=True)
