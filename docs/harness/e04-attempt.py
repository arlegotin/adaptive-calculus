"""Immutable E04 receipts; admit complete serialized writes before directory creation."""
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024**3
MIB = 1024**2


def encoded_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def audit_storage(*, store_bytes, data_free, git_free):
    if store_bytes > 32*GIB or data_free < 64*GIB or git_free < 512*MIB:
        raise ValueError('E04 complete-store cap or retained Data/Git reserve exceeded')


def admit_attempt(base, startup, *, archive_bytes, other_fixed_bytes,
                  child_output_allowance, finish_allowance, store_bytes,
                  data_free, git_free, forecast=None):
    """Budget exact fixed bytes plus declared output bounds; create only if safe.

    Serializing the inventory can dominate the attempt. Converge the few byte
    count fields to include their own encoded sizes, then write these exact bytes.
    The separate 16 MiB small-source/Git allowance is never a Data artifact cap.
    """
    values = [archive_bytes, other_fixed_bytes, child_output_allowance, finish_allowance]
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError('Nonnegative integer archive/output allowances required')
    if forecast is not None and (type(forecast) is not int or forecast < 0):
        raise ValueError('Nonnegative integer caller forecast required')
    storage = startup['storage']
    storage.update(allocated_bytes=store_bytes, git_free_before=git_free, data_free_before=data_free,
        complete_remaining_fix_forecast_bytes=forecast or 0,
        forecast_scope='exact source archive, encoded startup/inventory, runtime/diff and bounded child/finish outputs')
    budget = {'archive_bytes': archive_bytes, 'other_fixed_bytes': other_fixed_bytes,
        'child_output_allowance_bytes': child_output_allowance, 'finish_allowance_bytes': finish_allowance,
        'small_source_git_allowance_bytes': 16*MIB, 'startup_bytes': 0, 'minimum_forecast_bytes': 0}
    storage['write_budget'] = budget
    for _ in range(32):
        encoded = encoded_json(startup)
        minimum = archive_bytes + other_fixed_bytes + len(encoded) + child_output_allowance + finish_allowance
        chosen = minimum if forecast is None else forecast
        if (budget['startup_bytes'] == len(encoded) and budget['minimum_forecast_bytes'] == minimum
                and storage['complete_remaining_fix_forecast_bytes'] == chosen):
            break
        budget.update(startup_bytes=len(encoded), minimum_forecast_bytes=minimum)
        storage['complete_remaining_fix_forecast_bytes'] = chosen
    else:
        raise ValueError('Serialized startup size did not converge')
    if chosen < minimum:
        raise ValueError(f'Caller forecast {chosen} is below complete write requirement {minimum}')
    audit_storage(store_bytes=store_bytes+chosen, data_free=data_free-chosen, git_free=git_free-16*MIB)
    Path(base).mkdir(parents=True, exist_ok=False)
    return encoded


def write_bytes(path, content):
    with path.open('xb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())


def main():
    started = time.perf_counter()
    launcher_started_utc = datetime.now(timezone.utc).isoformat()
    root = Path.cwd()
    store = Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03')
    assert hashlib.sha256((store/'store.json').read_bytes()).hexdigest() == '16f61f5f5425f0ea95d4416bab16fa47936abaf592844e358013519cb4255af2'
    assert hashlib.sha256((root/'requirements.lock').read_bytes()).hexdigest() == '488142a35a4b1a6abe132f08b5895621ca30bb535b0326e82516446903fcb32b'
    name, module, *args = sys.argv[1:]
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('One new task-owned attempt directory name required')
    base = store/('e04-'+datetime.now(timezone.utc).strftime('%Y%m%d'))/name
    inventory = [(p, p.stat().st_size) for p in store.rglob('*') if p.is_file()]
    git_before, data_before = shutil.disk_usage(root).free, shutil.disk_usage(store).free
    paths = sorted(set(list(root.glob('src/ralup/**/*.py'))+list(root.glob('tests/**/*.py'))+
        list(root.glob('docs/harness/e04*.py'))+list(root.glob('docs/harness/e04*.json'))+
        list(root.glob('docs/harness/e04*.md'))+list(root.glob('configs/e04*.json'))+
        [root/'requirements.lock', root/'artifacts/environment/environment.json']))
    contents = {str(path.relative_to(root)): path.read_bytes() for path in paths}
    sources = {name: {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
               for name, content in contents.items()}
    observed = []
    for command in (['sw_vers'], ['uname', '-m']):
        result = subprocess.run(command, capture_output=True, text=True)
        observed.append({'command': command, 'exit_code': result.returncode,
                         'stdout': result.stdout, 'stderr': result.stderr})
    runtime = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
        'executable': sys.executable, 'machine': platform.machine(), 'system': platform.system(),
        'commands': observed, 'torch_distribution_version': importlib.metadata.version('torch'),
        'F02_role': 'archived historical environment bytes, not a current native admission',
        'scientific_native_admission': 'see explicit child purpose and separate current-runtime gate'}
    runtime_bytes = encoded_json(runtime)
    args = [a.replace('@ATTEMPT@', str(base)) for a in args]
    command = [str(root/'.venv/bin/python')]+([module] if module.endswith('.py') else ['-m', module])+args
    env = {**os.environ, 'TMPDIR': str(base/'tmp'), 'PYTHONPATH': str(root/'src'),
           'PYTHONDONTWRITEBYTECODE': '1', 'E04_ATTEMPT_DIR': str(base)}
    git_head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
    git_status = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True, check=True)
    git_diff = subprocess.run(['git', 'diff', '--binary'], capture_output=True, check=True)
    inputs = json.loads(os.environ.get('E04_INPUT_REFS', '[]'))
    for ref in inputs:
        assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256']
    startup = {'utc': datetime.now(timezone.utc).isoformat(), 'launcher_started_utc': launcher_started_utc,
        'local_time': datetime.now().astimezone().isoformat(), 'research_cutoff': '2026-10-03', 'inputs': inputs, 'command': command,
        'fallback': env.get('PYTORCH_ENABLE_MPS_FALLBACK'), 'source': sources,
        'git': {'head': git_head.stdout.strip(), 'status': git_status.stdout,
                'diff_sha256': hashlib.sha256(git_diff.stdout).hexdigest(),
                'git_command_stderr': [git_head.stderr, git_status.stderr, git_diff.stderr.decode()]},
        'runtime_observation': runtime, 'storage': {'inventory': [
            {'path': str(p.relative_to(store)), 'bytes': size} for p, size in inventory]},
        'later_e07_reserved_bytes': 2*GIB+882*MIB, 'cost_scope': 'complete launcher plus child process; enclosing/editing/tool costs unknown'}
    child_allowance = int(os.environ.get('E04_CHILD_OUTPUT_ALLOWANCE_BYTES', str(16*MIB)))
    finish_allowance = 8*MIB
    encoded = admit_attempt(base, startup, archive_bytes=sum(map(len, contents.values())),
        other_fixed_bytes=len(runtime_bytes)+len(git_diff.stdout), child_output_allowance=child_allowance,
        finish_allowance=finish_allowance, store_bytes=sum(size for _, size in inventory),
        data_free=data_before-(2*GIB+882*MIB), git_free=git_before,
        forecast=int(os.environ['E04_FORECAST_BYTES']) if 'E04_FORECAST_BYTES' in os.environ else None)
    # No permanent attempt output, archive or child imports precede admission.
    for relative, content in contents.items():
        target = base/'producer'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write_bytes(target, content)
    write_bytes(base/'runtime-observation.json', runtime_bytes)
    write_bytes(base/'producer-git-diff.patch', git_diff.stdout)
    write_bytes(base/'startup.json', encoded)
    (base/'tmp').mkdir()
    code = 1
    with (base/'stdout.txt').open('xb') as stdout, (base/'stderr.txt').open('xb') as stderr:
        try:
            code = subprocess.run(command, env=env, stdout=stdout, stderr=stderr).returncode
        except BaseException:
            import traceback
            stderr.write(traceback.format_exc().encode())
        finally:
            stdout.flush(); stderr.flush(); os.fsync(stdout.fileno()); os.fsync(stderr.fileno())
    usage, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    objects = [{'path': str(p.relative_to(base)), 'bytes': p.stat().st_size,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in sorted(base.rglob('*')) if p.is_file()]
    before_finish = sum(p.stat().st_size for p in store.rglob('*') if p.is_file())
    after_git, after_data = shutil.disk_usage(root).free, shutil.disk_usage(store).free
    fixed = sum(map(len, contents.values()))+len(runtime_bytes)+len(git_diff.stdout)+len(encoded)
    child_bytes = sum(row['bytes'] for row in objects)-fixed
    if child_bytes > child_allowance:
        code = 1  # Preserve child results and report the actual allowance violation.
    finish = {'exit_code': code, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.perf_counter()-started, 'launcher_cpu_user_seconds': usage.ru_utime,
        'launcher_cpu_system_seconds': usage.ru_stime, 'child_cpu_user_seconds': child.ru_utime,
        'child_cpu_system_seconds': child.ru_stime, 'child_maxrss': child.ru_maxrss,
        'git_free_after': after_git, 'data_free_after': after_data,
        'store_bytes_before_finish': before_finish, 'child_output_bytes': child_bytes,
        'child_output_within_allowance': child_bytes <= child_allowance, 'completed_objects': objects}
    finish_bytes = encoded_json(finish)
    if len(finish_bytes) > finish_allowance:
        raise ValueError('Finish receipt exceeds its separately reserved write allowance')
    # Include the about-to-be-written receipt in the pre-write audit as well.
    audit_storage(store_bytes=before_finish+len(finish_bytes), data_free=after_data-len(finish_bytes), git_free=after_git)
    write_bytes(base/'finish.json', finish_bytes)
    audit_storage(store_bytes=sum(p.stat().st_size for p in store.rglob('*') if p.is_file()),
                  data_free=shutil.disk_usage(store).free, git_free=shutil.disk_usage(root).free)
    print(json.dumps({'attempt': name, 'exit_code': code, 'path': str(base)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
