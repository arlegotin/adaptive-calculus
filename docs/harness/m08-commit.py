"""Authorized small local M08 commit with exact command and cost receipts."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

root = Path.cwd()
started = time.perf_counter()
index = json.loads((root / 'docs/harness/m08-feature-evidence.json').read_bytes())
for name, digest in index['source_identity'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'feat/ralup-execution'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == index['task_base']
paths = sorted(p for p in subprocess.check_output(
    ['git', 'ls-files', '--modified', '--others', '--exclude-standard', '-z']).decode().split('\0') if p)
allowed = {'src/ralup/__main__.py', 'src/ralup/train.py', 'src/ralup/state.py',
           'src/ralup/optim/base.py', 'src/ralup/optim/matrix_adapter.py',
           'src/ralup/baselines/trs.py', 'src/ralup/baselines/qar.py',
           'src/ralup/baselines/noise.py', 'src/ralup/baselines/learned_scale.py',
           'tests/unit/test_policy_features.py', 'tests/unit/test_checkpoint_integrity.py',
           'tests/integration/test_history_lifecycle.py', 'tests/integration/test_ewgs_hvp.py'}
assert paths and all(p in allowed or p.startswith('docs/harness/m08') or p in
                    {'src/ralup/policy/features.py', 'src/ralup/policy/history.py',
                     'src/ralup/policy/normalizer.py', 'src/ralup/policy/checks.py'} for p in paths)
selected_bytes = sum((root / p).stat().st_size for p in paths)
before = shutil.disk_usage(root).free
assert selected_bytes <= 16 * 1024**2 and before - 16 * 1024**2 >= 512 * 1024**2
commands = [['git', 'diff', '--check'], ['git', 'add', '--', *paths],
            ['git', 'diff', '--cached', '--check'],
            ['git', 'commit', '-m', 'feat: add causal cell history features and frozen normalization'],
            ['git', 'rev-parse', 'HEAD'], ['git', 'status', '--short']]
results = []
for command in commands:
    result = subprocess.run(command, capture_output=True, text=True)
    results.append({'command': command, 'exit_code': result.returncode,
                    'stdout': result.stdout, 'stderr': result.stderr})
    print(result.stdout, end='')
    print(result.stderr, end='')
    if result.returncode:
        break
after = shutil.disk_usage(root).free
assert after >= 512 * 1024**2
record = {'selected_paths': paths, 'selected_bytes': selected_bytes,
          'git_free_before': before, 'git_free_after': after,
          'complete_seconds': time.perf_counter() - started, 'commands': results,
          'task_status': index['status'], 'source_matches_terminal_native': True,
          'report': str(root / '.superpowers/sdd/2026-10-03-ralup/M08-report.md')}
with (Path(os.environ['M08_ATTEMPT_DIR']) / 'commit.json').open('x') as stream:
    json.dump(record, stream, sort_keys=True, indent=2)
    stream.write('\n')
raise SystemExit(results[-1]['exit_code'])
