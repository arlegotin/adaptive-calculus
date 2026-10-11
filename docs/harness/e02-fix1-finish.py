"""Commit the narrow E02 recorder correction inside its guarded attempt."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path.cwd()
base = Path(os.environ['E02_ATTEMPT_DIR'])
report = root/'.superpowers/sdd/2026-10-03-ralup/E02-report.md'
files = ['docs/harness/e02-attempt.py', 'tests/unit/test_e02_attempt_storage.py',
         'docs/harness/e02-contract.md', 'docs/harness/e02-fix1-evidence.json',
         'docs/harness/e02-fix1-finish.py']
assert sum((root/path).stat().st_size for path in files)+report.stat().st_size < 16*1024**2
assert shutil.disk_usage(root).free-16*1024**2 >= 512*1024**2

def write(path, content):
    with path.open('xb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())

write(base/'report-before-commit.md', report.read_bytes())
subprocess.run(['git', 'diff', '--check'], check=True)
subprocess.run(['git', 'add', '--', *files], check=True)
subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
result = subprocess.run(['git', 'commit', '-m', 'fix: budget complete serialized E02 attempt writes'], capture_output=True, text=True)
write(base/'commit-command.json', json.dumps({'exit_code': result.returncode, 'stdout': result.stdout,
    'stderr': result.stderr}, sort_keys=True, indent=2).encode()+b'\n')
if result.returncode:
    raise SystemExit(result.returncode)
sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
status = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True, check=True).stdout
with report.open('a') as stream:
    stream.write(f'\n`{sha}` — `fix: budget complete serialized E02 attempt writes`.\n')
    stream.write(f'Post-commit tracked/untracked status: {status!r}; fresh scoped I1 review required.\n')
write(base/'E02-report.md', report.read_bytes())
write(base/'commit.json', json.dumps({'sha': sha, 'status': status,
    'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
    'git_free_after': shutil.disk_usage(root).free}, sort_keys=True, indent=2).encode()+b'\n')
print(json.dumps({'sha': sha, 'status': status, 'report': str(base/'E02-report.md')}))
