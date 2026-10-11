"""Commit E03 I1 result-state correction and immutable full report; run inside e03-attempt."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path.cwd()
base = Path(os.environ['E03_ATTEMPT_DIR'])
report = root/'.superpowers/sdd/2026-10-03-ralup/E03-report.md'
files = ['src/ralup/evaluate.py', 'src/ralup/records.py', 'tests/integration/test_protected_access.py',
         'docs/harness/e03-contract.md', 'docs/harness/e03-fix1-evidence.json', 'docs/harness/e03-fix1-finish.py']
assert sum((root/path).stat().st_size for path in files)+report.stat().st_size < 16*1024**2
assert shutil.disk_usage(root).free-16*1024**2 >= 512*1024**2

def write(path, value):
    with path.open('xb') as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())

write(base/'report-before-commit.md', report.read_bytes())
subprocess.run(['git', 'diff', '--check'], check=True)
subprocess.run(['git', 'add', '--', *files], check=True)
subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
commit = subprocess.run(['git', 'commit', '-m', 'fix: require successful payload before protected results'],
                        capture_output=True, text=True)
write(base/'commit-command.json', json.dumps({'exit_code': commit.returncode,
    'stdout': commit.stdout, 'stderr': commit.stderr}, sort_keys=True, indent=2).encode()+b'\n')
if commit.returncode:
    raise SystemExit(commit.returncode)
sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
status = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True, check=True).stdout
with report.open('a') as stream:
    stream.write(f'\n`{sha}` — `fix: require successful payload before protected results`.\n')
    stream.write(f'Post-commit tracked/untracked status: {status!r}. Fresh scoped I1 coordinator review required.\n')
write(base/'E03-report.md', report.read_bytes())
write(base/'commit.json', json.dumps({'sha': sha, 'status': status,
    'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
    'git_free_after': shutil.disk_usage(root).free}, sort_keys=True, indent=2).encode()+b'\n')
print(json.dumps({'sha': sha, 'status': status, 'report': str(base/'E03-report.md')}))
