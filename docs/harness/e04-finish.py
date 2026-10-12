"""Small-source E04 commit plus immutable administration; run under e04-attempt."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root=Path.cwd();base=Path(os.environ['E04_ATTEMPT_DIR'])
report=root/'.superpowers/sdd/2026-10-03-ralup/E04-report.md'
files=['README.md','src/ralup/__main__.py','src/ralup/runner.py','src/ralup/analysis/confirm.py',
    'tests/conftest.py','tests/integration/test_confirmation_runner.py','tests/unit/test_paired_inference.py',
    'tests/unit/test_e04_attempt_storage.py','docs/harness/e04-attempt.py','docs/harness/e04-finish.py',
    'docs/harness/e04-contract.md','docs/harness/e04-evidence.json']
assert sum((root/path).stat().st_size for path in files)+report.stat().st_size<=16*1024**2
assert shutil.disk_usage(root).free-16*1024**2>=512*1024**2
assert subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()=='85a555b544278e98af90757c149acef798157584'
assert subprocess.run(['git','branch','--show-current'],capture_output=True,text=True,check=True).stdout.strip()=='feat/ralup-execution'
subprocess.run(['git','diff','--cached','--quiet'],check=True)

def write(path,value):
    with path.open('xb') as stream:
        stream.write(value);stream.flush();os.fsync(stream.fileno())

def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

# The final source/test snapshot already verified the changed interfaces.
index=json.loads((root/'docs/harness/e04-evidence.json').read_bytes())
final=next(row for row in index['attempts'] if row['attempt']=='focused-final-v1')
for name,row in final['source'].items():
    if name.startswith(('src/','tests/')):
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==row['sha256'],name
assert not (root/'artifacts/confirm_manifest.json').exists()
write(base/'report-before-commit.md',report.read_bytes())
subprocess.run(['git','diff','--check'],check=True)
subprocess.run(['git','add','--',*files],check=True)
subprocess.run(['git','diff','--cached','--check'],check=True)
commit=subprocess.run(['git','commit','-m','feat: add locked paired confirmation and honest failure inference'],capture_output=True,text=True)
write(base/'commit-command.json',encoded({'exit_code':commit.returncode,'stdout':commit.stdout,'stderr':commit.stderr}))
if commit.returncode:raise SystemExit(commit.returncode)
sha=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
status=subprocess.run(['git','status','--short'],capture_output=True,text=True,check=True).stdout
with report.open('a') as stream:
    stream.write(f'\nCommit `{sha}` — `feat: add locked paired confirmation and honest failure inference`.\n')
    stream.write(f'Post-commit tracked/untracked status: {status!r}. Coordinator fresh FULL BASE..HEAD task review remains pending.\n')
write(base/'E04-report.md',report.read_bytes())
write(base/'commit.json',encoded({'sha':sha,'status':status,'base':index['base'],
    'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'git_free_after':shutil.disk_usage(root).free,
    'small_source_bytes':sum((root/path).stat().st_size for path in files),'source_tests_match_final_verified':True}))
if status:raise RuntimeError('Post-commit worktree is not clean')
print(json.dumps({'sha':sha,'status':status,'report':str(base/'E04-report.md')}))
