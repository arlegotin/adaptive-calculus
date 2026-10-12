"""Scoped E04 fix1 commit and immutable report; execute under e04-attempt."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root=Path.cwd();base=Path(os.environ['E04_ATTEMPT_DIR'])
report=root/'.superpowers/sdd/2026-10-03-ralup/E04-report.md'
files=['src/ralup/runner.py','src/ralup/analysis/confirm.py',
    'tests/integration/test_confirmation_persistence.py','docs/harness/e04-contract.md',
    'docs/harness/e04-fix1-evidence.json','docs/harness/e04-fix1-finish.py']
fixbase='a781c3179bcc08742b902a3397c32c5b04351070'
assert sum((root/p).stat().st_size for p in files)+report.stat().st_size<=16*1024**2
assert shutil.disk_usage(root).free-16*1024**2>=512*1024**2
assert subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()==fixbase
assert subprocess.run(['git','branch','--show-current'],capture_output=True,text=True,check=True).stdout.strip()=='feat/ralup-execution'
subprocess.run(['git','diff','--cached','--quiet'],check=True)
index=json.loads((root/'docs/harness/e04-fix1-evidence.json').read_bytes())
final=next(r for r in index['attempts'] if r['attempt']=='fix1-focused-final-v1')
startup_ref=final['receipts']['startup.json'];raw=Path(startup_ref['path']).read_bytes()
assert hashlib.sha256(raw).hexdigest()==startup_ref['sha256']
startup=json.loads(raw)
for name,row in startup['source'].items():
    if name.startswith(('src/','tests/')):assert hashlib.sha256((root/name).read_bytes()).hexdigest()==row['sha256'],name
assert not (root/'artifacts/confirm_manifest.json').exists()

def write(path,value):
    with path.open('xb') as stream:
        stream.write(value);stream.flush();os.fsync(stream.fileno())

def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

write(base/'report-before-commit.md',report.read_bytes())
subprocess.run(['git','diff','--check'],check=True)
subprocess.run(['git','add','--',*files],check=True)
subprocess.run(['git','diff','--cached','--check'],check=True)
commit=subprocess.run(['git','commit','-m','fix: retain warm execution and publication cost lower bounds'],capture_output=True,text=True)
write(base/'commit-command.json',encoded({'exit_code':commit.returncode,'stdout':commit.stdout,'stderr':commit.stderr}))
if commit.returncode:raise SystemExit(commit.returncode)
sha=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
status=subprocess.run(['git','status','--short'],capture_output=True,text=True,check=True).stdout
with report.open('a') as stream:
    stream.write(f'\nFix1 commit `{sha}` — `fix: retain warm execution and publication cost lower bounds`.\n')
    stream.write(f'Post-commit tracked/untracked status: {status!r}. Fresh SCOPED {fixbase}..{sha} coordinator review pending.\n')
write(base/'E04-report.md',report.read_bytes())
write(base/'commit.json',encoded({'sha':sha,'fix_base':fixbase,'status':status,
    'small_source_bytes':sum((root/p).stat().st_size for p in files),
    'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'source_tests_match_final_verified':True,
    'git_free_after':shutil.disk_usage(root).free,'complete_elapsed_available':False}))
if status:raise RuntimeError('Post-commit worktree is not clean')
print(json.dumps({'sha':sha,'status':status,'report':str(base/'E04-report.md')}))
