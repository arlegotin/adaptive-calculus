"""Scoped I1/I2 E05 fix commit and immutable report; run under e05-attempt."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root=Path.cwd();base=Path(os.environ['E05_ATTEMPT_DIR'])
report=root/'.superpowers/sdd/2026-10-03-ralup/E05-report.md'
files=['src/ralup/analysis/attribution.py','tests/unit/test_ablation_contracts.py',
       'docs/harness/e05-contract.md','docs/harness/e05-fix1-evidence.json',
       'docs/harness/e05-fix1-finish.py']
fixbase='8b1add1e3e9d772b4a9a44cdb5417bf5c599fe50'
assert sum((root/p).stat().st_size for p in files)+report.stat().st_size<=16*1024**2
assert shutil.disk_usage(root).free-16*1024**2>=512*1024**2
assert subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()==fixbase
assert subprocess.run(['git','branch','--show-current'],capture_output=True,text=True,check=True).stdout.strip()=='feat/ralup-execution'
subprocess.run(['git','diff','--cached','--quiet'],check=True)
index=json.loads((root/'docs/harness/e05-fix1-evidence.json').read_bytes())
final=next(a for a in index['attempts'] if a['attempt']=='fix1-green-v1')
assert final['exit_code']==0 and final['child_output_within_allowance'] is True
startup_ref=final['receipts']['startup.json'];content=Path(startup_ref['path']).read_bytes()
assert hashlib.sha256(content).hexdigest()==startup_ref['sha256'];startup=json.loads(content)
for name,item in startup['source'].items():
    if name.startswith(('src/','tests/')):assert hashlib.sha256((root/name).read_bytes()).hexdigest()==item['sha256'],name
assert not (root/'artifacts/confirm_manifest.json').exists()


def write(path,value):
    with path.open('xb') as stream:stream.write(value);stream.flush();os.fsync(stream.fileno())


def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

write(base/'report-before-commit.md',report.read_bytes())
subprocess.run(['git','diff','--check'],check=True)
subprocess.run(['git','add','--',*files],check=True)
subprocess.run(['git','diff','--cached','--check'],check=True)
commit=subprocess.run(['git','commit','-m','fix: authenticate attribution passes and retain article variation'],capture_output=True,text=True)
write(base/'commit-command.json',encoded({'exit_code':commit.returncode,'stdout':commit.stdout,'stderr':commit.stderr}))
if commit.returncode:raise SystemExit(commit.returncode)
sha=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
status=subprocess.run(['git','status','--short'],capture_output=True,text=True,check=True).stdout
with report.open('a') as stream:
    stream.write(f'\nFix1 commit `{sha}` — `fix: authenticate attribution passes and retain article variation`.\n')
    stream.write(f'Post-commit tracked/untracked status: {status!r}. Coordinator scoped I1/I2 {fixbase}..HEAD review pending; root owns final whole-branch review after E07.\n')
write(base/'E05-report.md',report.read_bytes())
write(base/'commit.json',encoded({'sha':sha,'fix_base':fixbase,'status':status,
    'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),
    'source_tests_match_focused_green':True,
    'small_source_bytes':sum((root/p).stat().st_size for p in files),
    'git_free_after':shutil.disk_usage(root).free,'scientific_mechanism_established':False,
    'unknown_enclosing_engineering_cost':None}))
if status:raise RuntimeError('Post-commit worktree is not clean')
print(json.dumps({'sha':sha,'status':status,'report':str(base/'E05-report.md')}))
