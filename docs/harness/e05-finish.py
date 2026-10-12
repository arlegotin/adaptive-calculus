"""E05 local commit and immutable administration; run under e05-attempt."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root=Path.cwd();base=Path(os.environ['E05_ATTEMPT_DIR'])
report=root/'.superpowers/sdd/2026-10-03-ralup/E05-report.md'
files=['src/ralup/analysis/attribution.py','src/ralup/__main__.py','src/ralup/search.py',
       'src/ralup/baselines/registry.py','src/ralup/policy/actions.py',
       'tests/unit/test_ablation_contracts.py','configs/e05-attribution-prerequisites.json',
       'docs/harness/e05-attempt.py','docs/harness/e05-finish.py',
       'docs/harness/e05-contract.md','docs/harness/e05-evidence.json']
files += [str(p.relative_to(root)) for p in sorted((root/'configs/ablations').glob('*.json'))]
base_sha='b6aba9fa1d7eb239c7c2a1c08fe3a0f52caafc44'
assert sum((root/p).stat().st_size for p in files)+report.stat().st_size<=16*1024**2
assert shutil.disk_usage(root).free-16*1024**2>=512*1024**2
assert subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()==base_sha
assert subprocess.run(['git','branch','--show-current'],capture_output=True,text=True,check=True).stdout.strip()=='feat/ralup-execution'
subprocess.run(['git','diff','--cached','--quiet'],check=True)
index=json.loads((root/'docs/harness/e05-evidence.json').read_bytes())
final=next(a for a in index['attempts'] if a['attempt']=='focused-final-cpu-v3')
assert final['exit_code']==0 and final['child_output_within_allowance'] is True
startup_ref=final['receipts']['startup.json'];content=Path(startup_ref['path']).read_bytes()
assert hashlib.sha256(content).hexdigest()==startup_ref['sha256'];startup=json.loads(content)
for name,item in startup['source'].items():
    if name.startswith(('src/','tests/','configs/')):assert hashlib.sha256((root/name).read_bytes()).hexdigest()==item['sha256'],name
cli=next(a for a in index['attempts'] if a['attempt']=='attribution-cli-v3')
assert cli['exit_code']==2 and cli['child_output_within_allowance'] is True
receipt=cli['receipts']['startup.json'];content=Path(receipt['path']).read_bytes()
assert hashlib.sha256(content).hexdigest()==receipt['sha256'];science_startup=json.loads(content)
for name,item in science_startup['source'].items():
    if name.startswith(('src/','tests/','configs/')):assert hashlib.sha256((root/name).read_bytes()).hexdigest()==item['sha256'],name
assert not (root/'artifacts/confirm_manifest.json').exists()


def write(path,value):
    with path.open('xb') as stream:stream.write(value);stream.flush();os.fsync(stream.fileno())


def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

write(base/'report-before-commit.md',report.read_bytes())
subprocess.run(['git','diff','--check'],check=True)
subprocess.run(['git','add','--',*files],check=True)
subprocess.run(['git','diff','--cached','--check'],check=True)
commit=subprocess.run(['git','commit','-m','feat: add mechanism attribution and evidence-triggered extensions'],capture_output=True,text=True)
write(base/'commit-command.json',encoded({'exit_code':commit.returncode,'stdout':commit.stdout,'stderr':commit.stderr}))
if commit.returncode:raise SystemExit(commit.returncode)
sha=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
status=subprocess.run(['git','status','--short'],capture_output=True,text=True,check=True).stdout
with report.open('a') as stream:
    stream.write(f'\nCommit `{sha}` — `feat: add mechanism attribution and evidence-triggered extensions`.\n')
    stream.write(f'Post-commit tracked/untracked status: {status!r}. Coordinator fresh FULL {base_sha}..HEAD E05 review pending; root owns final whole-branch review after E07.\n')
write(base/'E05-report.md',report.read_bytes())
write(base/'commit.json',encoded({'sha':sha,'base':base_sha,'status':status,
    'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),
    'source_tests_configs_match_final_verification_and_cli':True,
    'small_source_bytes':sum((root/p).stat().st_size for p in files),
    'git_free_after':shutil.disk_usage(root).free,'scientific_mechanism_established':False,
    'unknown_enclosing_engineering_cost':None}))
if status:raise RuntimeError('Post-commit worktree is not clean')
print(json.dumps({'sha':sha,'status':status,'report':str(base/'E05-report.md')}))
