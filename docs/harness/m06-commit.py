"""Measured small M06 source/config/pointer Git transaction; native capture."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path.cwd();start=time.perf_counter();forecast=16*1024**2
paths=['src/ralup/baselines/learned_scale.py','src/ralup/baselines/noise.py','src/ralup/baselines/registry.py',
    'src/ralup/evaluate.py','src/ralup/train.py','src/ralup/state.py','src/ralup/__main__.py',
    'tests/unit/test_learned_scale.py','tests/unit/test_noise_controls.py',
    'docs/harness/m06-scale-noise.md','docs/harness/m06-evidence.json','docs/harness/m06-attempt.py','docs/harness/m06-exp-diagnostic.py',
    'docs/harness/m06-replay.py','docs/harness/m06-mutants.py','docs/harness/m06-verify.py',
    'docs/harness/m06-evidence.py','docs/harness/m06-publish.py','docs/harness/m06-commit.py']
def git(*args):
    p=subprocess.run(['git',*args],capture_output=True,text=True)
    if p.stdout:print(p.stdout,end='')
    if p.stderr:print(p.stderr,end='',file=sys.stderr)
    p.check_returncode();return p.stdout.strip()
before=shutil.disk_usage(root).free;size=sum((root/p).stat().st_size for p in paths)
assert size<=forecast and before-forecast>=512*1024**2
assert git('branch','--show-current')=='feat/ralup-execution'
git('merge-base','--is-ancestor','b0b471e','HEAD');git('diff','--check');git('add','--',*paths)
git('commit','-m','feat: add learned-scale damping and masking controls')
commit=git('rev-parse','HEAD');after=shutil.disk_usage(root).free
assert after>=512*1024**2 and before-after<=forecast
value={'commit':commit,'paths':paths,'source_config_pointer_bytes':size,'forecast_bytes':forecast,'git_free_before':before,
    'git_free_after':after,'git_free_delta_bytes':before-after,'transaction_seconds':time.perf_counter()-start,
    'scope':'new source/config/pointer/Git metadata only; all bulk in new task-owned Data paths; no old artifact mutation'}
with (Path(os.environ['M06_ATTEMPT_DIR'])/'transaction.json').open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(value,sort_keys=True))
