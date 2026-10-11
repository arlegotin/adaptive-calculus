"""Guarded new FIX1 source/config/Git transaction; native streams captured."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path.cwd();started=time.perf_counter();forecast=16*1024**2
paths=['src/ralup/train.py','tests/unit/test_learned_scale.py','docs/harness/m06-scale-noise.md',
    'docs/harness/m06-fix1-attempt.py','docs/harness/m06-fix1-publish.py','docs/harness/m06-fix1-commit.py','docs/harness/m06-evidence-fix1.json']
def git(*args):
    result=subprocess.run(['git',*args],capture_output=True,text=True)
    if result.stdout:print(result.stdout,end='')
    if result.stderr:print(result.stderr,end='',file=sys.stderr)
    result.check_returncode();return result.stdout.strip()
before=shutil.disk_usage(root).free;source_bytes=sum((root/p).stat().st_size for p in paths)
assert source_bytes<=forecast and before-forecast>=512*1024**2
assert git('branch','--show-current')=='feat/ralup-execution'
assert git('rev-parse','HEAD')=='439eef1e6e6ecf88607e4407c0ecd879e9ff2dff'
git('diff','--check');git('add','--',*paths)
git('commit','-m','fix: preserve learned parameter identities on successful steps')
commit=git('rev-parse','HEAD');after=shutil.disk_usage(root).free
assert after>=512*1024**2 and before-after<=forecast
receipt={'commit':commit,'paths':paths,'source_config_pointer_bytes':source_bytes,'forecast_bytes':forecast,
    'git_free_before':before,'git_free_after':after,'git_free_delta_bytes':before-after,
    'transaction_seconds':time.perf_counter()-started,'scope':'new FIX1 source/config/Git metadata only; no historical evidence mutation'}
with (Path(os.environ['M06_FIX1_ATTEMPT_DIR'])/'transaction.json').open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(receipt,sort_keys=True))
