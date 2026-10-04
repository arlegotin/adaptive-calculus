"""Measured M05 source/config/pointer Git transaction; native child capture."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path.cwd();start=time.perf_counter();forecast=16*1024**2
paths=['src/ralup/baselines/qar.py','src/ralup/baselines/rounding.py','src/ralup/baselines/registry.py','src/ralup/train.py','src/ralup/state.py','src/ralup/__main__.py',
    'tests/unit/test_endpoint_routing.py','tests/integration/test_stochastic_context.py','docs/superpowers/specs/2026-10-03-ralup-design.md',
    'docs/harness/m05-routing-rounding.md','docs/harness/m05-evidence.json','docs/harness/m05-attempt.py','docs/harness/m05-mutants.py','docs/harness/m05-evidence.py','docs/harness/m05-publish.py','docs/harness/m05-commit.py']
def git(*args):
    result=subprocess.run(['git',*args],capture_output=True,text=True)
    if result.stdout:print(result.stdout,end='')
    if result.stderr:print(result.stderr,end='',file=sys.stderr)
    result.check_returncode();return result.stdout.strip()
before=shutil.disk_usage(root).free;size=sum((root/p).stat().st_size for p in paths)
assert size<=forecast and before-forecast>=512*1024**2
assert git('branch','--show-current')=='feat/ralup-execution'
git('merge-base','--is-ancestor','215c1d6','HEAD');git('diff','--check');git('add','--',*paths)
git('commit','-m','feat: add quantized endpoint routing and rounding controls')
commit=git('rev-parse','HEAD');after=shutil.disk_usage(root).free
assert after>=512*1024**2 and before-after<=forecast
value={'commit':commit,'paths':paths,'source_config_pointer_bytes':size,'forecast_bytes':forecast,'git_free_before':before,'git_free_after':after,'git_free_delta_bytes':before-after,'transaction_seconds':time.perf_counter()-start,'scope':'new source/config/pointer/Git metadata only; all bulk in new task-owned Data paths; no old artifact movement/deletion/overwrite'}
with (Path(os.environ['M05_ATTEMPT_DIR'])/'transaction.json').open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(value,sort_keys=True))
