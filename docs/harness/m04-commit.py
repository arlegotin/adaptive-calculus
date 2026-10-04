"""Measured M04 source/config/pointer-only Git transaction; no bulk assets."""
import json,shutil,subprocess,time
from pathlib import Path
root=Path.cwd();start=time.perf_counter();forecast=16*1024**2
paths=['src/ralup/baselines/trs.py','src/ralup/baselines/registry.py','src/ralup/state.py','src/ralup/__main__.py',
       'tests/unit/test_trs.py','tests/integration/test_trs_accumulation.py',
       'docs/harness/m04-trs.md','docs/harness/m04-evidence.json','docs/harness/m04-attempt.py','docs/harness/m04-mutants.py',
       'docs/harness/m04-helper-exit.py','docs/harness/m04-evidence.py','docs/harness/m04-commit.py']
before=shutil.disk_usage(root).free;source_bytes=sum((root/p).stat().st_size for p in paths)
assert source_bytes<forecast and before-forecast>=512*1024**2
subprocess.run(['git','diff','--check'],check=True)
subprocess.run(['git','add','--',*paths],check=True)
subprocess.run(['git','commit','-m','feat: add TRS release fixture and logical-step adapter'],check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();after=shutil.disk_usage(root).free
assert after>=512*1024**2 and before-after<=forecast
value={'commit':commit,'paths':paths,'source_config_pointer_bytes':source_bytes,'forecast_bytes':forecast,'git_free_before':before,'git_free_after':after,'git_free_delta_bytes':before-after,'transaction_seconds':time.perf_counter()-start,'scope':'new source/config/pointer/Git-metadata only; all bulk retained in new Data task paths; no old artifacts removed/moved/overwritten'}
receipt=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m04-20261004/commit/transaction.json')
with receipt.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(value,sort_keys=True))
