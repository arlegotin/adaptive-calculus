"""One small authorized source transaction after focused RED/GREEN evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from ralup.policy.checks import _write

base=Path(os.environ['M11_ATTEMPT_DIR']);started=time.perf_counter()
files=['src/ralup/meta/tasks.py','tests/integration/test_meta_rollouts.py']
assert sum(Path(p).stat().st_size for p in files)<=16*1024**2
free_before=shutil.disk_usage(Path.cwd()).free;assert free_before>=512*1024**2+16*1024**2
assert subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip()==''
commands=[]
for command in [['git','diff','--check'],['git','add','--',*files],['git','commit','-m','fix: retain completed policy actions in meta rollout evidence']]:
    result=subprocess.run(command,text=True,capture_output=True)
    commands.append({'command':command,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    assert result.returncode==0
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
free_after=shutil.disk_usage(Path.cwd()).free;assert free_after>=512*1024**2
_write(base/'report.json',{'status':'source_transaction_committed','commit':commit,'commands':commands,
    'source_input_bytes':sum(Path(p).stat().st_size for p in files),'git_free_before':free_before,'git_free_after':free_after,
    'elapsed_seconds':time.perf_counter()-started,'scope':'Only the two named source/test files; prior native evidence preserves original producer hashes.'})
print(json.dumps({'commit':commit,'commands':commands}),flush=True)
