"""Retain exact earlier CPU RED/GREEN stdout; their startup archives are unknown.

Then run a fresh current-source focused CPU command under the outer recorder.
No historical native suite is run. Every path below is an explicit task-owned
support output; original bytes remain intact in /private/tmp.
"""
import hashlib,json,os,subprocess,sys
from pathlib import Path
base=Path(os.environ['M09_ATTEMPT_DIR']);root=Path.cwd();retained=[]
for name in ('m09-red-controllers.txt','m09-green-controllers.txt','m09-red-adapter.txt','m09-red-adapter-v2.txt','m09-green-adapter-v1.txt','m09-green-adapter-v2.txt','m09-green-expanded-v1.txt','m09-red-guards.txt','m09-green-scope-v1.txt','m09-red-cli.txt','m09-green-checks-v1.txt','m09-green-projection.txt','m09-green-prefreeze.txt','m09-full-cpu.txt'):
 p=Path('/private/tmp')/name;data=p.read_bytes();target=base/'earlier-support'/name;target.parent.mkdir(exist_ok=True);target.write_bytes(data)
 retained.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'scope':'exact previously completed CPU/sandbox stdout; pre-import startup/source archive not captured, unknown enclosing/editing cost'})
command=[sys.executable,'-m','pytest','tests/unit/test_controllers.py','tests/integration/test_policy_identity.py','tests/unit/test_policy_features.py','tests/integration/test_history_lifecycle.py','-q']
result=subprocess.run(command,capture_output=True,text=True)
print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
with (base/'support.json').open('x') as f:json.dump({'retained':retained,'current_command':command,'current_exit_code':result.returncode,'scope':'CPU-only focused support; no prior accepted native suites'},f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
raise SystemExit(result.returncode)
