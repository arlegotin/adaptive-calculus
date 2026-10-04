"""Retain every M01 executable attempt, including failures, before subsequent work."""
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import shutil
import sys
import time

start = time.perf_counter()
root = Path.cwd()
name, module, *args = sys.argv[1:]
base = root / 'artifacts/checks/m01-v1' / name
base.parent.mkdir(parents=True, exist_ok=True)
paths = sorted(list(root.glob('src/ralup/**/*.py')) + list(root.glob('tests/**/*.py')) +
               list(root.glob('docs/harness/m01*.py')) + [root / 'requirements.lock', root / 'artifacts/environment/environment.json', Path(__file__).resolve()])
identity = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def publish(suffix, value):
    with Path(str(base) + suffix).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
publish('-start.json', {'utc':datetime.now(timezone.utc).isoformat(), 'pid':os.getpid(),
        'module':module, 'arguments':args, 'source_sha256':identity,
        'fallback':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        'free_git_bytes':shutil.disk_usage(root).free, 'purpose':'M01 finite development engineering',
        'synthetic_fixture_work':'not corpus/scientific bytes; actual pilot counts in its per-step journal'})
code = 0
with Path(str(base) + '.txt').open('x', buffering=1) as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
    try:
        sys.argv = [module, *args]
        if module.endswith('.py'): runpy.run_path(module, run_name='__main__')
        else: runpy.run_module(module, run_name='__main__', alter_sys=True)
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
    except BaseException:
        import traceback
        traceback.print_exc(); code = 1
    finally:
        stream.flush(); os.fsync(stream.fileno())
usage = resource.getrusage(resource.RUSAGE_SELF)
children = resource.getrusage(resource.RUSAGE_CHILDREN)
publish('-finish.json', {'exit_code':code,'full_process_seconds':time.perf_counter()-start,
        'cpu_user_seconds':usage.ru_utime,'cpu_system_seconds':usage.ru_stime,
        'child_cpu_user_seconds':children.ru_utime,'child_cpu_system_seconds':children.ru_stime,
        'output_sha256':hashlib.sha256(Path(str(base)+'.txt').read_bytes()).hexdigest(),
        'new_work_receipt':'complete launched module scope; child wall is inside process wall, not added',
        'free_git_bytes':shutil.disk_usage(root).free})
print(json.dumps({'attempt':name,'exit_code':code,'output':str(base)+'.txt'}))
raise SystemExit(code)
