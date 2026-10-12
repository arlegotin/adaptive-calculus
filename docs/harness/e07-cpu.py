"""Final-tree CPU suite, explicitly excluding native availability in child processes."""
import functools
import os
from pathlib import Path
import sys
import torch
import pytest

out=Path(os.environ['E07_OPERATION']);bootstrap=out/'cpu-bootstrap';bootstrap.mkdir()
# Existing authenticated CPU-scope mechanism also scopes fresh test subprocesses.
source=Path('docs/harness/e01-cpu-sitecustomize.py').read_bytes()
(bootstrap/'sitecustomize.py').write_bytes(source)
os.environ['E01_CPU_REGRESSION_SCOPE']='1'
sys.path.insert(0,str(Path.cwd()))
os.environ['PYTHONPATH']=str(bootstrap)+os.pathsep+str(Path.cwd())+os.pathsep+os.environ['PYTHONPATH']
print('Current-tree CPU integration. Native cases explicitly unavailable in this test process; no native pass inferred.',flush=True)
torch.backends.mps.is_available=functools.lru_cache(maxsize=None)(lambda:False)
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(out/'pytest'),
    '--deselect=tests/integration/test_export_forward.py::test_actual_mps_export_composition']))
