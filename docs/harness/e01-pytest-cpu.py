"""Full CPU regression, explicitly suppress historical native test waves.

Only this test interpreter reports MPS unavailable. This is CPU coverage, never
device-support evidence; E01 runs no new native scientific comparison.
"""
import json
import functools
import os
from pathlib import Path
import torch
import pytest
print(json.dumps({'test_scope':'CPU-only full regression','actual_mps_available_before_scope_override':torch.backends.mps.is_available(),
    'native_scope':'not admitted by this test run; not part of this E01 gated disposition'}),flush=True)
torch.backends.mps.is_available=functools.lru_cache(maxsize=None)(lambda:False)
bootstrap=Path(os.environ['E01_ATTEMPT_DIR'])/'cpu-bootstrap'
bootstrap.mkdir()
(bootstrap/'sitecustomize.py').write_bytes(Path('docs/harness/e01-cpu-sitecustomize.py').read_bytes())
os.environ['E01_CPU_REGRESSION_SCOPE']='1'
os.environ['PYTHONPATH']=str(bootstrap)+os.pathsep+os.environ.get('PYTHONPATH','')
raise SystemExit(pytest.main(['-q']))
