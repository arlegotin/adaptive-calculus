"""Actually recreate the immutable core lock in a separate environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path.cwd(); output=Path(os.environ['E07_OPERATION']); venv=output/'core'
lock=root/'requirements.lock'
assert hashlib.sha256(lock.read_bytes()).hexdigest()=='488142a35a4b1a6abe132f08b5895621ca30bb535b0326e82516446903fcb32b'
subprocess.run([sys.executable,'-m','venv','--without-pip',str(venv)],check=True)
subprocess.run([sys.executable,'-m','pip','--python',str(venv/'bin/python'),'install',
    '--require-hashes','--no-cache-dir','--no-compile','-r',str(lock),'--report',str(output/'install-report.json')],check=True)
script='import importlib.metadata,json,sys; print(json.dumps({"python":sys.version,"packages":dict(sorted((d.metadata["Name"].lower(),d.version) for d in importlib.metadata.distributions()))},sort_keys=True))'
original=json.loads(subprocess.check_output([sys.executable,'-B','-c',script]))
fresh=json.loads(subprocess.check_output([str(venv/'bin/python'),'-B','-c',script]))
assert original==fresh,(original,fresh)
(output/'environment-comparison.json').write_text(json.dumps({'original':original,'recreated':fresh,'exact_match':True},indent=2)+'\n')
print(json.dumps({'separate_executable':str(venv/'bin/python'),'exact_distribution_and_python_match':True}))
