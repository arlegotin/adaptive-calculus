"""Separate plotting-only environment; core lock is never modified."""
import json
import os
from pathlib import Path
import subprocess
import sys

out=Path(os.environ['E07_OPERATION']);env=out/'analysis';python=env/'bin/python'
subprocess.run([sys.executable,'-m','venv','--without-pip',str(env)],check=True)
pip=[sys.executable,'-m','pip','--python',str(python),'install','--no-cache-dir','--only-binary=:all:','--no-compile']
subprocess.run([*pip,'--dry-run','--ignore-installed','--report',str(out/'resolution.json'),'matplotlib==3.10.8'],check=True)
resolution=json.loads((out/'resolution.json').read_bytes())
lines=['# Plotting-only environment; resolver report retained separately.','--only-binary=:all:']
for item in sorted(resolution['install'],key=lambda x:x['metadata']['name'].lower()):
    lines.append(f"{item['metadata']['name']}=={item['metadata']['version']} --hash=sha256:{item['download_info']['archive_info']['hashes']['sha256']}")
lock=out/'analysis.lock';lock.write_text('\n'.join(lines)+'\n')
subprocess.run([*pip,'--require-hashes','-r',str(lock),'--report',str(out/'install-report.json')],check=True)
print(json.dumps({'analysis_executable':str(python),'analysis_lock':str(lock),'scope':'plotting only; no core numerical imports/lock upgrade'}))
