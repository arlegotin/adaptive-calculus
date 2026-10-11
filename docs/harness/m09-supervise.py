"""Task-owned monotonic child supervision, independent of a blocking model call."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run_bounded(command, seconds, output):
    if not 0 < seconds <= 7200:raise ValueError('Explicit finite child deadline required')
    started=time.perf_counter();process=subprocess.Popen(command)
    expired=False;signals=[]
    try:
        code=process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        expired=True
        process.send_signal(signal.SIGINT);signals.append('SIGINT')
        try:code=process.wait(timeout=2.)
        except subprocess.TimeoutExpired:
            process.kill();signals.append('SIGKILL');code=process.wait()
    record={'command':command,'owned_pid':process.pid,'child_deadline_seconds':seconds,
        'shutdown_grace_seconds':2.,'deadline_exceeded':expired,'signals':signals,'child_returncode':code,
        'complete_seconds':time.perf_counter()-started,
        'cost_scope':'this supervisor interval encloses its child; never add child wall again; outer recorder encloses both',
        'partial_scientific_work':'consult retained block/step records; any in-flight work is unknown'}
    with Path(output).open('x') as stream:
        json.dump(record,stream,sort_keys=True,indent=2);stream.flush();os.fsync(stream.fileno())
    return 124 if expired else code


if __name__=='__main__':
    raise SystemExit(run_bounded([sys.executable,*sys.argv[2:]],float(sys.argv[1]),
        Path(os.environ['M09_ATTEMPT_DIR'])/'supervision.json'))
