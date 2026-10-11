"""Actual child deadlines cannot wait for a blocked scientific callback."""
import importlib.util
import json
from pathlib import Path
import sys
import time


def module():
    path=Path(__file__).parents[2]/'docs/harness/m07-supervise.py'
    spec=importlib.util.spec_from_file_location('m07_supervise',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_monotonic_supervision_interrupts_a_blocking_owned_child(tmp_path):
    runner=module();started=time.perf_counter()
    code=runner.run_bounded([sys.executable,'-c','import time; time.sleep(10)'],.2,tmp_path/'supervision.json')
    record=json.loads((tmp_path/'supervision.json').read_bytes())
    assert code != 0 and record['deadline_exceeded'] is True
    assert record['child_returncode'] < 0
    assert record['complete_seconds'] < 3 and time.perf_counter()-started < 3
    assert record['partial_scientific_work']=='consult retained block/step records; any in-flight work is unknown'


def test_supervision_retains_actual_normal_failure_status(tmp_path):
    runner=module()
    assert runner.run_bounded([sys.executable,'-c','raise SystemExit(7)'],2,tmp_path/'supervision.json') == 7
    record=json.loads((tmp_path/'supervision.json').read_bytes())
    assert record['child_returncode']==7 and record['deadline_exceeded'] is False
