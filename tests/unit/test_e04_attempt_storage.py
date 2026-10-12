"""Admission includes encoded complete inventory and bounded child/finish writes."""
import importlib.util
from pathlib import Path

import pytest


def helper():
    path=Path(__file__).resolve().parents[2]/'docs/harness/e04-attempt.py'
    spec=importlib.util.spec_from_file_location('e04_attempt',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_encoded_inventory_cannot_be_omitted_from_complete_forecast(tmp_path):
    module=helper();base=tmp_path/'refused'
    startup={'storage':{'inventory':[{'path':'x'*1000,'bytes':100} for _ in range(100)]}}
    with pytest.raises(ValueError,match='below complete write requirement'):
        module.admit_attempt(base,startup,archive_bytes=100,other_fixed_bytes=100,
            child_output_allowance=100,finish_allowance=100,store_bytes=1,
            data_free=70*module.GIB,git_free=module.GIB,forecast=1000)
    assert not base.exists()


def test_complete_forecast_and_reserve_enforced_before_mkdir(tmp_path):
    module=helper();base=tmp_path/'refused'
    with pytest.raises(ValueError,match='reserve'):
        module.admit_attempt(base,{'storage':{}},archive_bytes=100,other_fixed_bytes=100,
            child_output_allowance=module.MIB,finish_allowance=module.MIB,store_bytes=1,
            data_free=64*module.GIB,git_free=module.GIB)
    assert not base.exists()


def test_exact_encoded_size_converges_and_strict_json_rejects_nonfinite(tmp_path):
    module=helper();startup={'storage':{}}
    encoded=module.admit_attempt(tmp_path/'admitted',startup,archive_bytes=100,other_fixed_bytes=200,
        child_output_allowance=300,finish_allowance=400,store_bytes=1,data_free=70*module.GIB,git_free=module.GIB)
    assert startup['storage']['write_budget']['startup_bytes']==len(encoded)
    assert startup['storage']['write_budget']['minimum_forecast_bytes']==1000+len(encoded)
    with pytest.raises(ValueError):module.encoded_json({'not_science':float('nan')})
