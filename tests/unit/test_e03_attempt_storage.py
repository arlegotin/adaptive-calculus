"""E03 recorder storage arithmetic; synthetic inventory, no scientific work."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def load_recorder(monkeypatch):
    path = Path('docs/harness/e03-attempt.py')
    spec = importlib.util.spec_from_file_location('e03_attempt', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, 'argv', [str(path)])
    try:
        spec.loader.exec_module(module)
    except ValueError as error:
        pytest.fail(f'Recorder guard cannot be exercised before execution: {error}')
    return module


def startup():
    return {'storage': {'inventory': [{'path': 'longé/'+('x'*400), 'bytes': 123} for _ in range(20)]}}


def prepare(recorder, tmp_path, **changes):
    kwargs = {'archive_bytes': 1024, 'other_fixed_bytes': 53,
        'child_output_allowance': 4096, 'finish_allowance': 2048,
        'store_bytes': 0, 'data_free': 128*1024**3, 'git_free': 1024**3}
    kwargs.update(changes)
    return recorder.admit_attempt(tmp_path/'attempt', startup(), **kwargs)


def test_encoded_startup_and_archives_are_in_complete_forecast(monkeypatch, tmp_path):
    recorder = load_recorder(monkeypatch)
    encoded = prepare(recorder, tmp_path)
    payload = json.loads(encoded)
    budget = payload['storage']['write_budget']
    assert budget['startup_bytes'] == len(encoded)
    assert budget['minimum_forecast_bytes'] == 1024 + 53 + len(encoded) + 4096 + 2048
    assert payload['storage']['complete_remaining_fix_forecast_bytes'] == budget['minimum_forecast_bytes']
    assert (tmp_path/'attempt').is_dir()


def test_underestimated_caller_forecast_rejects_before_directory(monkeypatch, tmp_path):
    recorder = load_recorder(monkeypatch)
    with pytest.raises(ValueError, match='forecast'):
        prepare(recorder, tmp_path, forecast=8192)
    assert not (tmp_path/'attempt').exists()


@pytest.mark.parametrize('boundary', ['store', 'data', 'git'])
def test_preflight_rejects_boundary_crossing_without_writes(monkeypatch, tmp_path, boundary):
    recorder = load_recorder(monkeypatch)
    options = {'forecast': 65536}
    if boundary == 'store': options['store_bytes'] = 32*1024**3 - 65536 + 1
    if boundary == 'data': options['data_free'] = 64*1024**3 + 65536 - 1
    if boundary == 'git': options['git_free'] = 512*1024**2 + 16*1024**2 - 1
    with pytest.raises(ValueError):
        prepare(recorder, tmp_path, **options)
    assert not (tmp_path/'attempt').exists()


def test_exact_boundaries_admit_encoded_receipt(monkeypatch, tmp_path):
    recorder = load_recorder(monkeypatch)
    encoded = prepare(recorder, tmp_path, forecast=65536, store_bytes=32*1024**3-65536,
        data_free=64*1024**3+65536, git_free=512*1024**2+16*1024**2)
    assert json.loads(encoded)['storage']['complete_remaining_fix_forecast_bytes'] == 65536


@pytest.mark.parametrize('violation', ['store', 'data', 'git'])
def test_postflight_checks_whole_store_and_both_reserves(monkeypatch, violation):
    recorder = load_recorder(monkeypatch)
    values = {'store_bytes': 32*1024**3, 'data_free': 64*1024**3, 'git_free': 512*1024**2}
    recorder.audit_storage(**values)
    key = {'store': 'store_bytes', 'data': 'data_free', 'git': 'git_free'}[violation]
    values[key] += 1 if violation == 'store' else -1
    with pytest.raises(ValueError):
        recorder.audit_storage(**values)
