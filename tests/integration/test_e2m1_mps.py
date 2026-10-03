"""Actual fallback-disabled device parity against float64 and FP32-coordinate oracles."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def test_e2m1_mps(tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable in this process; run unsandboxed fallback-disabled check-numerics')
    output = tmp_path / 'mps-quantizer.json'
    result = subprocess.run([sys.executable, '-m', 'ralup', 'check-numerics', '--suite', 'quantizer',
                             '--device', 'mps', '--output', str(output)],
                            env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0', PYTHONPATH=str(ROOT / 'src')),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(output.read_text())
    assert report['fallback_environment'] == '0' and report['device'] == 'mps'
    assert report['checks']['exact']['code_mismatches'] == 0
    assert report['checks']['exact']['value_max_error'] == 0
    assert report['checks']['scaled_fp32']['code_mismatches'] == 0
    assert report['checks']['neighbors']['mismatches'] == 0
    assert report['checks']['invalid_inputs']['accepted_invalid'] == 0
    assert report['checks']['saturation']['code_mismatches'] == 0
    assert report['checks']['saturation']['value_max_error'] == 0
    assert report['checks']['scalable']['parameters'] >= 65536
    assert report['checks']['scalable']['code_mismatches'] == 0
