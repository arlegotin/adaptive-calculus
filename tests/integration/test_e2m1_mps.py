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
    assert report['checks']['subnormal']['code_mismatches'] == 0
    assert report['checks']['subnormal']['reconstruction_bit_mismatches'] == 0
    assert report['checks']['subnormal']['parameters'] >= 4096
    assert report['checks']['subnormal']['tiny_direction_mismatches'] == 0
    assert report['checks']['invalid_inputs']['accepted_invalid'] == 0
    assert report['checks']['saturation']['code_mismatches'] == 0
    assert report['checks']['saturation']['value_max_error'] == 0
    assert report['checks']['scalable']['parameters'] >= 65536
    assert report['checks']['scalable']['code_mismatches'] == 0


@pytest.mark.parametrize('scale_bits', [1, 3, 5, 0x00400000, 0x007fffff, 0x00800000])
def test_subnormal_e2m1_mps(scale_bits):
    import numpy as np
    from ralup.quantization import e2m1_oracle, quantize_e2m1
    if not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    scale = np.array(scale_bits, dtype=np.uint32).view(np.float32)
    factors = np.array([0., 1., -1., 6., -6., .5, -.5, .75, -.75, 1.5, -1.5], dtype=np.float32)
    x = factors * scale
    expected = e2m1_oracle(np.divide(x, scale, dtype=np.float32), np.array(1.))
    levels = np.array([-6, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 4, 6])
    expected_codes = np.searchsorted(levels, expected).astype(np.uint8)
    expected_values = (expected * float(scale)).astype(np.float32)
    values, codes = quantize_e2m1(torch.tensor(x, device='mps'), torch.tensor(scale, device='mps'))
    assert values.device.type == codes.device.type == 'mps'
    np.testing.assert_array_equal(codes.cpu().numpy(), expected_codes)
    np.testing.assert_array_equal(values.view(torch.int32).cpu().numpy().view(np.uint32), expected_values.view(np.uint32))


def test_subnormal_underflow_ties_and_directions_mps():
    import numpy as np
    from ralup.quantization import directional_neighbor, quantize_e2m1
    if not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    minimum = np.nextafter(np.float32(0), np.float32(1))
    x = np.array([1, -1, 4, -4, 2, -2, 7, -7], dtype=np.float32) * minimum
    scale = np.array([3, 3, 3, 3, 5, 5, 5, 5], dtype=np.float32) * minimum
    values, codes = quantize_e2m1(torch.tensor(x, device='mps'), torch.tensor(scale, device='mps'))
    np.testing.assert_array_equal(codes.cpu().numpy(), [8, 6, 10, 4, 8, 6, 10, 4])
    np.testing.assert_array_equal(values.view(torch.int32).cpu().numpy().view(np.uint32), [2, 0x80000002, 4, 0x80000004, 2, 0x80000002, 8, 0x80000008])
    neighbor, feasible = directional_neighbor(torch.full((4,), 7, dtype=torch.uint8, device='mps'),
        torch.tensor(np.array([-minimum, minimum, -0., 0.], dtype=np.float32), device='mps'))
    np.testing.assert_array_equal(neighbor.cpu().numpy(), [6, 8, 7, 7])
    np.testing.assert_array_equal(feasible.cpu().numpy(), [True, True, False, False])
