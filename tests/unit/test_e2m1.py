"""Encoding-derived cases catch ordinal-parity rounding and FP32 boundary drift."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from ralup.quantization import directional_neighbor, e2m1_oracle, quantize_e2m1
from ralup.quantization.formats import E2M1_VALUES, E2M1_NIBBLES, E2M1_EVEN

ROOT = Path(__file__).resolve().parents[2]
LEVELS = np.array([-6, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 4, 6])
TIES = np.array([-5, -3.5, -2.5, -1.75, -1.25, -.75, -.25, .25, .75, 1.25, 1.75, 2.5, 3.5, 5])
WINNERS = np.array([-4, -4, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 4, 4])
ORDINALS = np.array([1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13], dtype=np.uint8)


def test_encoding_tables_match_independent_primary_source_reference():
    reference = json.loads((ROOT / 'docs/numerics/e2m1-encoding-reference.json').read_text())
    assert list(E2M1_VALUES) == reference['unique_numeric_levels']
    assert list(E2M1_NIBBLES) == reference['canonical_nibbles_in_numeric_order']
    by_nibble = {entry['nibble']: entry for entry in reference['encodings']}
    assert list(E2M1_EVEN) == [by_nibble[n]['lsb_even'] for n in E2M1_NIBBLES]
    np.testing.assert_array_equal(e2m1_oracle(TIES, np.array(1.)), WINNERS)


def test_oracle_all_fixed_points_and_signed_zero():
    np.testing.assert_array_equal(e2m1_oracle(LEVELS, np.array(1.)), LEVELS)
    result = e2m1_oracle(np.array([-0., 0., -.25, .25]), np.array(1.))
    np.testing.assert_array_equal(result, [0., 0., 0., 0.])
    assert not np.signbit(result).any()
    assert result.dtype == np.float64


def test_oracle_every_midpoint_and_adjacent_float64_inputs():
    np.testing.assert_array_equal(e2m1_oracle(TIES, np.array(1.)), WINNERS)
    np.testing.assert_array_equal(e2m1_oracle(np.nextafter(TIES, -np.inf), np.array(1.)), LEVELS[:-1])
    np.testing.assert_array_equal(e2m1_oracle(np.nextafter(TIES, np.inf), np.array(1.)), LEVELS[1:])


def test_sorted_ordinal_even_rule_is_wrong_at_negative_midpoint():
    # Both distances are one at -5; ordinal-even picks -6, encoding-even picks -4.
    distances = np.abs(LEVELS - (-5.))
    tied_ordinals = np.flatnonzero(distances == distances.min())
    naive = LEVELS[tied_ordinals[tied_ordinals % 2 == 0][0]]
    assert naive == -6
    assert e2m1_oracle(np.array(-5.), np.array(1.)) == -4
    assert E2M1_NIBBLES[1] == 14 and E2M1_EVEN[1]


@pytest.mark.parametrize('scale', [.125, .5, 1., 2., 8.])
def test_tensor_exact_fixed_points_ties_and_adjacent_float32(scale):
    ties = TIES.astype(np.float32)
    probes = np.concatenate([LEVELS.astype(np.float32), ties,
                             np.nextafter(ties, np.float32(-np.inf)),
                             np.nextafter(ties, np.float32(np.inf)), [-0., 0.]]).astype(np.float32)
    expected = np.concatenate([LEVELS, WINNERS, LEVELS[:-1], LEVELS[1:], [0., 0.]]) * scale
    expected_codes = np.concatenate([np.arange(15), ORDINALS, np.arange(14), np.arange(1, 15), [7, 7]])
    values, codes = quantize_e2m1(torch.tensor(probes * scale), torch.tensor(scale, dtype=torch.float32))
    np.testing.assert_array_equal(values.numpy(), expected)
    np.testing.assert_array_equal(codes.numpy(), expected_codes)
    assert values.dtype == torch.float32 and codes.dtype == torch.uint8
    assert not np.signbit(values.numpy()[values.numpy() == 0]).any()


def test_broadcast_scales_and_scalar_shapes():
    x = np.array([[.25, .75, 1.75], [-.5, -1.5, -3.5]], dtype=np.float32)
    scale = np.array([[1.], [2.]], dtype=np.float32)
    expected = np.array([[0., 1., 2.], [0., -2., -4.]])
    np.testing.assert_array_equal(e2m1_oracle(x, scale), expected)
    values, codes = quantize_e2m1(torch.tensor(x), torch.tensor(scale))
    np.testing.assert_array_equal(values.numpy(), expected)
    np.testing.assert_array_equal(codes.numpy(), [[7, 9, 11], [7, 5, 3]])
    values, codes = quantize_e2m1(torch.tensor(-.75), torch.tensor(1.))
    assert values.shape == codes.shape == torch.Size([])
    assert values == -1 and codes == 5


def test_finite_saturation_including_scaled_division_overflow():
    x = np.array([-np.finfo(np.float32).max, -7., 7., np.finfo(np.float32).max], dtype=np.float32)
    np.testing.assert_array_equal(e2m1_oracle(x, np.array(1.)), [-6., -6., 6., 6.])
    values, codes = quantize_e2m1(torch.tensor(x), torch.tensor(1.))
    np.testing.assert_array_equal(values.numpy(), [-6., -6., 6., 6.])
    np.testing.assert_array_equal(codes.numpy(), [0, 0, 14, 14])
    tiny = np.float32(np.finfo(np.float32).tiny)
    values, codes = quantize_e2m1(torch.tensor(x), torch.tensor(tiny))
    np.testing.assert_array_equal(codes.numpy(), [0, 0, 14, 14])
    np.testing.assert_array_equal(values.numpy(), np.array([-6, -6, 6, 6], dtype=np.float32) * tiny)
    np.testing.assert_array_equal(e2m1_oracle(np.array([-np.finfo(float).max, np.finfo(float).max]), np.array(np.finfo(float).tiny)), [-6*np.finfo(float).tiny, 6*np.finfo(float).tiny])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_reject_nonfinite_original_input(bad):
    with pytest.raises(ValueError):
        e2m1_oracle(np.array([bad]), np.array(1.))
    with pytest.raises(ValueError):
        quantize_e2m1(torch.tensor([bad]), torch.tensor(1.))


@pytest.mark.parametrize('bad', [0., -0., -1., np.nan, np.inf, -np.inf])
def test_reject_nonpositive_or_nonfinite_scale(bad):
    with pytest.raises(ValueError):
        e2m1_oracle(np.array([0.]), np.array(bad))
    with pytest.raises(ValueError):
        quantize_e2m1(torch.tensor([0.]), torch.tensor(bad))


def test_tensor_requires_fp32_and_compatible_broadcasts():
    with pytest.raises(TypeError):
        quantize_e2m1(torch.tensor([1.], dtype=torch.float64), torch.tensor(1.))
    with pytest.raises(TypeError):
        quantize_e2m1(torch.tensor([1.]), torch.tensor(1., dtype=torch.float64))
    with pytest.raises((ValueError, RuntimeError)):
        quantize_e2m1(torch.ones(2, 3), torch.ones(2))
    with pytest.raises(ValueError):
        e2m1_oracle(np.ones((2, 3)), np.ones(2))


def test_fp32_scaled_coordinate_roundoff_is_distinct_from_exact_float64_scale():
    x = np.array([.175, -.175], dtype=np.float32)
    scale = np.array(.1, dtype=np.float32)
    exact = e2m1_oracle(x, scale)
    np.testing.assert_allclose(exact, [1.5*float(scale), -1.5*float(scale)], rtol=0, atol=0)
    scaled_fp32 = np.divide(x, scale, dtype=np.float32)
    np.testing.assert_array_equal(scaled_fp32, [1.75, -1.75])
    values, codes = quantize_e2m1(torch.tensor(x), torch.tensor(scale))
    np.testing.assert_array_equal(codes.numpy(), [11, 3])
    np.testing.assert_array_equal(values.numpy(), np.array([2., -2.], dtype=np.float32) * scale)
    assert np.all(np.abs(values.numpy().astype(float) - exact) <= .5*float(scale) + np.finfo(np.float32).eps)


def test_every_directional_neighbor_and_boundaries():
    codes = torch.arange(15, dtype=torch.uint8)
    for direction, expected, feasible in [(-1., [0]+list(range(14)), [False]+[True]*14),
                                         (1., list(range(1, 15))+[14], [True]*14+[False]),
                                         (0., list(range(15)), [False]*15)]:
        neighbors, mask = directional_neighbor(codes, torch.tensor(direction))
        np.testing.assert_array_equal(neighbors.numpy(), expected)
        np.testing.assert_array_equal(mask.numpy(), feasible)
        assert neighbors.dtype == torch.uint8 and mask.dtype == torch.bool
    neighbor, mask = directional_neighbor(torch.tensor([7, 7], dtype=torch.uint8), torch.tensor([-2., 3.]))
    np.testing.assert_array_equal(neighbor.numpy(), [6, 8])
    assert mask.all()


@pytest.mark.parametrize('codes,direction', [(torch.tensor([15], dtype=torch.uint8), torch.tensor(1.)),
                                             (torch.tensor([-1]), torch.tensor(1.)),
                                             (torch.tensor([7], dtype=torch.uint8), torch.tensor(float('nan')))])
def test_directional_neighbor_rejects_invalid_codes_and_direction(codes, direction):
    with pytest.raises((ValueError, TypeError)):
        directional_neighbor(codes, direction)


def test_check_numerics_dispatch_runs_real_quantizer_and_persists_report(tmp_path):
    output = tmp_path / 'quantizer.json'
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTORCH_ENABLE_MPS_FALLBACK='0')
    result = subprocess.run([sys.executable, '-m', 'ralup', 'check-numerics', '--suite', 'quantizer',
                             '--device', 'cpu', '--output', str(output)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report['status'] == 'pass' and report['device'] == 'cpu'
    assert report['checks']['exact']['code_mismatches'] == 0
    assert report['checks']['scaled_fp32']['code_mismatches'] == 0
    assert report['checks']['exact_scale']['code_mismatches'] > 0
    assert report['checks']['neighbors']['mismatches'] == 0
    assert report['source']['ocp_pdf_sha256'] is None
    assert len(json.loads(result.stdout)['sha256']) == 64


def test_reconstruction_overflow_is_a_named_storage_error():
    for dtype in (np.float32, np.float64):
        maximum = np.finfo(dtype).max
        x = np.array([maximum], dtype=dtype)
        scale = np.array(maximum / 5.5, dtype=dtype)
        with pytest.raises(ValueError, match='reconstruction overflow'):
            e2m1_oracle(x, scale) if dtype == np.float64 else quantize_e2m1(torch.tensor(x), torch.tensor(scale))
