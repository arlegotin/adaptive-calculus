"""Independent cell quadrature catches analytic moments and fitting mistakes."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

LEVELS = np.array([-6., -4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 6.])


def integrated_mse(a):
    q = a * LEVELS / 6
    bounds = np.r_[-np.inf, (q[:-1] + q[1:]) / 2, np.inf]
    return math.fsum(quad(lambda z: (z-value)**2 * math.exp(-z*z/2)/math.sqrt(2*math.pi), lo, hi,
                          epsabs=2e-14, epsrel=2e-13)[0]
                     for value, lo, hi in zip(q, bounds[:-1], bounds[1:]))


def test_gaussian_fit_deterministic_bracketed_and_independently_integrated():
    from ralup.quantization.scales import fit_gaussian_e2m1
    a = fit_gaussian_e2m1()
    b = fit_gaussian_e2m1()
    assert a == b
    assert a['a'] == pytest.approx(2.922475071373709, abs=2e-7)
    assert a['scale_constant'] == pytest.approx(a['a']/6, abs=1e-15)
    assert a['objective'] == pytest.approx(.01268490413871003, abs=2e-13)
    assert a['objective'] == pytest.approx(integrated_mse(a['a']), abs=2e-13)
    assert abs(a['derivative_residual']) < 2e-9
    assert a['integration_residual'] < 2e-12
    assert len(a['bracket_history']) > 1
    assert a['bracket_objectives'][1] < min(a['bracket_objectives'][0], a['bracket_objectives'][2])


@pytest.mark.parametrize('a', [.25, .5, 1., 2., 4., 8., 16.])
def test_analytic_gaussian_moments_match_independent_integral(a):
    from ralup.quantization.scales import gaussian_e2m1_mse
    assert gaussian_e2m1_mse(a) == pytest.approx(integrated_mse(a), abs=2e-12)


def test_quantizer_cli_reports_actual_fit_context_and_surrogate_checks(tmp_path):
    import json
    import subprocess
    import sys
    output = tmp_path / 'quantizer.json'
    result = subprocess.run([sys.executable, '-m', 'ralup', 'check-numerics', '--suite', 'quantizer',
                             '--device', 'cpu', '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    checks = report['checks']['context']
    assert checks['fit']['integration_residual'] < 2e-12
    assert checks['fit']['objective'] == pytest.approx(.01268490413871, abs=2e-13)
    assert checks['microbatch_backward']['independent_backward_calls'] == 2
    assert checks['detached_context']['tensors_requiring_grad'] == 0
    assert checks['grouping_labels'] == ['last_axis_row', 'whole_tensor']
    assert checks['gradual_underflow']['sparse_normal_max_gaussian_scale_bits'] == {
        'quest_fp4_row_reference': [[2042958]], 'quest_fp4_tensor_reference': [[2042958]],
    }
    assert report['acceptance_gate'] == 'F03_F04_only; full_WP01_pending'
