"""Discriminating diagnostics must report both benefit and harm, and oracle cost."""
import pytest


def run(seed=17, horizons=None):
    from ralup.diagnostics.quadratics import run_quadratic_diagnostics
    return run_quadratic_diagnostics(seed, [0, 2] if horizons is None else horizons)


def test_helpful_and_harmful_scalar_crossing_and_named_controls():
    report = run()
    assert report['target_1_1_after'] > report['target_1_1_before']
    assert report['target_1_8_after'] < report['target_1_8_before']
    assert report['q_exponential']['identity_max_abs_error'] < 2e-15
    assert report['finite_q_quadratic']['q_values'] == [.5, .9, .99, .999, 1.]
    assert report['fp32_nextafter']['next_bits'] > report['fp32_nextafter']['base_bits']
    assert report['persistent_shadow']['first_code_change_step'] > 1
    assert report['persistent_shadow']['reset_code_change_count'] == 0
    assert report['stochastic_expectation']['expectation'] == pytest.approx(1.2)
    assert report['scale_homogeneity']['max_abs_error'] == 0.


def test_coupled_seeded_trajectories_horizons_and_privilege():
    report = run()
    assert report == run()
    assert report != run(18)
    assert len(report['coupled']) == 18
    for case in report['coupled']:
        assert case['dimension'] in (2, 4, 8) and case['condition_number'] in (1, 10, 100)
        assert case['horizon'] in (0, 2)
        assert case['enumerated_actions'] == 3**case['dimension']
        assert case['oracle_objective_evaluations'] == 3**case['dimension']*(case['horizon']+1)
        assert case['oracle_privilege'] == 'known_quadratic_objective_and_full_candidate_rollouts'
        assert case['oracle_final_loss'] <= case['baseline_final_loss']+1e-12
        assert len(case['oracle_trajectory']) == case['horizon']+1
        assert len(case['baseline_trajectory']) == case['horizon']+1


def test_horizon_validation():
    for horizons in ([], [-1], [True], [1.5]):
        with pytest.raises(ValueError):
            run(horizons=horizons)


def test_numerics_cli_default_includes_mechanism(tmp_path):
    import json
    from ralup.__main__ import main
    output = tmp_path/'checks.json'
    assert main(['check-numerics', '--output', str(output)]) == 0
    result = json.loads(output.read_text())
    assert result['suite'] == 'numerics' and result['status'] == 'pass'
    assert result['checks']['actions']['coordinate_count_per_tau'] == 100000
    assert result['checks']['quadratics']['target_1_1_after'] > result['checks']['quadratics']['target_1_1_before']
    assert 'src/ralup/policy/actions.py' in result['source']['files']


@pytest.mark.parametrize('cohort,field,bad', [('scale_homogeneity', 'max_abs_error', 1.),
                                             ('persistent_shadow', 'reset_code_change_count', 1),
                                             ('q_exponential', 'identity_max_abs_error', .1)])
def test_numerical_gate_rejects_broken_mechanism_invariant(tmp_path, monkeypatch, cohort, field, bad):
    import json
    from ralup.diagnostics import quadratics
    from ralup.__main__ import main
    real = quadratics.run_quadratic_diagnostics
    def altered(seed, horizons):
        result = real(seed, horizons)
        result[cohort][field] = bad
        return result
    monkeypatch.setattr(quadratics, 'run_quadratic_diagnostics', altered)
    output = tmp_path/'negative.json'
    assert main(['check-numerics', '--output', str(output)]) == 1
    result = json.loads(output.read_text())
    assert result['status'] == 'negative' and result['blockers']


def test_jackson_quadratic_exact_formula_derivative_limit_and_zero_convention():
    report = run()['finite_q_quadratic']
    assert report['definition'] == '(f(q*w)-f(w))/((q-1)*w)'
    assert report['zero_domain_convention'] == 'continuous_extension_-target'
    assert report['analytic_max_abs_error'] < 2e-12
    assert report['derivative_at_q_one'] == [-1.1, -.6000000000000001, -.10000000000000009, .8999999999999999]
    assert report['limit_max_abs_error'] <= .0010000000001
    assert report['jackson_derivatives'][0] == [-1.1, -.7250000000000001, -.3500000000000001, .3999999999999999]


def test_original_appendix_jackson_fixture_and_sampled_signed_identity():
    report = run()
    fixture = report['finite_q_quadratic']['appendix_a']
    assert fixture['w'] == 1.7 and fixture['target'] == -.2 and fixture['q'] == 1.3
    assert fixture['direct'] == 2.1550000000000007 and fixture['derived'] == 2.155
    sampled = report['q_exponential']['sampled_signed_identity']
    assert sampled['sample_count'] == 1024
    assert sampled['negative_w_count'] > 0 and sampled['positive_w_count'] > 0
    assert sampled['relative_tolerance_exceeded'] == 0
