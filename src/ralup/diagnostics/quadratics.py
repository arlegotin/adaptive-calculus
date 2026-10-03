"""Seeded cheap CPU-float64 diagnostics, including an explicitly privileged oracle."""
import hashlib
import itertools
import math
import numpy as np
import torch
from ..policy.actions import assist_boundaries
from ..quantization.context import prepare_context
from ..quantization.formats import E2M1_VALUES
from ..quantization.oracle import e2m1_oracle


def run_quadratic_diagnostics(seed: int, horizons: list[int]) -> dict:
    """Enumerate one immediate {-1,0,+1} ordinal move, then K baseline steps.

    Every candidate uses the same deterministic quantized-gradient baseline,
    learning rate 0.1/condition_number, persistent continuous shadows and unit
    fixed scale. Oracle privilege/cost are descriptive, never a controller budget.
    """
    if not horizons or any(isinstance(k, bool) or not isinstance(k, int) or k < 0 for k in horizons):
        raise ValueError('Horizons must be nonempty nonnegative integer steps')
    rng = np.random.default_rng(seed)
    u, d = torch.tensor([[1.]]), torch.tensor([[.04]])
    ctx = prepare_context(u, {'name': 'row_fixed_max', 'previous_scales': torch.ones((1, 1))}, 0, None)
    assisted, _ = assist_boundaries(u, d, .125, ctx)
    after = float(e2m1_oracle(assisted.numpy(), np.array(1.)).item())
    report = {'seed': seed, 'horizons': list(horizons), 'diagnostic_only': True,
              'target_1_1_before': (1.-1.1)**2, 'target_1_1_after': (after-1.1)**2,
              'target_1_8_before': (1.-1.8)**2, 'target_1_8_after': (after-1.8)**2}
    w = np.array([0., .5, 1., 2.])
    qs = [.5, .9, .99, .999, 1.]
    target = 1.1
    multiplied = np.asarray(qs)[:, None]*w
    exponentiated = w*np.exp(np.log(np.asarray(qs)))[:, None]
    report['q_exponential'] = {'q_values': qs, 'definition': 'q*w=w*exp(log(q)); q>0',
                               'identity_max_abs_error': float(np.max(np.abs(multiplied-exponentiated))),
                               'identity_rtol': 4*float(np.finfo(np.float64).eps),
                               'q_one_exact': bool(np.array_equal(exponentiated[-1], w)),
                               'interpretation': 'positive_exponential_reparameterization; no_separate_q_action'}
    signed_w = rng.uniform(-2., 2., 1024)
    epsilon = rng.uniform(-.1, .1, 1024)
    positive_q = np.array([math.exp(float(value)) for value in epsilon])
    by_q, by_exp = signed_w*positive_q, signed_w*np.exp(epsilon)
    signed_error = np.abs(by_q-by_exp)
    report['q_exponential']['sampled_signed_identity'] = {
        'sample_count': signed_w.size, 'negative_w_count': int(np.count_nonzero(signed_w < 0)),
        'positive_w_count': int(np.count_nonzero(signed_w > 0)), 'epsilon_interval': [-.1, .1],
        'q_min': float(positive_q.min()), 'q_max': float(positive_q.max()),
        'input_sha256': hashlib.sha256(signed_w.tobytes()+epsilon.tobytes()).hexdigest(),
        'max_abs_error': float(signed_error.max()),
        'relative_tolerance_exceeded': int(np.count_nonzero(signed_error > 4*np.finfo(np.float64).eps*np.abs(by_exp))),
        'independent_exponentials': 'Python_math_exp_vs_NumPy_exp; signed_w_is_never_logged'}
    ordinary = w-target
    derivatives, quotient_errors = [], []
    for q in qs:
        analytic = ordinary+.5*(q-1)*w
        derivatives.append(analytic.tolist())
        if q != 1.:
            # w=0 has 0/0 in the displayed quotient. Use its continuous
            # quadratic extension -target explicitly, not a numerical divide.
            nonzero = w != 0
            quotient = (.5*(q*w[nonzero]-target)**2-.5*(w[nonzero]-target)**2)/((q-1)*w[nonzero])
            quotient_errors.append(float(np.max(np.abs(quotient-analytic[nonzero]))))
    report['finite_q_quadratic'] = {'q_values': qs, 'w_values': w.tolist(), 'target': target,
                                    'definition': '(f(q*w)-f(w))/((q-1)*w)',
                                    'objective': 'f(w)=.5*(w-target)**2',
                                    'analytic_identity': 'w-target+.5*(q-1)*w',
                                    'zero_domain_convention': 'continuous_extension_-target',
                                    'jackson_derivatives': derivatives,
                                    'derivative_at_q_one': ordinary.tolist(),
                                    'analytic_max_abs_error': max(quotient_errors),
                                    'quotient_atol': 4e-12,
                                    'limit_max_abs_error': float(np.max(np.abs(np.asarray(derivatives[-2])-ordinary))),
                                    'limit_analytic_bound': .5*(1-.999)*float(np.max(np.abs(w)))}
    ww, aa, qq = 1.7, -.2, 1.3
    direct = (((qq*ww-aa)**2-(ww-aa)**2)/2)/((qq-1)*ww)
    derived = ww-aa+.5*(qq-1)*ww
    report['finite_q_quadratic']['appendix_a'] = {'w': ww, 'target': aa, 'q': qq,
                                                 'direct': direct, 'derived': derived,
                                                 'max_abs_error': abs(direct-derived),
                                                 'evidence': 'independently_rerun_in_this_implementation'}
    x = np.float32(1.25)
    next_x = np.nextafter(x, np.float32(np.inf))
    report['fp32_nextafter'] = {'base_bits': int(x.view(np.int32)), 'next_bits': int(next_x.view(np.int32)),
                                'ulp': float(next_x-x), 'specified_nudge': .04/1024}
    shadow = 1.
    trajectory, reset_changes = [], 0
    for step in range(1, 17):
        shadow += .04
        trajectory.append(float(e2m1_oracle(np.array(shadow), np.array(1.))))
        reset_changes += int(e2m1_oracle(np.array(1.+.04), np.array(1.)) != 1.)
    report['persistent_shadow'] = {'trajectory': trajectory,
                                   'first_code_change_step': next(i+1 for i, q in enumerate(trajectory) if q != 1.),
                                   'reset_code_change_count': reset_changes}
    probability = (1.2-1.)/(1.5-1.)
    draws = rng.random(100000) < probability
    report['stochastic_expectation'] = {'coordinate': 1.2, 'lower': 1., 'upper': 1.5,
                                        'upper_probability': probability, 'expectation': 1.+.5*probability,
                                        'sample_count': draws.size, 'measured_mean': float((1.+.5*draws).mean()),
                                        'sampling_standard_error': float(.5*np.sqrt(probability*(1-probability)/draws.size)),
                                        'scope': 'diagnostic_expectation; stochastic_quantizer_generation_owned_by_M05'}
    errors = []
    for scale in (.125, .5, 2., 8.):
        c = prepare_context(u*scale, {'name': 'row_fixed_max', 'previous_scales': torch.tensor([[scale]])}, 0, None)
        actual, _ = assist_boundaries(u*scale, d*scale, .125, c)
        errors.append(float((actual-assisted*scale).abs().max()))
    report['scale_homogeneity'] = {'scales': [.125, .5, 2., 8.], 'max_abs_error': max(errors)}
    coupled = []
    levels = np.asarray(E2M1_VALUES)
    for dimension in (2, 4, 8):
        for condition in (1, 10, 100):
            basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
            matrix = (basis*np.geomspace(1., condition, dimension))@basis.T
            target = rng.uniform(-1.8, 1.8, dimension)
            initial_codes = rng.integers(4, 11, dimension)
            initial = levels[initial_codes]
            actions = np.array(list(itertools.product((-1, 0, 1), repeat=dimension)))
            candidates = levels[initial_codes[None, :]+actions]
            rate = .1/condition
            def losses(shadows):
                difference = e2m1_oracle(shadows, np.array(1.))-target
                return np.einsum('bi,ij,bj->b', difference, matrix, difference)/2
            for horizon in horizons:
                shadows = candidates.copy()
                trajectory = [losses(shadows)]
                for _ in range(horizon):
                    shadows -= rate*(e2m1_oracle(shadows, np.array(1.))-target)@matrix
                    trajectory.append(losses(shadows))
                best = int(np.argmin(trajectory[-1]))
                zero = int(np.flatnonzero((actions == 0).all(axis=1))[0])
                coupled.append({'dimension': dimension, 'condition_number': condition, 'horizon': horizon,
                                'matrix': matrix.tolist(), 'target': target.tolist(), 'initial': initial.tolist(),
                                'fixture_sha256': hashlib.sha256(matrix.tobytes()+target.tobytes()+initial.tobytes()).hexdigest(),
                                'baseline_learning_rate': rate, 'enumerated_actions': len(actions),
                                'oracle_objective_evaluations': len(actions)*(horizon+1),
                                'oracle_baseline_steps': len(actions)*horizon,
                                'oracle_privilege': 'known_quadratic_objective_and_full_candidate_rollouts',
                                'oracle_action': actions[best].tolist(),
                                'oracle_final_loss': float(trajectory[-1][best]),
                                'baseline_final_loss': float(trajectory[-1][zero]),
                                'oracle_trajectory': [float(v[best]) for v in trajectory],
                                'baseline_trajectory': [float(v[zero]) for v in trajectory]})
    report['coupled'] = coupled
    return report
