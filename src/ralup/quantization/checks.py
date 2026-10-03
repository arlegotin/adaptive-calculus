"""Measured F03/F04 diagnostics; this is not the full WP01 mechanism gate."""
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from ..config import canonical_bytes
from ..records import ArtifactRef, sha256_file
from .formats import E2M1_VALUES, directional_neighbor, quantize_e2m1
from .oracle import e2m1_oracle

# Frozen before device runs. Codes/exact cases have no tolerance. Reconstruction
# alone allows two FP32 eps relative, zero absolute; midpoint code drift is
# reported separately, never swallowed by an output-value tolerance.
VALUE_RTOL = 2 * float(np.finfo(np.float32).eps)
VALUE_ATOL = 0.0


def _reference(x, scale, *, rounded_coordinates):
    with np.errstate(over='ignore', under='ignore'):
        coordinates = (x / scale).astype(np.float32) if rounded_coordinates else x.astype(np.float64) / scale.astype(np.float64)
    # Internal overflow is a saturation event, not an original nonfinite input.
    coordinates = np.clip(coordinates, -6., 6.)
    q = e2m1_oracle(coordinates, np.array(1.))
    return q * scale.astype(np.float64), np.searchsorted(np.asarray(E2M1_VALUES), q).astype(np.uint8)


def _comparison(values, codes, expected_values, expected_codes):
    delta = np.abs(values.astype(np.float64) - expected_values)
    return {'parameters': int(values.size), 'code_mismatches': int(np.count_nonzero(codes != expected_codes)),
            'value_max_error': float(delta.max(initial=0)),
            'value_tolerance_exceeded': int(np.count_nonzero(delta > VALUE_ATOL + VALUE_RTOL*np.abs(expected_values)))}


def _run_checks(device):
    levels = np.array([-6, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    ties = np.array([-5, -3.5, -2.5, -1.75, -1.25, -.75, -.25, .25, .75, 1.25, 1.75, 2.5, 3.5, 5], dtype=np.float32)
    chosen = np.array([1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13], dtype=np.uint8)
    probes = np.concatenate([levels, ties, np.nextafter(ties, np.float32(-np.inf)),
                             np.nextafter(ties, np.float32(np.inf)), [-0., 0., -1e20, 1e20]]).astype(np.float32)
    ordinal = np.concatenate([np.arange(15), chosen, np.arange(14), np.arange(1, 15), [7, 7, 0, 14]]).astype(np.uint8)
    scales = np.array([.125, .5, 1., 2., 8.], dtype=np.float32)[:, None]
    x = probes[None, :] * scales
    values_t, codes_t = quantize_e2m1(torch.tensor(x, device=device), torch.tensor(scales, device=device))
    if values_t.device.type != device or codes_t.device.type != device:
        raise AssertionError('Quantizer outputs left the requested device')
    values, codes = values_t.cpu().numpy(), codes_t.cpu().numpy()
    oracle = e2m1_oracle(x, scales)
    expected_codes = np.broadcast_to(ordinal, x.shape)
    exact = _comparison(values, codes, oracle, expected_codes)
    exact['signed_zero_count'] = int(np.count_nonzero(np.signbit(values[values == 0])))
    if not np.array_equal(oracle, levels[expected_codes]*scales):
        raise AssertionError('Float64 oracle disagrees with hand-derived exact cases')

    # Non-power-of-two scales deliberately expose division-rounding boundaries.
    scales = np.array([.1, .3, 1.1, 3.], dtype=np.float32)[:, None]
    x = np.concatenate([ties, np.nextafter(ties, np.float32(-np.inf)),
                         np.nextafter(ties, np.float32(np.inf))])[None, :] * scales
    values_t, codes_t = quantize_e2m1(torch.tensor(x, device=device), torch.tensor(scales, device=device))
    if values_t.device.type != device or codes_t.device.type != device:
        raise AssertionError('Quantizer outputs left the requested device')
    values, codes = values_t.cpu().numpy(), codes_t.cpu().numpy()
    expected_values, expected_codes = _reference(x, scales, rounded_coordinates=True)
    scaled_fp32 = _comparison(values, codes, expected_values, expected_codes)
    exact_values, exact_codes = _reference(x, scales, rounded_coordinates=False)
    exact_scale = _comparison(values, codes, exact_values, exact_codes)
    # A different code is allowed here only when exact division is within one
    # FP32 ULP of a midpoint and rounded division lands on that midpoint.
    coordinates64 = x.astype(np.float64) / scales.astype(np.float64)
    coordinates32 = x / scales
    differing = codes != exact_codes
    records = []
    unexpected = 0
    for row, column in np.argwhere(differing):
        midpoint = float(coordinates32[row, column])
        distance = abs(float(coordinates64[row, column]) - midpoint)
        ulp = abs(float(np.spacing(np.float32(midpoint))))
        if midpoint not in ties or distance > ulp or abs(int(codes[row, column])-int(exact_codes[row, column])) != 1:
            unexpected += 1
        records.append({'row': int(row), 'column': int(column), 'x': float(x[row, column]),
                        'scale': float(scales[row, 0]), 'coordinate_float64': float(coordinates64[row, column]),
                        'coordinate_float32': midpoint, 'distance_to_tie': distance, 'one_fp32_ulp': ulp,
                        'device_ordinal': int(codes[row, column]), 'exact_scale_ordinal': int(exact_codes[row, column]),
                        'device_value': float(values[row, column]), 'exact_scale_value': float(exact_values[row, column])})
    exact_scale['unexplained_code_mismatches'] = unexpected
    exact_scale['discrepancies'] = records

    ordinal_t = torch.arange(15, device=device, dtype=torch.uint8)[None, :]
    direction_t = torch.tensor([[-1.], [0.], [1.]], device=device)
    neighbor, feasible = directional_neighbor(ordinal_t, direction_t)
    if neighbor.device.type != device or feasible.device.type != device:
        raise AssertionError('Neighbor outputs left the requested device')
    expected_neighbor = np.array([[0]+list(range(14)), list(range(15)), list(range(1, 15))+[14]])
    expected_mask = np.array([[False]+[True]*14, [False]*15, [True]*14+[False]])
    neighbors = {'mismatches': int(np.count_nonzero(neighbor.cpu().numpy() != expected_neighbor) +
                                  np.count_nonzero(feasible.cpu().numpy() != expected_mask))}

    invalid = [(float('nan'), 1.), (float('inf'), 1.), (-float('inf'), 1.),
               (0., 0.), (0., -1.), (0., float('nan')), (0., float('inf')),
               (np.finfo(np.float32).max, np.float32(np.finfo(np.float32).max / 5.5))]
    rejected = 0
    for original, scale in invalid:
        try:
            quantize_e2m1(torch.tensor([original], dtype=torch.float32, device=device),
                           torch.tensor(scale, dtype=torch.float32, device=device))
        except ValueError:
            rejected += 1
    invalid_inputs = {'cases': len(invalid), 'accepted_invalid': len(invalid)-rejected}

    # Original finite inputs whose FP32 quotient overflows still saturate.
    maximum = np.finfo(np.float32).max
    tiny = np.float32(np.finfo(np.float32).tiny)
    saturated_values, saturated_codes = quantize_e2m1(
        torch.tensor([-maximum, maximum], dtype=torch.float32, device=device),
        torch.tensor(tiny, dtype=torch.float32, device=device))
    if saturated_values.device.type != device or saturated_codes.device.type != device:
        raise AssertionError('Saturation outputs left the requested device')
    saturation_values = np.array([-6., 6.], dtype=np.float32) * tiny
    saturation = _comparison(saturated_values.cpu().numpy(), saturated_codes.cpu().numpy(),
                             saturation_values.astype(np.float64), np.array([0, 14], dtype=np.uint8))

    rng = np.random.default_rng(20261003)
    x = rng.normal(size=(64, 1024)).astype(np.float32)
    scales = np.exp(rng.uniform(-3., 2., size=(64, 1))).astype(np.float32)
    started = time.perf_counter()
    values_t, codes_t = quantize_e2m1(torch.tensor(x, device=device), torch.tensor(scales, device=device))
    if device == 'mps':
        torch.mps.synchronize()
    seconds = time.perf_counter() - started
    if values_t.device.type != device or codes_t.device.type != device:
        raise AssertionError('Scalable outputs left the requested device')
    expected_values, expected_codes = _reference(x, scales, rounded_coordinates=True)
    scalable = _comparison(values_t.cpu().numpy(), codes_t.cpu().numpy(), expected_values, expected_codes)
    scalable['seconds_including_input_transfer'] = seconds
    scalable['allocation_contract'] = '14 sequential comparisons, O(P) storage; no [P,15] or production tensor CPU copies'
    # Exercise a mixed domain so the exceptional path also preserves ordinary
    # normal operands. Expected storage bits are rounded once on CPU, including
    # gradual underflow ties; relative normal-output bounds do not apply here.
    bit_rng = np.random.default_rng(20261004)
    scale_bits = np.concatenate([np.array([1, 3, 5, 0x400000, 0x7fffff, 0x800000, 0x800001, 0x1000000], dtype=np.uint32),
                                bit_rng.integers(1, 0x800000, size=24, dtype=np.uint32),
                                bit_rng.integers(0x800000, 0x7d800000, size=32, dtype=np.uint32)])
    scales = scale_bits.view(np.float32)[:, None]
    factors = np.concatenate([levels, ties, np.nextafter(ties, np.float32(-np.inf)),
                              np.nextafter(ties, np.float32(np.inf)),
                              [-0., 0., 1/3, -1/3, 4/3, -4/3, 1.4]]).astype(np.float32)
    x = factors[None, :] * scales
    expected_values, expected_codes = _reference(x, scales, rounded_coordinates=True)
    expected_storage = expected_values.astype(np.float32)
    values_t, codes_t = quantize_e2m1(torch.tensor(x, device=device), torch.tensor(scales, device=device))
    if values_t.device.type != device or codes_t.device.type != device:
        raise AssertionError('Subnormal outputs left the requested device')
    # Read through integer dtype views so diagnostics cannot flush these bits.
    actual_bits = values_t.view(torch.int32).cpu().numpy().view(np.uint32)
    subnormal = _comparison(actual_bits.view(np.float32), codes_t.cpu().numpy(),
                            expected_storage.astype(np.float64), expected_codes)
    subnormal['reconstruction_bit_mismatches'] = int(np.count_nonzero(actual_bits != expected_storage.view(np.uint32)))
    subnormal['input_scale_sha256'] = hashlib.sha256(x.tobytes() + scales.tobytes()).hexdigest()
    subnormal['seed'] = 20261004
    subnormal['scale_bits'] = [int(value) for value in scale_bits]
    tiny = np.nextafter(np.float32(0), np.float32(1))
    neighbor, mask = directional_neighbor(torch.full((4,), 7, dtype=torch.uint8, device=device),
        torch.tensor(np.array([-tiny, tiny, -0., 0.], dtype=np.float32), device=device))
    subnormal['tiny_direction_mismatches'] = int(np.count_nonzero(neighbor.cpu().numpy() != [6, 8, 7, 7]) +
                                                np.count_nonzero(mask.cpu().numpy() != [True, True, False, False]))
    checks = {'exact': exact, 'scaled_fp32': scaled_fp32, 'exact_scale': exact_scale,
              'neighbors': neighbors, 'invalid_inputs': invalid_inputs, 'saturation': saturation, 'scalable': scalable,
              'subnormal': subnormal}
    passing = (exact['code_mismatches'] == exact['value_max_error'] == exact['signed_zero_count'] == 0 and
               scaled_fp32['code_mismatches'] == scaled_fp32['value_tolerance_exceeded'] == 0 and
               unexpected == neighbors['mismatches'] == invalid_inputs['accepted_invalid'] == 0 and
               saturation['code_mismatches'] == saturation['value_max_error'] == 0 and
               scalable['code_mismatches'] == scalable['value_tolerance_exceeded'] == 0 and
               subnormal['code_mismatches'] == subnormal['reconstruction_bit_mismatches'] == subnormal['tiny_direction_mismatches'] == 0)
    return checks, passing



# Frozen F04 absolute/relative bounds before CPU/MPS measurements. These apply
# only to the small explicit rotation/backward fixtures, not scientific training.
CONTEXT_ATOL = 8e-6
CONTEXT_RTOL = 2e-6


def _run_context_checks(device):
    from .context import prepare_context, quantized_view
    from .rotation import from_storage, hadamard, to_storage
    from .scales import fit_gaussian_e2m1

    fit = fit_gaussian_e2m1()
    if fit != fit_gaussian_e2m1():
        raise AssertionError('Gaussian fitting is not exactly deterministic')
    if abs(fit['a']-2.922475071373709) > 2e-7 or abs(fit['objective']-.01268490413871003) > 2e-13:
        raise AssertionError('Gaussian fitting differs from independent numerical reference')
    w = torch.tensor([[1., 2., -3., 4.], [0., 1., 3., -2.]], device=device, requires_grad=True)
    dense = torch.tensor([[1., 1., 1., 1.], [1., -1., 1., -1.],
                          [1., 1., -1., -1.], [1., -1., -1., 1.]], device=device) / 2
    double = hadamard(hadamard(w))
    torch.testing.assert_close(double, w, atol=CONTEXT_ATOL, rtol=CONTEXT_RTOL)
    torch.testing.assert_close(to_storage(w, 'full_hadamard'), w@dense, atol=CONTEXT_ATOL, rtol=CONTEXT_RTOL)
    torch.testing.assert_close(from_storage(to_storage(w, 'full_hadamard'), 'full_hadamard'), w,
                               atol=CONTEXT_ATOL, rtol=CONTEXT_RTOL)
    upstream = torch.tensor([[1., 2., -1., 3.], [-2., 0., 1., 4.]], device=device)
    rotation_gradient, = torch.autograd.grad((hadamard(w)*upstream).sum(), w)
    torch.testing.assert_close(rotation_gradient, upstream@dense, atol=CONTEXT_ATOL, rtol=CONTEXT_RTOL)
    rotation = {'id': 'full_hadamard', 'scope': 'full_last_axis',
                'involution_max_abs_error': float((double-w).detach().abs().max()),
                'gradient_max_abs_error': float((rotation_gradient-upstream@dense).abs().max())}
    impulse = torch.zeros((2, 256), device=device)
    impulse[:, 0] = 1.
    rotated_impulse = hadamard(impulse)
    torch.testing.assert_close(rotated_impulse, torch.full_like(impulse, 1/16), atol=CONTEXT_ATOL, rtol=CONTEXT_RTOL)
    rotation['full_row_width'] = 256
    rotation['full_row_impulse_max_abs_error'] = float((rotated_impulse-1/16).abs().max())
    zero_w = torch.tensor([[0., 0., 0., 0.], [6., -3., 0., 1.]], device=device)
    initial = prepare_context(zero_w, {'name': 'row_fixed_max'}, 0, None)
    torch.testing.assert_close(initial.scales, torch.ones((2, 1), device=device), atol=0., rtol=0.)
    if initial.zero_groups.cpu().tolist() != [[True], [False]]:
        raise AssertionError('Zero sentinel misclassified original rows')
    fixed = prepare_context(zero_w*12, {'name': 'row_fixed_max', 'previous_scales': initial.scales,
                                      'previous_zero_groups': initial.scale_zero_groups}, 1, None)
    dynamic = prepare_context(zero_w*12, {'name': 'row_dynamic_max'}, 1, None)
    torch.testing.assert_close(fixed.scales, initial.scales, atol=0., rtol=0.)
    torch.testing.assert_close(dynamic.scales, torch.tensor([[1.], [12.]], device=device), atol=0., rtol=0.)
    whole = prepare_context(zero_w*12, {'name': 'tensor_dynamic_max'}, 1, None)
    torch.testing.assert_close(whole.scales, torch.tensor([[12.]], device=device), atol=0., rtol=0.)
    warm = prepare_context(zero_w*12, {'name': 'row_fixed_max', 'warm_start': zero_w}, 1, None)
    torch.testing.assert_close(warm.scales, initial.scales, atol=0., rtol=0.)
    rms_cases = []
    for magnitude in (1e30, 1e-30):
        u = torch.tensor([[magnitude, -magnitude, magnitude, -magnitude]], device=device)
        c = prepare_context(u, {'name': 'quest_fp4_row_reference'}, 0, None)
        torch.testing.assert_close(c.scales, torch.tensor([[fit['scale_constant']*magnitude]], device=device),
                                   atol=0., rtol=CONTEXT_RTOL)
        rms_cases.append({'magnitude': magnitude, 'scale': float(c.scales.item())})
    underflow = torch.tensor([[1, 0, 0, 0]], dtype=torch.int32, device=device).view(torch.float32)
    try:
        prepare_context(underflow, {'name': 'row_dynamic_max'}, 0, None)
    except ValueError as error:
        if 'underflow' not in str(error):
            raise
    else:
        raise AssertionError('Nonzero underflow group was silently treated as zero')
    surrogate_results = {}
    for rule, expected in [('ste', [1., 1., 1., 1.]), ('trust', [1., 0., 1., 1.]),
                           ('clipped_ste', [1., 0., 1., 0.])]:
        shadow = torch.tensor([[.1, 7.1, -1., 6.2]], device=device, requires_grad=True)
        c = prepare_context(shadow, {'name': 'row_fixed_max', 'previous_scales': torch.ones((1, 1), device=device),
                                     'backward': rule}, 2, None)
        q = quantized_view(shadow, c)
        torch.testing.assert_close(q, torch.tensor([[0., 6., -1., 6.]], device=device), atol=0., rtol=0.)
        q.sum().backward()
        want = torch.tensor([expected], device=device)
        torch.testing.assert_close(shadow.grad, want, atol=0., rtol=0.)
        surrogate_results[rule] = {'gradient': shadow.grad.cpu().tolist(), 'max_abs_error': float((shadow.grad-want).abs().max())}
    rotated_trust_w = torch.tensor([[6.2, -7.1, 1., .1]], device=device, requires_grad=True)
    rotated_trust_c = prepare_context(rotated_trust_w, {'name': 'row_fixed_max', 'rotation': 'full_hadamard',
                                                      'previous_scales': torch.ones((1, 1), device=device),
                                                      'backward': 'trust'}, 0, None)
    (quantized_view(rotated_trust_w, rotated_trust_c)*torch.tensor([[1., 2., 3., 4.]], device=device)).sum().backward()
    torch.testing.assert_close(rotated_trust_w.grad, torch.tensor([[4., 0., -3., 1.]], device=device),
                               atol=CONTEXT_ATOL, rtol=CONTEXT_RTOL)
    surrogate_results['rotated_trust'] = {'gradient': rotated_trust_w.grad.cpu().tolist()}
    context = prepare_context(w, {'name': 'row_dynamic_max', 'rotation': 'full_hadamard'}, 7, None)
    cache_before = [t.clone() for t in context.numeric_tensors()]
    x1 = torch.tensor([[1., 2., 0., -1.]], device=device)
    x2 = torch.tensor([[-1., 3., 2., 1.]], device=device)
    first = quantized_view(w, context)
    ((x1@dense)@first.T).sum().backward()
    second = quantized_view(w, context)
    ((x2@dense)@second.T).sum().backward()
    torch.testing.assert_close(first, second, atol=0., rtol=0.)
    torch.testing.assert_close(w.grad, (x1+x2).expand_as(w), atol=CONTEXT_ATOL, rtol=CONTEXT_RTOL)
    if first.grad_fn is second.grad_fn:
        raise AssertionError('Microbatch view reused an autograd graph')
    for before, after in zip(cache_before, context.numeric_tensors()):
        torch.testing.assert_close(before, after, atol=0., rtol=0.)
        if after.requires_grad or after.grad_fn is not None or after.device.type != device:
            raise AssertionError('Numeric context retained a graph or left requested device')
    rng = torch.Generator().manual_seed(20261004)
    rng_before = rng.get_state().clone()
    supplied = torch.tensor([[7, 8, 9, 14]], dtype=torch.uint8, device=device)
    supplied_context = prepare_context(torch.tensor([[1., 2., 3., 6.]], device=device),
                                       {'name': 'row_dynamic_max', 'codes': supplied}, 3, rng)
    supplied.fill_(0)
    torch.testing.assert_close(supplied_context.storage_values, torch.tensor([[0., .5, 1., 6.]], device=device),
                               atol=0., rtol=0.)
    invalid_zero_codes = torch.full((2, 4), 7, dtype=torch.uint8, device=device)
    invalid_zero_codes[0, 0] = 8
    try:
        prepare_context(torch.zeros((2, 4), device=device), {'name': 'row_dynamic_max', 'codes': invalid_zero_codes}, 0, None)
    except ValueError as error:
        if 'zero group' not in str(error):
            raise
    else:
        raise AssertionError('Supplied realization turned an all-zero group nonzero')
    if not torch.equal(rng.get_state(), rng_before):
        raise AssertionError('Context preparation advanced the supplied training RNG')
    subnormal_scale = torch.tensor([[3]], dtype=torch.int32, device=device).view(torch.float32)
    subnormal_w = torch.tensor([[7, -2147483641]], dtype=torch.int32, device=device).view(torch.float32).requires_grad_()
    subnormal_c = prepare_context(subnormal_w, {'name': 'row_fixed_max', 'previous_scales': subnormal_scale}, 0, None)
    subnormal_q = quantized_view(subnormal_w, subnormal_c)
    if subnormal_q.view(torch.int32).cpu().tolist() != [[6, -2147483642]]:
        raise AssertionError('Fixed-scale STE changed F03 subnormal forward bits')
    subnormal_q.sum().backward()
    torch.testing.assert_close(subnormal_w.grad, torch.ones_like(subnormal_w), atol=0., rtol=0.)
    for rule in ('trust', 'clipped_ste'):
        try:
            prepare_context(subnormal_w, {'name': 'row_fixed_max', 'previous_scales': subnormal_scale, 'backward': rule}, 0, None)
        except ValueError as error:
            if 'underflow' not in str(error):
                raise
        else:
            raise AssertionError('Subnormal mask arithmetic failed to declare underflow risk')
    return {'fit': fit, 'rotation': rotation, 'grouping_labels': [initial.grouping, whole.grouping],
            'scales': {'zero_group_count': initial.metadata['zero_group_count'],
                       'sentinel': 1., 'fixed_previous_preserved': True, 'explicit_warm_start_preserved': True,
                       'dynamic_recomputed': True, 'stable_rms_cases': rms_cases, 'nonzero_underflow_rejected': True,
                       'fixed_subnormal_ste_bits_preserved': True, 'subnormal_masks_rejected': True},
            'surrogates': surrogate_results,
            'microbatch_backward': {'independent_backward_calls': 2, 'retained_graph': False,
                                    'gradient_max_abs_error': float((w.grad-(x1+x2)).abs().max())},
            'detached_context': {'tensor_count': len(context.numeric_tensors()), 'tensors_requiring_grad': 0,
                                 'numeric_cache_unchanged': True, 'output_coordinates': 'storage_Q(U)'},
            'supplied_realization': {'codes_owned': True, 'rng_unchanged': True, 'zero_groups_preserved': True, 'generation': 'reserved_M05'},
            'reference_label': 'project_S14_refit_full_row; upstream_release_parity_pending_M07'}

def check_quantizer(device: str, output: Path) -> ArtifactRef:
    """Persist measured device/oracle evidence, including blocked/negative results."""
    if device not in ('cpu', 'mps'):
        raise ValueError('Quantizer device must be cpu or mps')
    root = Path(__file__).resolve().parents[3]
    source_paths = ['docs/numerics/e2m1-encoding-reference.json', 'docs/numerics/ocp-source.json',
                    'docs/numerics/ocp-inspection-notes.txt', 'src/ralup/quantization/formats.py',
                    'src/ralup/quantization/oracle.py', 'src/ralup/quantization/checks.py',
                    'src/ralup/quantization/scales.py', 'src/ralup/quantization/rotation.py',
                    'src/ralup/quantization/surrogates.py', 'src/ralup/quantization/context.py',
                    'docs/numerics/quantization-context.md', 'docs/numerics/gaussian-e2m1-reference.json', 'requirements.lock']
    report = {'schema_version': 1, 'suite': 'quantizer', 'scope': 'F03 format and F04 project fit/rotation/context/surrogates only; F05 and full WP01 pending',
              'acceptance_gate': 'F03_F04_only; full_WP01_pending',
              'recorded_utc': datetime.now(timezone.utc).isoformat(), 'device': device,
              'fallback_environment': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset'),
              'environment': {'python': platform.python_version(), 'machine': platform.machine(),
                              'packages': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scipy')},
                              'mps_built': torch.backends.mps.is_built(), 'mps_available': torch.backends.mps.is_available()},
              'source': {'ocp_pdf_sha256': None, 'ocp_pdf_checksum_status': 'unresolved: raw official PDF download HTTP403',
                         'files': {path: sha256_file(root / path) for path in source_paths}},
              'tolerances': {'context_atol': CONTEXT_ATOL, 'context_rtol': CONTEXT_RTOL,
                             'gaussian_objective_atol': 2e-13, 'gaussian_a_atol': 2e-7, 'gaussian_integration_residual_max': 2e-12,
                             'exact_values_atol': 0., 'code_mismatches': 0, 'reconstruction_rtol': VALUE_RTOL,
                             'reconstruction_atol': VALUE_ATOL, 'exact_scale_boundary_distance_max_fp32_ulp': 1.,
                             'exact_scale_boundary_mismatch_rule': 'adjacent ordinals; FP32 division lands on exact midpoint'},
              'checks': {}, 'status': 'blocked', 'blockers': []}
    if device == 'mps' and report['fallback_environment'] != '0':
        report['blockers'].append('MPS gate requires PYTORCH_ENABLE_MPS_FALLBACK=0 before interpreter startup')
    if device == 'mps' and not torch.backends.mps.is_available():
        report['blockers'].append('MPS device unavailable in this process; CPU success is not MPS evidence')
    if not report['blockers']:
        try:
            report['checks'], passing = _run_checks(device)
            report['checks']['context'] = _run_context_checks(device)
            report['status'] = 'pass' if passing else 'negative'
            if not passing:
                report['blockers'].append('Measured quantizer discrepancy exceeds predeclared acceptance')
        except (AssertionError, RuntimeError, ValueError, TypeError) as error:
            report['status'] = 'negative'
            report['blockers'].append(f'{type(error).__name__}: {error}')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(report) + b'\n')
    return ArtifactRef(output.resolve(), sha256_file(output))
