"""Catch coordinate, scale lifecycle, sentinel, and surrogate rule regressions."""
import os

import pytest
import torch


def api():
    from ralup.quantization.context import prepare_context, quantized_view
    from ralup.quantization.rotation import hadamard, to_storage, from_storage
    from ralup.quantization.scales import compute_scales
    return prepare_context, quantized_view, hadamard, to_storage, from_storage, compute_scales


@pytest.fixture(params=['cpu', 'mps'])
def device(request):
    if request.param == 'mps':
        if not torch.backends.mps.is_available():
            pytest.skip('Actual MPS unavailable; CPU does not establish MPS parity')
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    return request.param


def test_full_hadamard_involution_and_inverse_gradient(device):
    _, _, hadamard, to_storage, from_storage, _ = api()
    x = torch.arange(24, dtype=torch.float32, device=device).reshape(3, 8).requires_grad_()
    upstream = torch.tensor([[1., 2., -1., 3., 0., 5., 2., -4.]], device=device).expand_as(x)
    torch.testing.assert_close(hadamard(hadamard(x)), x, atol=8e-6, rtol=2e-6)
    (hadamard(x) * upstream).sum().backward()
    # Direct dense orthonormal Sylvester matrix, independent of butterfly.
    h = torch.tensor([[1., 1.], [1., -1.]], device=device)
    h = torch.kron(torch.kron(h, h), h) / (8 ** .5)
    torch.testing.assert_close(x.grad, upstream @ h, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(to_storage(x, 'full_hadamard'), x @ h, atol=8e-6, rtol=2e-6)
    torch.testing.assert_close(from_storage(to_storage(x, 'full_hadamard'), 'full_hadamard'), x, atol=8e-6, rtol=2e-6)
    torch.testing.assert_close(to_storage(x, 'identity'), x)
    with pytest.raises(ValueError):
        to_storage(x, 'release_block128')
    with pytest.raises(ValueError):
        hadamard(x[:, :3])


def test_zero_sentinel_and_fixed_scale_ownership(device):
    prepare, _, _, _, _, scales = api()
    w = torch.tensor([[0., 0., 0., 0.], [6., -3., 0., 1.]], device=device, requires_grad=True)
    torch.testing.assert_close(scales(w, 'row_fixed_max', None), torch.tensor([[1.], [1.]], device=device))
    initial = prepare(w, {'name': 'row_fixed_max'}, 0, None)
    assert initial.zero_groups.cpu().tolist() == [[True], [False]]
    assert initial.metadata['zero_group_count'] == 1
    old = initial.scales.clone()
    current = w.detach() * 12
    recipe = {'name': 'row_fixed_max', 'previous_scales': initial.scales}
    fixed = prepare(current, recipe, 1, None)
    dynamic = prepare(current, {'name': 'row_dynamic_max'}, 1, None)
    torch.testing.assert_close(fixed.scales, old)
    torch.testing.assert_close(dynamic.scales, torch.tensor([[1.], [12.]], device=device))
    recipe['previous_scales'].fill_(3.)
    torch.testing.assert_close(fixed.scales, old)
    assert fixed.metadata['scale_source'] == 'previous_fixed_scales'
    assert all(not t.requires_grad and t.grad_fn is None for t in fixed.numeric_tensors())
    assert fixed.storage_values.dtype == torch.float32 and fixed.codes.dtype == torch.uint8


def test_fixed_warm_start_and_tensor_grouping(device):
    prepare, _, _, _, _, scales = api()
    warm = torch.tensor([[6., 0., 0., 0.], [12., 0., 0., 0.]], device=device)
    w = warm * 5
    context = prepare(w, {'name': 'row_fixed_max', 'warm_start': warm}, 9, None)
    torch.testing.assert_close(context.scales, torch.tensor([[1.], [2.]], device=device))
    torch.testing.assert_close(scales(w, 'tensor_dynamic_max', None), torch.tensor([[10.]], device=device))
    assert prepare(w, {'name': 'tensor_dynamic_max'}, 9, None).grouping == 'whole_tensor'
    with pytest.raises(ValueError):
        scales(w, 'row_dynamic_max', torch.ones((2, 1), device=device))
    with pytest.raises(ValueError):
        scales(w, 'row_fixed_max', torch.ones((1, 1), device=device))


def test_stable_rms_extremes_and_nonzero_underflow(device):
    prepare, _, _, _, _, scales = api()
    huge = torch.tensor([[1e30, -1e30, 1e30, -1e30]], device=device)
    context = prepare(huge, {'name': 'quest_fp4_row_reference'}, 0, None)
    torch.testing.assert_close(context.scales, torch.tensor([[4.8707917856e29]], device=device))
    tiny = torch.tensor([[1e-30, -1e-30, 1e-30, -1e-30]], device=device)
    context = prepare(tiny, {'name': 'quest_fp4_row_reference'}, 0, None)
    torch.testing.assert_close(context.scales, torch.tensor([[4.8707917856e-31]], device=device), atol=0., rtol=2e-6)
    # Smallest FP32 subnormal must fail explicitly, never become an all-zero row.
    underflow = torch.tensor([[1, 0, 0, 0]], dtype=torch.int32, device=device).view(torch.float32)
    with pytest.raises(ValueError, match='underflow'):
        scales(underflow, 'row_dynamic_max', None)
    for bad in (float('nan'), float('inf')):
        with pytest.raises(ValueError):
            scales(torch.tensor([[bad]], device=device), 'row_dynamic_max', None)


@pytest.mark.parametrize('rule,expected', [('ste', [1., 1., 1., 1.]), ('trust', [1., 0., 1., 1.]), ('clipped_ste', [1., 0., 1., 0.])])
def test_surrogates_have_named_distinct_gradients(device, rule, expected):
    prepare, view, _, _, _, _ = api()
    # Caller-owned fixed unit scale: physical trust half-gap is s=1.
    w = torch.tensor([[.1, 7.1, -1., 6.2]], device=device, requires_grad=True)
    c = prepare(w, {'name': 'row_fixed_max', 'previous_scales': torch.ones((1, 1), device=device), 'backward': rule}, 2, None)
    out = view(w, c)
    torch.testing.assert_close(out, torch.tensor([[0., 6., -1., 6.]], device=device))
    out.sum().backward()
    torch.testing.assert_close(w.grad, torch.tensor([expected], device=device))
    tiny_upstream = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=device).view(torch.float32)
    tiny_gradient, = torch.autograd.grad(view(w, c), w, tiny_upstream)
    expected_bits = {'ste': [1, 2, 3, 4], 'trust': [1, 0, 3, 4], 'clipped_ste': [1, 0, 3, 0]}
    assert tiny_gradient.view(torch.int32).cpu().tolist() == [expected_bits[rule]]


def test_rng_untouched_supplied_codes_reused_and_inputs_unchanged(device):
    prepare, view, _, _, _, _ = api()
    w = torch.tensor([[1., 2., 3., 6.]], device=device, requires_grad=True)
    before = w.detach().clone()
    rng = torch.Generator().manual_seed(123)
    state = rng.get_state().clone()
    codes = torch.tensor([[7, 8, 9, 14]], dtype=torch.uint8, device=device)
    c = prepare(w, {'name': 'row_dynamic_max', 'codes': codes}, 3, rng)
    codes.fill_(0)
    torch.testing.assert_close(view(w, c), torch.tensor([[0., .5, 1., 6.]], device=device))
    torch.testing.assert_close(w, before)
    assert torch.equal(rng.get_state(), state)
    assert w.grad is None and c.step == 3
    with pytest.raises(ValueError):
        prepare(w, {'name': 'row_dynamic_max', 'codes': torch.tensor([[15]*4], dtype=torch.uint8, device=device)}, 3, rng)


def test_fixed_scale_initialization_flags_owned_independently_of_current_rows():
    prepare, _, _, _, _, _ = api()
    warm = torch.tensor([[0., 0.], [6., 0.]])
    initial = prepare(warm, {'name': 'row_fixed_max'}, 0, None)
    current = torch.tensor([[2., 0.], [0., 0.]])
    refreshed = prepare(current, {'name': 'row_fixed_max', 'previous_scales': initial.scales,
                                  'previous_zero_groups': initial.scale_zero_groups}, 1, None)
    assert refreshed.zero_groups.tolist() == [[False], [True]]
    assert refreshed.scale_zero_groups.tolist() == [[True], [False]]
    assert refreshed.metadata['scale_initialization_zero_group_count'] == 1
    unknown = prepare(current, {'name': 'row_fixed_max', 'previous_scales': initial.scales}, 1, None)
    assert unknown.scale_zero_groups is None
    assert unknown.metadata['scale_initialization_zero_group_count'] is None
    with pytest.raises(ValueError):
        prepare(current, {'name': 'row_dynamic_max', 'previous_zero_groups': initial.scale_zero_groups}, 1, None)


@pytest.mark.parametrize('rule, expected', [('trust', [1., 0., 1., 1.]), ('clipped_ste', [1., 0., 1., 0.])])
def test_subnormal_mask_arithmetic_preserves_physical_thresholds(device, rule, expected):
    prepare, view, _, _, _, _ = api()
    # Unit s has three minimum-subnormal units; errors are [0,4,0,1].
    scale = torch.tensor([[3]], dtype=torch.int32, device=device).view(torch.float32)
    w = torch.tensor([[0, 22, -2147483645, 19]], dtype=torch.int32, device=device).view(torch.float32).requires_grad_()
    c = prepare(w, {'name': 'row_fixed_max', 'previous_scales': scale, 'backward': rule}, 0, None)
    assert c.storage_values.view(torch.int32).cpu().tolist() == [[0, 18, -2147483645, 18]]
    view(w, c).sum().backward()
    torch.testing.assert_close(w.grad, torch.tensor([expected], device=device))
    tiny_upstream = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=device).view(torch.float32)
    tiny_gradient, = torch.autograd.grad(view(w, c), w, tiny_upstream)
    assert tiny_gradient.view(torch.int32).cpu().tolist() == [[1, 0, 3, 4 if rule == "trust" else 0]]


def test_fixed_subnormal_scale_plain_ste_keeps_f03_integer_bits(device):
    prepare, view, _, _, _, _ = api()
    scale = torch.tensor([[3]], dtype=torch.int32, device=device).view(torch.float32)
    w = torch.tensor([[7, -2147483641]], dtype=torch.int32, device=device).view(torch.float32).requires_grad_()
    c = prepare(w, {'name': 'row_fixed_max', 'previous_scales': scale, 'backward': 'ste'}, 0, None)
    q = view(w, c)
    assert q.view(torch.int32).cpu().tolist() == [[6, -2147483642]]
    q.sum().backward()
    torch.testing.assert_close(w.grad, torch.ones_like(w))


def test_full_rotation_is_not_release_block128(device):
    _, _, hadamard, _, _, _ = api()
    impulse = torch.zeros((2, 256), device=device)
    impulse[:, 0] = 1.
    # A block-128 rotation leaves the last 128 coordinates zero.
    torch.testing.assert_close(hadamard(impulse), torch.full_like(impulse, 1/16), atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(hadamard(hadamard(impulse)), impulse, atol=2e-6, rtol=2e-6)


def test_trust_gradient_masks_storage_then_inverts_rotation(device):
    prepare, view, _, _, _, _ = api()
    # Dense H4 maps this W to [.1,7.1,-1,6.2]. Frozen trust mask [1,0,1,1].
    w = torch.tensor([[6.2, -7.1, 1., .1]], device=device, requires_grad=True)
    c = prepare(w, {'name': 'row_fixed_max', 'rotation': 'full_hadamard', 'backward': 'trust',
                     'previous_scales': torch.ones((1, 1), device=device)}, 0, None)
    (view(w, c)*torch.tensor([[1., 2., 3., 4.]], device=device)).sum().backward()
    # [1,0,3,4]@H4 = [4,0,-3,1], computed independently.
    torch.testing.assert_close(w.grad, torch.tensor([[4., 0., -3., 1.]], device=device), atol=2e-6, rtol=2e-6)


def test_supplied_realization_cannot_turn_an_all_zero_group_nonzero(device):
    prepare, _, _, _, _, _ = api()
    w = torch.zeros((2, 4), device=device)
    codes = torch.full_like(w, 7, dtype=torch.uint8)
    codes[0, 0] = 8
    with pytest.raises(ValueError, match='zero group'):
        prepare(w, {'name': 'row_dynamic_max', 'codes': codes}, 0, None)



def test_supplied_finite_code_bypasses_unused_nearest_overflow(device):
    prepare, view, _, _, _, _ = api()
    w = torch.tensor([[3.3e38]], device=device, requires_grad=True)
    scale = torch.tensor([[6e37]], device=device)
    c = prepare(w, {'name': 'row_fixed_max', 'previous_scales': scale,
                     'codes': torch.tensor([[13]], dtype=torch.uint8, device=device)}, 0, None)
    torch.testing.assert_close(view(w, c), torch.tensor([[2.4e38]], device=device))
    view(w, c).sum().backward()
    torch.testing.assert_close(w.grad, torch.ones_like(w))
    with pytest.raises(ValueError, match='overflow'):
        prepare(w, {'name': 'row_fixed_max', 'previous_scales': scale}, 0, None)


@pytest.mark.parametrize('name', ['row_fixed_max', 'row_dynamic_max', 'tensor_dynamic_max', 'quest_fp4_row_reference'])
def test_automatic_scales_preserve_representable_subnormals(device, name):
    prepare, _, _, _, _, _ = api()
    w = torch.tensor([[12, 0]], dtype=torch.int32, device=device).view(torch.float32)
    c = prepare(w, {'name': name}, 0, None)
    # Max recipes: (12 units)/6=2 units. RMS fit: a/6 * sqrt(72)=4.133... units.
    expected = 4 if name.startswith('quest') else 2
    assert c.scales.view(torch.int32).cpu().tolist() == [[expected]]
    assert not bool(c.zero_groups.any())
    assert c.storage_values.view(torch.int32).cpu().tolist() == [[12, 0]]


@pytest.mark.parametrize('bits', [12, 0x800000])
def test_hadamard_preserves_finite_subnormal_operands_results_and_backward(device, bits):
    import numpy as np
    _, _, h, _, _, _ = api()
    x = torch.tensor([[bits, 0]], dtype=torch.int32, device=device).view(torch.float32).requires_grad_()
    q = h(x)
    expected = (np.array([bits], dtype=np.int32).view(np.float32)/np.float32(2**.5)).view(np.int32)[0]
    assert q.view(torch.int32).cpu().tolist() == [[int(expected), int(expected)]]
    # Incoming subnormal gradient also traverses the bit-safe inverse rotation.
    upstream = torch.tensor([[bits, 0]], dtype=torch.int32, device=device).view(torch.float32)
    gradient, = torch.autograd.grad(q, x, upstream)
    assert gradient.view(torch.int32).cpu().tolist() == [[int(expected), int(expected)]]


def test_mixed_tiny_max_scale_cohort_matches_independent_fp32_division(device):
    import numpy as np
    prepare, _, _, _, _, _ = api()
    rng = np.random.default_rng(20261004)
    raw = rng.integers(12, 0x2000000, size=(256, 8), dtype=np.int32)
    raw[:, 1] = 0
    x = raw.view(np.float32)
    expected = (x.max(axis=-1, keepdims=True)/np.float32(6)).view(np.int32)
    c = prepare(torch.tensor(raw, dtype=torch.int32, device=device).view(torch.float32), {'name': 'row_dynamic_max'}, 0, None)
    np.testing.assert_array_equal(c.scales.view(torch.int32).cpu().numpy(), expected)
    assert not bool(c.zero_groups.any())


def test_mixed_tiny_hadamard_matches_independent_gradual_underflow_butterfly(device):
    import numpy as np
    _, _, hadamard, _, _, _ = api()
    rng = np.random.default_rng(20261004)
    raw = rng.integers(1, 0x1000000, size=(256, 8), dtype=np.int32)
    raw[:, 1::2] |= np.int32(-2147483648)
    reference = raw.view(np.float32).copy()
    for stride in (1, 2, 4):
        groups = reference.reshape(256, -1, 2, stride)
        left, right = groups[:, :, 0, :]/np.float32(2**.5), groups[:, :, 1, :]/np.float32(2**.5)
        reference = np.stack((left+right, left-right), axis=-2).reshape(256, 8)
    x = torch.tensor(raw, device=device).view(torch.float32)
    actual = hadamard(x)
    np.testing.assert_array_equal(actual.view(torch.int32).cpu().numpy(), reference.view(np.int32))


def test_hadamard_high_range_finite_output_avoids_unnormalized_sum_overflow(device):
    _, _, hadamard, _, _, _ = api()
    x = torch.tensor([[2e38, 2e38]], device=device)
    torch.testing.assert_close(hadamard(x), torch.tensor([[2.8284271e38, 0.]], device=device), atol=0., rtol=2e-6)


def test_supplied_codes_still_reject_nonfinite_actual_weights_with_valid_warm_start(device):
    prepare, _, _, _, _, _ = api()
    with pytest.raises(ValueError, match='nonfinite'):
        prepare(torch.tensor([[float('nan')]], device=device),
                {'name': 'row_fixed_max', 'warm_start': torch.tensor([[6.]], device=device),
                 'codes': torch.tensor([[13]], dtype=torch.uint8, device=device)}, 0, None)


def test_gaussian_rms_keeps_subnormal_members_when_group_max_is_normal(device):
    import numpy as np
    prepare, _, _, _, _, _ = api()
    w = torch.tensor([[0x800000, 0x400000]], dtype=torch.int32, device=device).view(torch.float32)
    c = prepare(w, {'name': 'quest_fp4_row_reference'}, 0, None)
    factor = np.float32(np.sqrt(np.float32(.625))*np.float32(.4870791785622801))
    expected = (np.array([0x800000], dtype=np.int32).view(np.float32)*factor).view(np.int32)[0]
    assert c.scales.view(torch.int32).cpu().tolist() == [[int(expected)]]


@pytest.mark.parametrize('name, shape', [('quest_fp4_row_reference', (1, 256)), ('quest_fp4_tensor_reference', (2, 128))])
def test_sparse_normal_max_gaussian_scale_preserves_subnormal_product(device, name, shape):
    import numpy as np
    prepare, _, _, _, _, _ = api()
    raw = torch.zeros(shape, dtype=torch.int32, device=device)
    raw[0, 0] = 0x2000000
    c = prepare(raw.view(torch.float32), {'name': name}, 0, None)
    # One nonzero among 256 values: RMS = maximum/16; factor a/6 is FP32.
    maximum = np.array([0x2000000], dtype=np.int32).view(np.float32)[0]
    expected = np.array([maximum*np.float32(np.float32(.4870791785622801)/16)], dtype=np.float32).view(np.int32)[0]
    assert int(expected) == 2042958
    assert c.scales.view(torch.int32).cpu().tolist() == [[int(expected)]]
    assert not bool(c.zero_groups.any())
