"""A cached autograd graph or wrong rotation inverse breaks microbatch gradients."""
import os

import pytest
import torch


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('rotation', ['identity', 'full_hadamard'])
def test_two_independent_backward_graphs_share_detached_numeric_context(device, rotation):
    from ralup.quantization.context import prepare_context, quantized_view
    if device == 'mps':
        if not torch.backends.mps.is_available():
            pytest.skip('Actual MPS unavailable')
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    w = torch.tensor([[1., 2., -3., 4.], [0., 1., 3., -2.]], device=device, requires_grad=True)
    context = prepare_context(w, {'name': 'row_dynamic_max', 'rotation': rotation}, 7, None)
    x1 = torch.tensor([[1., 2., 0., -1.]], device=device)
    x2 = torch.tensor([[-1., 3., 2., 1.]], device=device)
    # H04 applies X@R; quantized_view explicitly returns storage Q(W@R).
    h = torch.tensor([[1., 1., 1., 1.], [1., -1., 1., -1.],
                      [1., 1., -1., -1.], [1., -1., -1., 1.]], device=device) / 2
    r = torch.eye(4, device=device) if rotation == 'identity' else h
    numeric_before = [t.clone() for t in context.numeric_tensors()]
    first = quantized_view(w, context)
    ((x1 @ r) @ first.T).sum().backward()
    second = quantized_view(w, context)
    ((x2 @ r) @ second.T).sum().backward()
    assert first.grad_fn is not second.grad_fn
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(w.grad, (x1+x2).expand_as(w), atol=2e-6, rtol=2e-6)
    for before, after in zip(numeric_before, context.numeric_tensors()):
        torch.testing.assert_close(before, after)
        assert not after.requires_grad and after.grad_fn is None


@pytest.mark.parametrize('rotation', ['identity', 'full_hadamard'])
def test_context_freezes_codes_until_explicit_next_logical_step(rotation):
    from ralup.quantization.context import prepare_context, quantized_view
    w = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
    c = prepare_context(w, {'name': 'row_dynamic_max', 'rotation': rotation}, 0, None)
    before = quantized_view(w, c).detach().clone()
    with torch.no_grad():
        w.mul_(3)
    torch.testing.assert_close(quantized_view(w, c), before)
    next_c = prepare_context(w, {'name': 'row_dynamic_max', 'rotation': rotation}, 1, None)
    torch.testing.assert_close(next_c.scales, c.scales*3)
    torch.testing.assert_close(quantized_view(w, next_c), before*3)
