"""S17/S18 behavior, including controls that defeat tempting shortcuts."""
import os
import numpy as np
import pytest
import torch
from ralup.quantization.context import prepare_context
from ralup.quantization.formats import E2M1_THRESHOLDS, quantize_e2m1


def api():
    from ralup.policy.actions import multiplicative_proposal, assist_boundaries, compose_action
    return multiplicative_proposal, assist_boundaries, compose_action


@pytest.fixture(params=['cpu', 'mps'])
def device(request):
    if request.param == 'mps':
        if not torch.backends.mps.is_available():
            pytest.skip('Actual MPS unavailable; CPU is not MPS evidence')
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    return request.param


def context(u, scale=1.):
    w = u.reshape(1, -1)
    return prepare_context(w, {'name': 'row_fixed_max', 'previous_scales': torch.tensor([[scale]], device=u.device)}, 0, None)


def test_sign_zero_exponent_cap_and_no_post_normalization(device):
    mult, _, _ = api()
    u = torch.tensor([-2., -1., 0., 1., 2.], device=device)
    da = torch.tensor([10., .2, .3, .2, -10.], device=device)
    dm = mult(u, da)
    assert dm[0] > 0 and dm[1] > 0 and dm[3] > 0 and dm[4] < 0
    assert dm[2] == 0 and torch.equal(torch.sign(u+dm), torch.sign(u))
    assert float(((u+dm)[u != 0]/u[u != 0]).log().abs().max()) <= .100001
    assert dm.norm() < da.norm() / 10
    # Naive negative-weight multiplication moves against the desired direction.
    assert (u[1]*torch.expm1(da[1])) < 0 < dm[1]
    assert torch.equal(mult(u, torch.zeros_like(u)), torch.zeros_like(u))
    assert torch.equal(mult(torch.zeros_like(u), da), torch.zeros_like(u))


def test_identity_decay_separation_and_original_commit_permission(device):
    _, assist, compose = api()
    u = torch.tensor([1., -1., 0., 6.], device=device)
    da = torch.tensor([.04, -.04, .1, .2], device=device)
    ctx = context(u)
    assert torch.equal(assist(u, da, 1., ctx)[0], u+da)
    out, info = compose(u, da, 0., 1., 1., ctx)
    assert torch.equal(out, u+da)
    assert info['identity_action'] and info['authoritative_original_commit_bypass_permitted']
    decayed = u*.9
    assert torch.equal(compose(decayed, torch.zeros_like(da), .5, 2., .125, ctx)[0], decayed)
    # h applies only to the update: common decay has already entered U0.
    assert torch.equal(compose(decayed, da, 0., 2., 1., ctx)[0], decayed+2*da)
    assert compose(u, da, 1., 1., 1., ctx)[0][2] == 0
    assert compose(u, da, 0., 1., 1., ctx)[0][2] > 0


def test_scale_one_neighbor_and_harmful_unconditional_control(device):
    _, assist, _ = api()
    u = torch.tensor([1.], device=device)
    d = torch.tensor([.04], device=device)
    out, info = assist(u, d, .125, context(u))
    assert float(out.item()) == pytest.approx(1.2500390625)
    assert info['accepted_mask'].item() and info['accepted_count'] == 1
    assert quantize_e2m1(out, torch.tensor(1., device=device))[0].item() == 1.5
    assert (1.5-1.1)**2 > (1.-1.1)**2
    assert (1.5-1.8)**2 < (1.-1.8)**2
    tiny = torch.tensor([1e-4], device=device)
    assert torch.equal(assist(u, tiny, .125, context(u))[0], u+tiny)
    # Unconditional nudging violates the specified amplification bound.
    assert abs(1.2500001-1.) > (8+1/1024)*tiny.item()


@pytest.mark.parametrize('tau', [.125, .25, .5, .75, 1.])
def test_100000_seeded_coordinates_neighbors_and_bound(device, tau):
    _, assist, _ = api()
    rng = np.random.default_rng(20261004)
    u = torch.tensor(rng.uniform(-6.5, 6.5, 100000).astype(np.float32), device=device)
    d = torch.tensor(rng.uniform(-.7, .7, 100000).astype(np.float32), device=device)
    out, info = assist(u, d, tau, context(u))
    accept, reject = info['accepted_mask'], info['rejected_mask']
    assert not bool((accept & reject).any())
    assert torch.equal(out[~accept], (u+d)[~accept])
    assert info['bound_tolerance_exceeded_count'] == 0
    # Independent float64 audit of accepted physical displacement.
    actual = out.cpu().double()-u.cpu().double()
    bound = (1/tau+1/1024)*d.cpu().double().abs()
    assert bool((actual.abs()[accept.cpu()] <= bound[accept.cpu()]*(1+info['bound_rtol'])).all())
    _, codes = quantize_e2m1(out, torch.tensor(1., device=device))
    if tau != 1.:
        assert torch.equal(codes[accept], info['neighbor_codes'][accept])
    if tau == 1.:
        assert torch.equal(out, u+d) and info['accepted_count'] == 0


@pytest.mark.parametrize('tau', [.125, .25, .5, .75, 1.])
def test_all_boundary_ties_saturation_and_existing_crossings(device, tau):
    _, assist, _ = api()
    u = torch.tensor(E2M1_THRESHOLDS, device=device).repeat(2)
    d = torch.cat([torch.full((14,), -.04, device=device), torch.full((14,), .04, device=device)])
    out, info = assist(u, d, tau, context(u))
    assert torch.equal(out, u+d)
    assert info['accepted_count'] == 0
    u = torch.tensor([-7., -6., 6., 7.], device=device)
    d = torch.tensor([.1, -.1, .1, -.1], device=device)
    assert torch.equal(assist(u, d, tau, context(u))[0], u+d)


def test_unrepresentable_nudge_is_rejected_without_margin_growth(device):
    _, assist, _ = api()
    # Smallest scale: midpoint and gap nudges round to zero FP32 units.
    scale = torch.tensor([[12]], dtype=torch.int32, device=device).view(torch.float32)
    u = torch.tensor([12], dtype=torch.int32, device=device).view(torch.float32)
    d = torch.tensor([1], dtype=torch.int32, device=device).view(torch.float32)
    ctx = prepare_context(u.reshape(1, 1), {'name': 'row_fixed_max', 'previous_scales': scale}, 0, None)
    out, info = assist(u, d, .125, ctx)
    assert info['eligible_mask'].item() and info['rejected_mask'].item()
    assert info['unrepresentable_nudge_count'] == 1 and info['accepted_count'] == 0
    assert out.view(torch.int32).cpu().tolist() == [13]


def test_assistance_uses_cached_dynamic_scales_and_homogeneity(device):
    _, assist, _ = api()
    u = torch.tensor([[1., 6.]], device=device)
    ctx = prepare_context(u, {'name': 'row_dynamic_max'}, 0, None)
    d = torch.tensor([[.04, 6.]], device=device)
    out, info = assist(u, d, .125, ctx)
    assert out[0, 0].item() == pytest.approx(1.2500390625)
    assert torch.equal(ctx.scales, torch.ones_like(ctx.scales))
    refreshed = prepare_context(out, {'name': 'row_dynamic_max'}, 1, None)
    assert refreshed.scales.item() == 2.
    out2, info2 = assist(2*u, 2*d, .125, context(2*u, 2.))
    assert torch.equal(out2, 2*out) and torch.equal(info['accepted_mask'], info2['accepted_mask'])


def test_invalid_action_inputs():
    mult, assist, compose = api()
    u, d = torch.tensor([1.]), torch.tensor([.04])
    for tau in (0., -1., 1.1, float('nan')):
        with pytest.raises(ValueError):
            assist(u, d, tau, context(u))
    with pytest.raises(TypeError):
        mult(u.double(), d.double())
    with pytest.raises(ValueError):
        compose(u, d, -1., 1., 1., context(u))
