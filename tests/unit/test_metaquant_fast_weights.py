"""Detect collapsed factors, cancelled meta derivatives and wrong calibration."""
import importlib
import pytest
import torch


def mq():
    assert importlib.util.find_spec('ralup.baselines.metaquant') is not None, 'M10 mechanism missing'
    return importlib.import_module('ralup.baselines.metaquant')


def test_factor_product_preserves_all_factor_gradients_against_unfused_layers():
    m=mq()
    with torch.random.fork_rng():
        torch.manual_seed(1010);net=m.FactorizedMetaNetwork().double()
    x=torch.tensor([[-.8],[.3],[.9]],dtype=torch.float64)
    upstream=torch.tensor([[1.],[2.],[-3.]],dtype=torch.float64)
    actual=net(x);reference=x
    factors=[p.detach().clone().requires_grad_() for p in net.parameters()]
    for p in factors:reference=torch.nn.functional.linear(reference,p)
    torch.testing.assert_close(actual,reference,atol=1e-14,rtol=1e-12)
    actual_grads=torch.autograd.grad((actual*upstream).sum(),tuple(net.parameters()))
    expected=torch.autograd.grad((reference*upstream).sum(),factors)
    assert sum(p.numel() for p in net.parameters())==10200
    for a,e in zip(actual_grads,expected):torch.testing.assert_close(a,e,atol=1e-14,rtol=1e-12)


def test_fast_value_and_independent_theta_derivative_and_simplification_negative_control():
    m=mq();theta=torch.tensor(3.,requires_grad=True)
    u=torch.tensor([[.3,-.5]],requires_grad=True);g=torch.tensor([[2.,-1.]])
    r=torch.tensor([[.1,.2]]);correction=theta*u.detach()*g
    fast=m.make_fast_weights(u,correction,r,.2)
    torch.testing.assert_close(fast,u-.2*r)
    actual=torch.autograd.grad(fast.square().sum(),theta)[0]
    expected=(-.4*(u.detach()-.2*r)*u.detach()*g).sum()
    torch.testing.assert_close(actual,expected)
    wrong=u-.2*r
    assert torch.autograd.grad(wrong.square().sum(),theta,allow_unused=True)[0] is None


def test_dorefa_forward_and_calibration_independently_match_published_equations():
    m=mq();w=torch.tensor([[-.8,.3,1.2]],requires_grad=True)
    g=torch.tensor([[1.,-2.,.5]])
    net=m.FactorizedMetaNetwork((1,2,2,1))
    with torch.no_grad():
        for p in net.parameters():p.fill_(.5)
    t=w.detach().tanh();maximum=t.abs().max();a=t/(2*maximum)+.5
    expected=2*(a*7).round()/7-1
    torch.testing.assert_close(m.uniform_dorefa_view(w,3).detach(),expected)
    correction=m.dorefa_correction(net,w,g)
    # Three .5 maps with two hidden coordinates compose to coefficient .5.
    target=.5*a*g*(1-t.square())/maximum
    torch.testing.assert_close(correction,target)
    m.uniform_dorefa_view(w,3).sum().backward()
    torch.testing.assert_close(w.grad,(1-t.square())/maximum)


@pytest.mark.parametrize('bad_lr',[-1.,float('nan'),float('inf')])
def test_fast_rejects_invalid_learning_rate(bad_lr):
    m=mq();u=torch.ones(1,2)
    with pytest.raises(ValueError):m.make_fast_weights(u,u,u,bad_lr)


def test_registry_dorefa_fixture_cannot_mislabel_fp4_or_h04_execution():
    m=mq()
    from ralup.baselines.registry import metaquant_record
    from ralup.optim.base import ParameterGroup
    record=metaquant_record('metaquant_uniform_fixture',(ParameterGroup('internal_projection',('w',)),),m.FIXTURE_CONFIG)
    assert record.update_rule_settings['quantizer']=='dorefa_uniform_3bit'
    assert 'functional_H04_target_backward' not in record.required_operations
    assert 'row_fixed_max' not in record.update_rule_settings.values()
