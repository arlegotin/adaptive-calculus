"""Breaks caught: adaptive bounds, bias omission, epsilon, zero escape or sign flip."""
import importlib.util,math
import pytest
import torch
from ralup.optim.base import SchedulePoint

def madam_rule(config):
    assert importlib.util.find_spec('ralup.optim.madam') is not None,'M02 Madam adapter missing'
    from ralup.optim.madam import MadamUpdateRule
    return MadamUpdateRule(config)

def test_independent_scalar_equations_negative_zero_bound_bias_and_clipping():
    rule=madam_rule({});w=torch.tensor([-2.,0.,.5]);initial=w.clone();bound=3*math.sqrt(4.25/3);state={};v=[0.,0.,0.];expected=w.tolist()
    for step,g in enumerate([[1.,0.,-2.],[0.,0.,0.],[1000.,7.,-2000.],[-5.,0.,5.]]):
        for i,x in enumerate(g):
            v[i]=.999*v[i]+.001*x*x
            normalized=0. if v[i]==0. else x/math.sqrt(v[i]/(1-.999**(step+1)))
            normalized=max(-10.,min(10.,normalized));sign=0 if expected[i]==0 else math.copysign(1,expected[i]);expected[i]=max(-bound,min(bound,expected[i]*math.exp(-.5*sign*normalized)))
        p=rule.propose(w,torch.tensor(g),state,SchedulePoint(.5,0.,step/4));rule.commit(w,p);state=p.new_optimizer_state
        torch.testing.assert_close(w,torch.tensor(expected),atol=2e-6,rtol=2e-6)
        assert torch.equal(w[initial==0],torch.zeros_like(w[initial==0]))
        assert bool((w.abs()<=state['bound']).all()) and torch.equal(torch.sign(w),torch.sign(initial))
        torch.testing.assert_close(state['bound'],torch.tensor(bound),atol=3e-7,rtol=0)
        if step==0:fixed_bound=state['bound'].clone()
        assert torch.equal(state['bound'],fixed_bound)

def test_fixed_initial_bound_does_not_follow_updated_RMS():
    rule=madam_rule({'p_scale':1.});w=torch.tensor([-1.,0.,1.]);state={}
    for _ in range(4):
        p=rule.propose(w,torch.tensor([1.,1.,-1.]),state,SchedulePoint(2.,0.,0.));rule.commit(w,p);state=p.new_optimizer_state
        torch.testing.assert_close(w,torch.tensor([-math.sqrt(2/3),0.,math.sqrt(2/3)]),atol=1e-7,rtol=0)
        assert float(state['bound'])==pytest.approx(math.sqrt(2/3))

def test_zero_over_zero_and_hidden_additive_escape_are_detectable():
    rule=madam_rule({});w=torch.tensor([-1.,0.,1.]);p=rule.propose(w,torch.zeros_like(w),{},SchedulePoint(.01,0.,0.));rule.commit(w,p)
    assert torch.equal(w,torch.tensor([-1.,0.,1.]))
    escape=rule.propose(w,torch.ones_like(w),p.new_optimizer_state,SchedulePoint(.01,0.,0.));rule.commit(w,escape)
    assert w[1]==0 and (w-.01)[1]!=0

@pytest.mark.parametrize('config',[{'beta':1.},{'p_scale':0.},{'g_bound':-1.},{'eps':1e-8}])
def test_rejects_unsupported_madam_settings(config):
    with pytest.raises(ValueError):madam_rule(config)

def test_madam_has_no_undeclared_decay():
    with pytest.raises(ValueError):madam_rule({}).propose(torch.ones(2),torch.ones(2),{},SchedulePoint(.01,.1,0.))


def test_configured_normalized_gradient_clip_changes_magnitude_before_exponentiation():
    rule=madam_rule({'g_bound':.25});w=torch.tensor([-1.,0.,1.]);p=rule.propose(w,torch.tensor([100.,0.,100.]),{},SchedulePoint(.1,0.,0.));rule.commit(w,p)
    torch.testing.assert_close(w,torch.tensor([-math.exp(.025),0.,math.exp(-.025)]),atol=1e-7,rtol=1e-7)


def test_default_clip_ten_saturates_a_sparse_observation_after_zero_moments():
    # After 199 zero gradients, the next normalized gradient exceeds 10.
    rule=madam_rule({});w=torch.tensor([-1.,0.,1.])
    state={'step':199,'exp_avg_sq':torch.zeros(3),'bound':torch.tensor(3*math.sqrt(2/3))}
    assert math.sqrt((1-.999**200)/.001)>10
    p=rule.propose(w,torch.tensor([1.,0.,1.]),state,SchedulePoint(.001,0.,0.));rule.commit(w,p)
    torch.testing.assert_close(w,torch.tensor([-math.exp(.01),0.,math.exp(-.01)]),atol=1e-7,rtol=1e-7)
    assert torch.equal(state['exp_avg_sq'],torch.zeros(3)) and state['step']==199
