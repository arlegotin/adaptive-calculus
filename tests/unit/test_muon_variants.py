"""Independent scalar equations; geometry and momentum controls are load bearing."""
import importlib.util
import math

import pytest
import torch

from ralup.optim.base import SchedulePoint


def scalar_polynomial(matrix):
    """Literal CPU float64 five-step equation, scalar sums (no production helper)."""
    x = matrix.tolist()
    tall = len(x) > len(x[0])
    if tall: x = [list(row) for row in zip(*x)]
    norm = math.sqrt(sum(v*v for row in x for v in row)) + 1e-7
    x = [[v/norm for v in row] for row in x]
    def mm(a, b):
        return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
    for _ in range(5):
        a = mm(x, [list(row) for row in zip(*x)])
        ax = mm(a, x)
        aax = mm(a, ax)
        x = [[3.4445*x[i][j]-4.775*ax[i][j]+2.0315*aax[i][j] for j in range(len(x[0]))] for i in range(len(x))]
    if tall: x = [list(row) for row in zip(*x)]
    return torch.tensor(x, dtype=torch.float64)


def polynomial_reference(x):
    tall = x.shape[0] > x.shape[1]
    x = x.T if tall else x
    x = x / (x.norm() + 1e-7)
    for _ in range(5):
        a = x @ x.T
        x = 3.4445*x - 4.775*(a@x) + 2.0315*(a@a@x)
    return x.T if tall else x


def normalize_reference(x, order):
    result = x.clone()
    for stage in (() if order == 'none' else order.split('_')):
        axis = -1 if stage == 'row' else -2
        result = result / torch.sqrt(result.square().sum(axis, keepdim=True) + 1e-8)
    return result


@pytest.mark.parametrize('shape', [(2,2),(3,2),(2,3),(1,4),(4,1)])
@pytest.mark.parametrize('zero', [False,True])
def test_five_step_matches_independent_float64_polynomial(shape, zero):
    assert importlib.util.find_spec('ralup.optim.muon') is not None, 'M01 five-step primitive missing'
    from ralup.optim.muon import orthogonalize_five_step
    x = torch.zeros(shape, dtype=torch.float64) if zero else torch.arange(1, math.prod(shape)+1, dtype=torch.float64).reshape(shape)/7
    actual = orthogonalize_five_step(x)
    torch.testing.assert_close(actual, scalar_polynomial(x), atol=2e-13, rtol=2e-13)
    assert actual.data_ptr() != x.data_ptr()


@pytest.mark.parametrize('shape', [(2,2),(3,2),(2,3)])
@pytest.mark.parametrize('order', ['none','row','col','row_col','col_row'])
def test_distinct_momentum_and_original_rectangular_displacements(shape, order):
    from ralup.optim.muon import MuonUpdateRule
    from ralup.optim.muon_plus import MuonPlusUpdateRule
    w = torch.nn.Parameter(torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)/17)
    initial = w.detach().clone()
    g = torch.arange(1,math.prod(shape)+1,dtype=torch.float32).reshape(shape)/9
    old = {'momentum':torch.full_like(w, .2)}
    point = SchedulePoint(.003,.1,0.)
    muon = MuonUpdateRule({})
    plus = MuonPlusUpdateRule({'normalization':order})
    for rule,expected_m,polar_input,multiplier in [
        (muon, .95*old['momentum']+g, g+.95*(.95*old['momentum']+g), math.sqrt(max(1,shape[0]/shape[1]))),
        (plus, .95*old['momentum']+g*(1-.95), .95*old['momentum']+g*(1-.95), math.sqrt(shape[0]/shape[1]))]:
        proposal = rule.propose(w,g,old,point)
        polar = polynomial_reference(polar_input)
        expected = normalize_reference(polar,order) if rule is plus else polar
        torch.testing.assert_close(proposal.gradient_displacement, -.003*multiplier*expected, atol=3e-8,rtol=2e-5)
        torch.testing.assert_close(proposal.new_optimizer_state['momentum'],expected_m)
        torch.testing.assert_close(proposal.reference_after_decay,initial*(1-.003*.1))
        assert torch.equal(w, initial) and torch.equal(old['momentum'],torch.full_like(w,.2))
        identity = id(w)
        rule.commit(w,proposal)
        assert id(w) == identity
        torch.testing.assert_close(w,proposal.reference_after_decay+proposal.gradient_displacement,atol=0,rtol=0)
        with torch.no_grad(): w.copy_(initial)


def test_normalization_toggle_uses_identical_ema_separate_from_nesterov():
    from ralup.optim.muon import MuonUpdateRule
    from ralup.optim.muon_plus import MuonPlusUpdateRule
    w = torch.zeros(2,3); g = torch.tensor([[1.,2.,0.],[0.,4.,3.]])
    point = SchedulePoint(.01,0.,0.)
    for rule in (MuonPlusUpdateRule({'normalization':'none'}),MuonPlusUpdateRule({'normalization':'row_col'})):
        p = rule.propose(w,g,{},point)
        assert torch.equal(p.new_optimizer_state['momentum'],g*.05)
    p = MuonUpdateRule({'nesterov':False}).propose(w,g,{},point)
    assert torch.equal(p.new_optimizer_state['momentum'],g)
    assert not torch.equal(p.gradient_displacement,MuonUpdateRule({}).propose(w,g,{},point).gradient_displacement)


@pytest.mark.parametrize('order', ['row','col','row_col','col_row','none'])
def test_zero_momentum_and_stabilizer(order):
    from ralup.optim.muon_plus import MuonPlusUpdateRule, normalize_polar
    tiny = torch.tensor([[1e-6,0.],[0.,0.]])
    torch.testing.assert_close(normalize_polar(tiny,order),normalize_reference(tiny,order),atol=0,rtol=0)
    p = MuonPlusUpdateRule({'normalization':order}).propose(torch.zeros(3,2),torch.zeros(3,2),{},SchedulePoint(.1,0.,0.))
    assert torch.equal(p.gradient_displacement,torch.zeros(3,2))


@pytest.mark.parametrize('config', [{'momentum':1.},{'nesterov':'yes'},{'unknown':True}])
def test_muon_configuration_rejects_invalid_or_unknown(config):
    from ralup.optim.muon import MuonUpdateRule
    with pytest.raises(ValueError): MuonUpdateRule(config)


def test_invalid_state_or_vector_rejected_and_configuration_owned():
    from ralup.optim.muon import MuonUpdateRule
    config = {'momentum':.9}; rule = MuonUpdateRule(config); config['momentum']=.1
    assert rule.configuration['momentum'] == .9
    for w,g,state in [(torch.ones(2),torch.ones(2),{}),(torch.ones(2,3),torch.ones(2,3),{'momentum':torch.zeros(3,2)}),
                      (torch.ones(2,3),torch.ones(2,3),{'extra':torch.zeros(2,3)})]:
        with pytest.raises(ValueError): rule.propose(w,g,state,SchedulePoint(.1,0.,0.))


def test_pinned_cpu64_literal_polynomial():
    from ralup.optim.muon import orthogonalize_five_step
    literal = torch.tensor([[-.6806609899651347,.825539610365519],[.7412954177537816,.25944002690644696]],dtype=torch.float64)
    actual = orthogonalize_five_step(torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64))
    torch.testing.assert_close(actual,literal,atol=2e-13,rtol=2e-13)
