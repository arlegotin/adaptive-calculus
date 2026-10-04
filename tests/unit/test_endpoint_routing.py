"""Independent literal equations catch unmasked powers, zero and gap mistakes."""
import importlib
import pytest
import torch

def modules():
    assert importlib.util.find_spec('ralup.baselines.qar') is not None,'M05 QAR missing'
    assert importlib.util.find_spec('ralup.baselines.rounding') is not None,'M05 rounding missing'
    return importlib.import_module('ralup.baselines.qar'),importlib.import_module('ralup.baselines.rounding')

def test_gamma_zero_zero_groups_missing_neighbors_and_saturation():
    q,_=modules();s=torch.tensor([[0.,1.,-2.,100.],[0.,0.,0.,0.]])
    f=torch.tensor([[True,True,True,False],[True,True,False,False]])
    p=q.routing_probabilities(s,torch.tensor([[1.,.5,.25,0.],[1.,1.,0.,0.]]),f,.1,0.,(-1,))
    torch.testing.assert_close(p,torch.tensor([[0.,.2,.8,0.],[0.,0.,0.,0.]]),atol=1e-7,rtol=1e-6)
    assert bool((q.routing_probabilities(torch.zeros_like(s),torch.ones_like(s),f,.1,0.,(-1,))==0).all())
    assert q.routing_probabilities(s,torch.ones_like(s),f,20.,1.,(-1,))[0,1:].tolist()==[1.,1.,0.]

def test_mask_before_power_normalization_and_grouping():
    q,_=modules();s=torch.tensor([[100.,1.],[1.,3.]])
    f=torch.tensor([[False,True],[True,True]])
    p=q.routing_probabilities(s,torch.ones_like(s),f,.1,1.,(-1,))
    torch.testing.assert_close(p,torch.tensor([[0.,.2],[.05,.45]]),atol=1e-7,rtol=1e-6)
    whole=q.routing_probabilities(s,torch.ones_like(s),f,.1,1.,(0,1))
    torch.testing.assert_close(whole,torch.tensor([[0.,.08],[.08,.72]]),atol=1e-7,rtol=1e-6)
    assert p[0,1].item()!=pytest.approx(.1/50.5)

def test_large_finite_power_is_stable_and_inputs_are_read_only():
    q,_=modules();s=torch.tensor([[3e38,1e38,0.]])
    before=s.clone();p=q.routing_probabilities(s,torch.ones_like(s),torch.ones_like(s,dtype=torch.bool),1.,100.,(-1,))
    assert bool(torch.isfinite(p).all()) and p[0,0]==1 and p[0,2]==0
    assert torch.equal(s,before)

@pytest.mark.parametrize('fault',['gamma','lr','axis','gap','feasible','nan'])
def test_invalid_routing_operands_fail_closed(fault):
    q,_=modules();s=torch.ones((1,2));g=torch.ones_like(s);f=torch.ones_like(s,dtype=torch.bool);lr=.1;gamma=0.;axes=(-1,)
    if fault=='gamma':gamma=-1.
    if fault=='lr':lr=float('nan')
    if fault=='axis':axes=(0,0)
    if fault=='gap':g[0,0]=0
    if fault=='feasible':f=f.float()
    if fault=='nan':s[0,0]=float('inf')
    with pytest.raises((ValueError,TypeError)):q.routing_probabilities(s,g,f,lr,gamma,axes)

def test_uniform_fixture_literal_eq8_endpoint_and_one_code_move():
    q,_=modules();codes=torch.tensor([[0,2,4,2]],dtype=torch.int64)
    signal=torch.tensor([[100.,1.,-2.,0.]])
    next_codes,p=q.uniform_qar_step(codes,signal,spacing=1.,minimum=0,maximum=4,lr=.1,gamma=1.,group_axes=(-1,),uniforms=torch.tensor([[0.,.1,.1,0.]]))
    torch.testing.assert_close(p,torch.tensor([[0.,.4,0.,0.]]),atol=1e-7,rtol=1e-6)
    assert next_codes.tolist()==[[0,1,4,2]] and int((next_codes-codes).abs().max())<=1

def test_stochastic_probability_expectation_and_enumerated_uniforms():
    _,r=modules();x=torch.full((100,1),1.04);s=torch.ones((1,1));uniforms=(torch.arange(100,dtype=torch.float32)[:,None]+.5)/100
    values,codes=r.stochastic_round(x,s,uniforms)
    # Between 1 and 1.5: p=(1.04-1)/.5=.08; E=1*.92+1.5*.08=1.04.
    assert int((codes==10).sum())==8 and int((codes==9).sum())==92
    assert float(values.mean())==pytest.approx(1.04,abs=2e-7)
    assert float(r.upper_probability(torch.tensor([[1.04]]),s))==pytest.approx(.08,abs=1e-7)

def test_exact_levels_endpoints_saturation_negative_values_and_explicit_uniforms():
    _,r=modules();x=torch.tensor([[-100.,-1.04,0.,1.,1.5,6.,100.]])
    a,ca=r.stochastic_round(x,torch.ones((1,1)),torch.zeros_like(x))
    b,cb=r.stochastic_round(x,torch.ones((1,1)),torch.full_like(x,.999))
    assert a.tolist()==[[-6.,-1.,0.,1.,1.5,6.,6.]]
    assert b.tolist()==[[-6.,-1.5,0.,1.,1.5,6.,6.]]
    assert ca.dtype==torch.uint8 and cb.dtype==torch.uint8
    for bad in (-.1,1.,float('nan')):
        with pytest.raises(ValueError):r.stochastic_round(x,torch.ones((1,1)),torch.full_like(x,bad))

@pytest.mark.parametrize('scale',[.1,.3,3.14159,1e-37])
def test_every_reconstructed_physical_level_is_deterministic(scale):
    _,r=modules()
    from ralup.quantization.formats import quantize_e2m1
    levels=torch.tensor([[-6.,-4.,-3.,-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.,3.,4.,6.]])
    scales=torch.tensor([[scale]])
    exact,codes=quantize_e2m1(levels*scales,scales)
    for u in (0.,.5,.999):
        values,sampled=r.stochastic_round(exact,scales,torch.full_like(exact,u))
        assert torch.equal(codes,sampled) and torch.equal(exact,values)
