"""Breaks caught: lost initial skip, decay order, refresh/permutation, W moment mapping."""
import copy,hashlib,importlib.util
from pathlib import Path
import pytest
import torch
from ralup.optim.base import SchedulePoint
from ralup.baselines.references import soap_reference

SOURCE=Path(soap_reference.__file__).resolve()

def oracle_type():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='9e9021335ad02371584622fa256e73fe75f3eb74446a8d8a216e956534148f6b'
    spec=importlib.util.spec_from_file_location('pinned_m02_soap',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.SOAP

def soap_rule(config):
    assert importlib.util.find_spec('ralup.optim.soap') is not None, 'M02 SOAP adapter missing'
    from ralup.optim.soap import SoapUpdateRule
    return SoapUpdateRule(config)

@pytest.mark.parametrize('config',[{'precondition_frequency':1},{'precondition_frequency':4},{'precondition_frequency':10},
 {'precondition_frequency':1,'max_precond_dim':2},{'merge_dims':True,'max_precond_dim':10},
 {'precondition_frequency':1,'correct_bias':False,'normalize_grads':True}])
def test_fixed_sequence_matches_untouched_pinned_source_and_original_W_moment(config):
    initial=torch.tensor([[-.3,0.,1.1],[.01,-2.,.7]])
    w=initial.clone();other=torch.nn.Parameter(initial.clone());rule=soap_rule(config)
    oracle=oracle_type()([other],lr=.03,weight_decay=.4,**config);state={};ema=torch.zeros_like(w)
    for step in range(12):
        g=torch.tensor([[.2,-.7,.4],[.3,.9,-.1]])*(1 if step%2 else -1)+step*.03
        old_w=w.clone();old_g=g.clone();before=copy.deepcopy(state);proposal=rule.propose(w,g,state,SchedulePoint(.03,.4,step/12))
        assert torch.equal(w,old_w) and torch.equal(g,old_g)
        for key in ('GG','Q'):
            if before:
                for actual,old in zip(state[key],before[key]):
                    if isinstance(actual,torch.Tensor):assert torch.equal(actual,old)
        for key in ('exp_avg','exp_avg_sq'):
            if before: assert torch.equal(state[key],before[key])
        other.grad=g.clone();oracle.step();rule.commit(w,proposal);state=proposal.new_optimizer_state
        torch.testing.assert_close(w,other,atol=0,rtol=0)
        if step==0:
            assert torch.equal(w,initial) and state['step']==0
            assert torch.equal(state['exp_avg'],torch.zeros_like(w))
        else:ema.mul_(.95).add_(g,alpha=.05)
        torch.testing.assert_close(rule.first_moment_in_original_coordinates(state),ema,atol=1e-6,rtol=1e-5)
        for key in ('exp_avg','exp_avg_sq'):
            torch.testing.assert_close(state[key],oracle.state[other][key],atol=0,rtol=0)
        for key in ('GG','Q'):
            for actual,expected in zip(state[key],oracle.state[other][key]):
                if isinstance(actual,torch.Tensor):torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                else:assert actual==expected==[]
    arrays=[state['exp_avg'],state['exp_avg_sq'],*[t for key in ('GG','Q') for t in state[key] if isinstance(t,torch.Tensor)]]
    inventory=rule.state_inventory(state)
    assert inventory['tensor_bytes']==sum(t.numel()*t.element_size() for t in arrays)
    assert inventory['factor_bytes']==sum(t.numel()*t.element_size() for t in state['GG'] if isinstance(t,torch.Tensor))
    assert inventory['basis_bytes']==inventory['factor_bytes']
    assert inventory['factorization_seconds']>=0 and inventory['transfer_bytes']==0

def test_no_initialization_skip_would_violate_reference():
    rule=soap_rule({});w=torch.ones(2,2);p=rule.propose(w,torch.ones_like(w),{},SchedulePoint(.1,.5,0.))
    broken=w*(1-.1*.5)-.1
    assert not torch.equal(broken,p.reference_after_decay+p.gradient_displacement)
    assert torch.equal(p.reference_after_decay,w) and torch.count_nonzero(p.gradient_displacement)==0

@pytest.mark.parametrize('config',[{'precondition_frequency':0},{'max_precond_dim':0},{'eps':-1},{'unknown':1}])
def test_rejects_unsupported_configuration(config):
    with pytest.raises(ValueError):soap_rule(config)
