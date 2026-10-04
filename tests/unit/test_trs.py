import importlib
import pytest
import torch


def trs():
    assert importlib.util.find_spec('ralup.baselines.trs') is not None, 'M04 TRS implementation missing'
    return importlib.import_module('ralup.baselines.trs')


def test_released_first_rate_ignored_and_multiplicative_fixed_values():
    m=trs();state=m.TrsState(1.,.1,torch.zeros(4),0)
    first,new=m.trs_effective_lr(state,.75,.1,.01,.99)
    assert first==.01 and new.K==.1 and new.r==1 and new.observations==1
    second,new=m.trs_effective_lr(new,.5,.1,.01,.99)
    assert new.K==pytest.approx(.104)
    assert new.r==pytest.approx(.9996153836908285)
    assert second==pytest.approx(.009996153836908285)
    third,new=m.trs_effective_lr(new,0.,.2,.01,.99)
    assert new.K==pytest.approx(.10296)
    assert new.r==pytest.approx(1.0090367762137316)
    assert third==pytest.approx(.010090367762137316)
    assert state.observations==0 and state.K==.1 and state.r==1
    new.previous_codes.add_(1);assert state.previous_codes.sum()==0


def test_release_fixture_coupled_decay_and_fixed_scale_against_pinned_outputs():
    m=trs();w=torch.tensor([[.3,-.2,.1,-.4]])
    fixture=m.TrsReleaseFixture(w,target_rate=.1,alpha=.01,mu=.99,betas=(.9,.999),eps=1e-8,bits=3,symmetric=False)
    scale=fixture.scale.clone()
    # First coupled observation is [.13,-.12,.31,-.24], not the raw gradient.
    fixture.step(w,torch.tensor([[.1,-.1,.3,-.2]]),target_rate=.1,weight_decay=.1)
    torch.testing.assert_close(w,torch.tensor([[.29000002,-.19,.09,-.39000002]]),atol=2e-8,rtol=1e-7)
    torch.testing.assert_close(fixture.optimizer_state['exp_avg'],torch.tensor([[.013,-.012,.031,-.024]]),atol=1e-8,rtol=1e-7)
    fixture.step(w,torch.tensor([[-.3,.2,-.1,.4]]),target_rate=.1,weight_decay=.1)
    assert fixture.state.observations==2
    assert fixture.state.K==pytest.approx(.099)
    assert fixture.last_lr==pytest.approx(.01000101009080604)
    torch.testing.assert_close(w,torch.tensor([[.29381345,-.19250204,.08566745,-.39248970]]),atol=5e-7,rtol=1e-6)
    assert torch.equal(scale,fixture.scale)
    assert m.release_normalized_codes(torch.tensor([[0.,100.,-100.]]),torch.tensor(2.),3,False).tolist()==[[0.,.75,-1.]]


def test_normalized_values_ignore_physical_scale_and_duplicate_zero_codes():
    m=trs()
    from ralup.quantization.context import prepare_context
    w=torch.tensor([[0.,.2,-.4,.8]])
    a=prepare_context(w,{'name':'row_fixed_max'},0,None)
    b=prepare_context(w*2,{'name':'row_fixed_max'},1,None)
    assert torch.equal(m.normalized_codes(a),m.normalized_codes(b))
    assert torch.equal(m.normalized_codes(a),a.storage_values/a.scales/6)


@pytest.mark.parametrize('args',[(float('nan'),.1,.01,.99),(.1,-1.,.01,.99),(.1,.1,-1.,.99),(.1,.1,.01,1.),(1.1,.1,.01,.99)])
def test_feedback_rejects_invalid_inputs(args):
    m=trs()
    with pytest.raises(ValueError):m.trs_effective_lr(m.TrsState(1,.1,torch.zeros(2),1),*args)


def test_distinct_registry_schedule_metadata_and_claim_limits():
    m=trs()
    from ralup.baselines.registry import default_registry
    from ralup.optim.base import ParameterGroup
    registry=default_registry((ParameterGroup('internal_projection',('w',)),))
    fixture=registry.require('trs_release_fixture',adaptation_label='released_multiplicative_coupled_Adam_clean_equations')
    adapter=registry.require('trs_backbone_adapter',adaptation_label='project_FP4_logical_step_decoupled_backbone')
    assert fixture.artifact_kind=='original_fixture' and adapter.artifact_kind=='project_adaptation'
    assert adapter.schedule_fields==('lr','weight_decay','progress','target_rate')
    assert adapter.update_rule_settings['paper_feedback']=='additive max(0,U+eta*(R-K)); not implemented as release'
    assert not registry.claim_eligibility('trs_release_fixture')['eligible']
    schedule=m.validate_config({**m.FIXTURE_CONFIG,'target_schedule':{'kind':'linear','start':.2,'end':.05}})
    assert m.target_rate(schedule,0)==.2 and m.target_rate(schedule,1)==.05
    with pytest.raises(ValueError):m.validate_config({**m.FIXTURE_CONFIG,'unknown':1})


def test_backbone_compatibility_is_explicit_and_unsupported_is_not_defeated():
    m=trs()
    from ralup.optim.soap import SoapUpdateRule
    from ralup.optim.madam import MadamUpdateRule
    for rule in (SoapUpdateRule({}),MadamUpdateRule({})):
        with pytest.raises(ValueError,match='trusted original-W'):
            m.TrsBackboneAdapter(rule,m.FIXTURE_CONFIG)
    from ralup.baselines.registry import default_registry
    from ralup.optim.base import ParameterGroup
    registry=default_registry((ParameterGroup('internal_projection',('w',)),))
    record=registry.require('trs_backbone_adapter',adaptation_label='project_FP4_logical_step_decoupled_backbone')
    assert record.update_rule_settings['unsupported_backbone_families'][:2]==['SOAP','Madam']
    assert registry.claim_eligibility('trs_backbone_adapter')['blocked_is_defeated_baseline'] is False
