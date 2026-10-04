import importlib
import pytest
import torch


def ewgs():
    spec=importlib.util.find_spec('ralup.baselines.ewgs')
    assert spec is not None, 'M03 EWGS implementation is missing'
    return importlib.import_module('ralup.baselines.ewgs')


def test_uniform_formula_signs_and_clipping():
    m=ewgs();x=torch.tensor([.1,.2,.6,.9]);q=torch.round(x*3)/3
    g=torch.tensor([2.,-3.,0.,-4.]);mask=torch.tensor([1.,1.,1.,0.])
    torch.testing.assert_close(m.ewgs_backward(g,x,q,.3,mask),g*(1+.3*g.sign()*(x-q))*mask)


def test_fp4_dimensionless_residual_and_rotation():
    m=ewgs()
    from ralup.quantization.context import prepare_context,quantized_view
    w=torch.tensor([[.14,-.21,.43,-.61]],requires_grad=True)
    c=prepare_context(w,{'name':'row_fixed_max','rotation':'full_hadamard','backward':'ewgs'},0,None)
    c=m.with_delta(c,.4);g=torch.tensor([[2.,-1.,3.,-4.]])
    quantized_view(w,c).backward(g)
    from ralup.quantization.rotation import from_storage
    z=(c.storage_u/c.scales/6).clamp(-1,1);q=c.storage_values/c.scales/6
    expected=from_storage(g*(1+.4*g.sign()*(z-q))*c.gradient_mask,c.rotation)
    torch.testing.assert_close(w.grad,expected)


def test_configuration_is_explicit_and_adaptive_is_gated():
    m=ewgs()
    with pytest.raises(ValueError):m.EwgsAdapter({})
    config=dict(m.FIXTURE_CONFIG,mode='adaptive',pilot_relevance=False)
    with pytest.raises(ValueError,match='relevance'):m.EwgsAdapter(config)
    with pytest.raises(ValueError):m.EwgsAdapter({**m.FIXTURE_CONFIG,'unknown':1})


def test_registry_separates_fixture_and_adaptation():
    m=ewgs()
    from ralup.baselines.registry import default_registry
    from ralup.optim.base import ParameterGroup
    registry=default_registry((ParameterGroup('internal_projection',('w',)),))
    original=registry.require('ewgs_uniform_fixture',adaptation_label='original_uniform_primitive_clean_equations')
    fp4=registry.require('ewgs_fp4',adaptation_label='project_FP4_population_std_detached_normalized_leaf')
    assert original.artifact_kind=='original_fixture' and fp4.artifact_kind=='project_adaptation'
    assert fp4.update_rule_settings['source_copy_status']=='blocked_GPL_inspection_only'
    assert not registry.claim_eligibility('ewgs_fp4')['blocked_is_defeated_baseline']


def test_fp4_fixed_out_of_range_clips_physical_upstream():
    m=ewgs()
    from ralup.quantization.context import prepare_context,quantized_view
    w=torch.tensor([[4.,-.2,.3,-4.]],requires_grad=True)
    context=prepare_context(w,{'name':'row_fixed_max','backward':'ewgs','previous_scales':torch.tensor([[.5]])},0,None)
    context=m.with_delta(context,.5);upstream=torch.tensor([[2.,-3.,4.,-5.]])
    quantized_view(w,context).backward(upstream)
    z=(w.detach()/.5/6).clamp(-1,1);q=context.storage_values/.5/6
    expected=upstream*(1+.5*upstream.sign()*(z-q))*torch.tensor([[0.,1.,1.,0.]])
    torch.testing.assert_close(w.grad,expected)


def test_ewgs_rule_owner_rejects_matrix_optimizer():
    m=ewgs()
    from ralup.optim.base import ParameterGroup
    from ralup.optim.muon import MuonUpdateRule
    with pytest.raises(ValueError,match='AdamW'):
        m.EwgsAdapter(m.FIXTURE_CONFIG).bind({'w':torch.nn.Parameter(torch.ones(2,2))},(ParameterGroup('internal_projection',('w',)),),MuonUpdateRule({}))
