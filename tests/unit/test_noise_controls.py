import importlib
import math
import pytest
import torch


def module():
    assert importlib.util.find_spec('ralup.baselines.noise'), 'M06 noise implementation missing'
    return importlib.import_module('ralup.baselines.noise')


def test_simple_doubles_and_magma_alignment_never_changes_probability():
    m=module();d=torch.tensor([[-.1,.2]]);moment=torch.tensor([[1.,-2.]]);g=moment.clone()
    simple,state=m.mask_displacement(d,moment,g,{},.49,{**m.MASK_CONFIG,'mode':'simple'})
    torch.testing.assert_close(simple,2*d)
    alignment=.9*.5+.1/(1+math.exp(-.5))
    accepted,state=m.mask_displacement(d,moment,g,{},.49,m.MASK_CONFIG)
    torch.testing.assert_close(accepted,alignment*d)
    rejected,new=m.mask_displacement(d,moment,g,{},.51,m.MASK_CONFIG)
    assert torch.count_nonzero(rejected)==0 and new['probability']==state['probability']==.5
    assert new['alignment']==pytest.approx(state['alignment'])


def test_zero_cosine_and_state_even_rejected():
    m=module();d=torch.ones(2)
    out,s=m.mask_displacement(d,torch.zeros(2),d,{},.9,m.MASK_CONFIG)
    assert s['alignment']==.5 and s['observations']==1 and s['cosine']==0 and out.sum()==0
    _,s=m.mask_displacement(d,d,-d,s,.9,m.MASK_CONFIG)
    assert s['observations']==2 and s['alignment']<.5


def test_damping_stopgrad_exact_penalty_and_gradient():
    m=module();w=torch.tensor([1.,2.],requires_grad=True);q=torch.tensor([0.,3.],requires_grad=True)
    loss=m.damping_penalty(w,q,.4);loss.backward()
    assert loss.item()==pytest.approx(.4)
    torch.testing.assert_close(w.grad,torch.tensor([.4,-.4]));assert q.grad is None
    with pytest.raises(ValueError):m.damping_penalty(w,q,-1.)


@pytest.mark.parametrize('mode',['simple','magma','damping'])
def test_adapter_dense_moments_mask_replay_decay_and_pass_accounting(mode):
    from test_learned_scale import fixture
    from ralup.optim.base import default_parameter_groups
    from ralup.train import train_step
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    from ralup.config import canonical_bytes
    m=module();state,batch=fixture()
    state.method=m.NoiseAdapter({**m.FIXTURE_CONFIG,'mode':mode,'coefficient':.1 if mode=='damping' else 0.}).bind(dict(state.model.named_parameters()),default_parameter_groups(state.model))
    state.optimizer_state=state.method.optimizer_states
    result=train_step(state,batch)
    assert all(v['step'].item()==1 for v in state.method.optimizer_states.values())
    assert result.work['backward']==result.work['loss_targets']==4
    if mode!='damping':
        accepted=[s for s in state.method.mask_states.values() if s['accepted']]
        rejected=[s for s in state.method.mask_states.values() if not s['accepted']]
        assert accepted and rejected
        assert result.work['noise_work']['dense_moment_updates']==len(state.method.parameters)
    else:assert result.work['damping_work']['extra_loss_passes']==0
    canonical_bytes(state.method.metadata)
    resumed=restore_snapshot(capture_snapshot(state),'cpu');train_step(state,batch);train_step(resumed,batch)
    assert exact_cpu_state_hash(state)==exact_cpu_state_hash(resumed)


def test_rejected_and_accepted_moves_both_advance_original_w_moments_and_decay():
    from ralup.optim.base import ParameterGroup,SchedulePoint
    from ralup.optim.adamw import AdamWUpdateRule
    m=module();w=torch.nn.Parameter(torch.ones(2,2));rule=AdamWUpdateRule()
    group=(ParameterGroup('internal_projection',('w',)),)
    method=m.NoiseAdapter(m.FIXTURE_CONFIG).bind({'w':w},group)
    w.grad=torch.ones_like(w);p=method.collect_proposals(SchedulePoint(.1,.2,0.))['w']
    accepted,s=m.mask_displacement(p.gradient_displacement,p.new_optimizer_state['exp_avg'],w.grad,{},.4,m.MASK_CONFIG)
    rejected,r=m.mask_displacement(p.gradient_displacement,p.new_optimizer_state['exp_avg'],w.grad,{},.6,m.MASK_CONFIG)
    assert r['observations']==s['observations']==1 and p.new_optimizer_state['step'].item()==1
    torch.testing.assert_close(p.reference_after_decay+rejected,torch.full_like(w,.98))
    assert not torch.equal(p.reference_after_decay+accepted,p.reference_after_decay)


def test_damping_gradient_and_result_independent_of_microbatch_packing():
    import hashlib
    from test_learned_scale import fixture
    from ralup.optim.base import default_parameter_groups
    from ralup.train import train_step
    from ralup.data.articles import Article
    from ralup.data.blocks import tile_article
    from ralup.data.batches import LogicalBatch
    m=module();a,_=fixture();b,_=fixture();b.microbatch=2
    for state in (a,b):
        state.method=m.NoiseAdapter({**m.FIXTURE_CONFIG,'mode':'damping','coefficient':.1}).bind(dict(state.model.named_parameters()),default_parameter_groups(state.model));state.optimizer_state=state.method.optimizer_states
    blocks=[x for raw in (b'abcd',b'efgh') for x in tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))]
    batch=LogicalBatch(blocks,8);train_step(a,batch);train_step(b,batch)
    for n,p in a.model.named_parameters():
        torch.testing.assert_close(p.grad,dict(b.model.named_parameters())[n].grad,atol=2e-7,rtol=2e-5)


@pytest.mark.parametrize('name',['muon_project_fp32','muon_plus_fp32'])
def test_masks_wrap_current_original_w_matrix_momentum(name):
    from test_learned_scale import fixture
    from ralup.baselines.trs import backbone_config,BACKBONES
    from ralup.optim.base import default_parameter_groups
    from ralup.train import train_step
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    m=module();state,batch=fixture()
    state.method=m.NoiseAdapter({**m.FIXTURE_CONFIG,'backbone':backbone_config(BACKBONES[name]({}))}).bind(dict(state.model.named_parameters()),default_parameter_groups(state.model));state.optimizer_state=state.method.optimizer_states
    train_step(state,batch);resumed=restore_snapshot(capture_snapshot(state),'cpu')
    for n in state.method.groups[0].parameter_names:assert state.method.optimizer_states[n]['momentum'].shape==state.method.parameters[n].shape
    train_step(state,batch);train_step(resumed,batch)
    assert exact_cpu_state_hash(state)==exact_cpu_state_hash(resumed)
