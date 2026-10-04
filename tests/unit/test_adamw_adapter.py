"""Original-W exact scalar AdamW and data-neutral adapter boundaries."""
import copy

import pytest
import torch


@pytest.mark.parametrize('betas,eps,decay', [((.9,.99),1e-8,0.),((.9,.95),1e-8,.1),((.95,.999),1e-4,.03),((0.,0.),.01,.2)])
def test_fixed_sequence_matches_scalar_torch_without_mutating_proposal_inputs(betas,eps,decay):
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.optim.base import SchedulePoint
    rule = AdamWUpdateRule(betas=betas,eps=eps)
    initial = torch.tensor([[-.31,0.,1.7],[.001,-4.2,.73]],dtype=torch.float32)
    weight = torch.nn.Parameter(initial.clone())
    identity = weight
    oracle = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([oracle],lr=.001,betas=betas,eps=eps,weight_decay=decay,foreach=False,fused=False)
    state = {}
    for index in range(40):
        grad = torch.zeros_like(weight) if index%5 == 0 else torch.tensor([[.13,-.7,0.],[.002,1.9,-.03]])*((index%3)-1)
        lr = .001 if index<20 else .00037
        before = weight.detach().clone()
        old = copy.deepcopy(state)
        proposal = rule.propose(weight,grad,state,SchedulePoint(lr,decay,index/40))
        assert torch.equal(weight,before)
        assert state.keys() == old.keys()
        assert all(torch.equal(value,old[key]) for key,value in state.items())
        optimizer.param_groups[0]['lr'] = lr
        oracle.grad = grad.clone()
        optimizer.step()
        rule.commit(weight,proposal)
        state = proposal.new_optimizer_state
        assert identity is weight
        assert torch.equal(weight,oracle), (index,weight,oracle)
        assert all(torch.equal(state[key],optimizer.state[oracle][key]) for key in state)


def test_telemetry_addition_cannot_replace_authoritative_rounded_commit():
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.optim.base import SchedulePoint
    weight = torch.nn.Parameter(torch.tensor([.001,-.31,1.7,-4.2]))
    oracle = torch.nn.Parameter(weight.detach().clone())
    optimizer = torch.optim.AdamW([oracle],lr=.001,weight_decay=.1,betas=(.9,.99),foreach=False,fused=False)
    oracle.grad = torch.tensor([.002,.13,-.7,1.9])
    optimizer.step()
    rule = AdamWUpdateRule()
    proposal = rule.propose(weight,oracle.grad,{},SchedulePoint(.001,.1,0.))
    algebraic = proposal.reference_after_decay + proposal.gradient_displacement
    rule.commit(weight,proposal)
    assert torch.equal(weight,oracle)
    assert not torch.equal(algebraic,weight)


def test_default_groups_exclude_embeddings_head_and_norm_and_cover_manifest():
    from ralup.optim.base import default_parameter_groups
    from ralup.models.transformer import ByteTransformer,ModelConfig
    model = ByteTransformer(ModelConfig(1,8,2))
    groups = default_parameter_groups(model)
    assert groups[0].parameter_names == tuple(item['name'] for item in model.quantized_matrix_manifest())
    assert groups[0].weight_decay is None
    assert groups[1].weight_decay == 0.
    assert {'embedding.weight','position.weight','output.weight','final_norm.weight','final_norm.bias'} <= set(groups[1].parameter_names)
    assert set(sum((group.parameter_names for group in groups),())) == set(dict(model.named_parameters()))


def test_adapter_backward_is_sum_and_finish_publishes_named_state_once():
    from ralup.optim.base import MethodAdapter,ParameterGroup,SchedulePoint
    from ralup.optim.adamw import AdamWUpdateRule
    parameter = torch.nn.Parameter(torch.tensor([2.]))
    class State:
        def summed_nll(self,batch_slice):
            return (parameter*batch_slice).sum()
    adapter = MethodAdapter({'w':parameter},(ParameterGroup('internal',('w',)),),AdamWUpdateRule())
    adapter.prepare_step(State(),object())
    work = adapter.backward_microbatch(State(),torch.tensor([3.]))
    assert parameter.grad.item() == 3.
    assert work['backward_passes'] == work['forward_passes'] == 1
    before = parameter.detach().clone()
    proposals = adapter.collect_proposals(SchedulePoint(.001,.1,0.))
    assert torch.equal(parameter,before)
    assert adapter.optimizer_states['w'] == {}
    adapter.commit_proposals(proposals)
    assert adapter.optimizer_states['w']['step'].item() == 1
    assert adapter.snapshot_schema['optimizer_states'] == 'named.original_W.adamw.v1'
    assert 'proposals' in adapter.snapshot_schema


def test_finish_validates_every_candidate_before_any_commit():
    from ralup.optim.base import MethodAdapter,ParameterGroup,SchedulePoint
    from ralup.optim.adamw import AdamWUpdateRule
    parameters = {name:torch.nn.Parameter(torch.tensor([value])) for name,value in [('a',1.),('b',2.)]}
    parameters['a'].grad = torch.tensor([.2])
    parameters['b'].grad = torch.tensor([float('nan')])
    adapter = MethodAdapter(parameters,(ParameterGroup('internal',('a','b')),),AdamWUpdateRule())
    with pytest.raises(ValueError,match='finite'):
        adapter.finish_step(None,SchedulePoint(.001,0.,0.))
    assert parameters['a'].item() == 1.
    assert parameters['b'].item() == 2.
    assert adapter.optimizer_states == {'a':{},'b':{}}


def test_missing_gradient_does_not_advance_moments_or_decay():
    from ralup.optim.base import MethodAdapter,ParameterGroup,SchedulePoint
    from ralup.optim.adamw import AdamWUpdateRule
    weight = torch.nn.Parameter(torch.tensor([2.]))
    adapter = MethodAdapter({'w':weight},(ParameterGroup('internal',('w',)),),AdamWUpdateRule())
    work = adapter.finish_step(None,SchedulePoint(.001,.1,0.))
    assert weight.item() == 2.
    assert adapter.optimizer_states['w'] == {}
    assert work['optimizer_updates'] == 0


@pytest.mark.parametrize('kwargs',[{'lr':-1.,'weight_decay':0.,'progress':0.},{'lr':.001,'weight_decay':float('nan'),'progress':0.},{'lr':.001,'weight_decay':0.,'progress':2.}])
def test_invalid_schedule_is_rejected(kwargs):
    from ralup.optim.base import SchedulePoint
    with pytest.raises(ValueError):
        SchedulePoint(**kwargs)


def test_nonfinite_optimizer_state_blocks_all_candidates_and_repeat_commit_is_rejected():
    from ralup.optim.base import MethodAdapter,ParameterGroup,SchedulePoint
    from ralup.optim.adamw import AdamWUpdateRule
    parameters = {name:torch.nn.Parameter(torch.tensor([value])) for name,value in [('a',1.),('b',2.)]}
    for parameter in parameters.values():
        parameter.grad = torch.tensor([.2])
    adapter = MethodAdapter(parameters,(ParameterGroup('internal',('a','b')),),AdamWUpdateRule())
    adapter.optimizer_states['b'] = {'step':torch.tensor(1.),'exp_avg':torch.tensor([float('nan')]),'exp_avg_sq':torch.tensor([.01])}
    proposals = adapter.collect_proposals(SchedulePoint(.001,0.,0.))
    with pytest.raises(ValueError,match='finite'):
        adapter.commit_proposals(proposals)
    assert parameters['a'].item() == 1.
    assert parameters['b'].item() == 2.
    adapter.proposals = {}
    adapter.optimizer_states['b'] = {}
    proposals = adapter.collect_proposals(SchedulePoint(.001,0.,0.))
    adapter.commit_proposals(proposals)
    with pytest.raises(ValueError,match='current uncommitted'):
        adapter.commit_proposals(proposals)
    assert adapter.optimizer_states['a']['step'].item() == 1.
