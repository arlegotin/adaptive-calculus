"""Real H04 target loss, prior histories, original-W moments and safe replay."""
import hashlib
import json
import importlib
import pytest
import torch


def mq():
    assert importlib.util.find_spec('ralup.baselines.metaquant') is not None, 'M10 mechanism missing'
    return importlib.import_module('ralup.baselines.metaquant')


def fixture(variant='online_target_training',rotation='identity',microbatch=1,device='cpu'):
    m=mq()
    from ralup.models.transformer import ByteTransformer,ModelConfig
    from ralup.optim.base import default_parameter_groups,SchedulePoint
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.train import TrainingState
    from ralup.data.articles import Article
    from ralup.data.blocks import tile_article
    from ralup.data.batches import LogicalBatch
    with torch.random.fork_rng():
        torch.manual_seed(1010);model=ByteTransformer(ModelConfig(1,8,2)).to(device)
    model.configure_quantization({'name':'row_fixed_max','rotation':rotation,'backward':'ste'})
    method=m.MetaQuantLMAdapter(AdamWUpdateRule(),{**m.FIXTURE_CONFIG,'variant':variant,'rotation':rotation}).bind(dict(model.named_parameters()),default_parameter_groups(model))
    blocks=[b for raw in (b'abcd',b'efgh') for b in tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))]
    return TrainingState(model,method,microbatch=microbatch,schedule={'step':0,'point':SchedulePoint(.002,.1,0.)}),LogicalBatch(blocks,8)


@pytest.mark.parametrize('variant',['frozen_theta','online_target_training'])
@pytest.mark.parametrize('rotation',['identity','full_hadamard'])
def test_priming_previous_signals_pretheta_commit_and_complete_snapshot_replay(variant,rotation):
    from ralup.train import train_step
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    from ralup.quantization.rotation import from_storage
    state,batch=fixture(variant,rotation);method=state.method
    weights={n:p.detach().clone() for n,p in state.model.named_parameters()};theta={n:p.detach().clone() for n,p in method.theta.items()}
    first=train_step(state,batch)
    assert first.work['metaquant_work']['priming_selected_matrix_updates']==0
    for n in method.gprev:assert torch.equal(method.parameters[n],weights[n])
    assert any(not torch.equal(p,weights[n]) for n,p in method.parameters.items() if n not in method.gprev)
    assert first.work['backward']==8 and first.work['additional_loss']==0
    events=first.work['metaquant_work']['loss_pass_events']
    assert sum(e['target_bytes'] for e in events)==8 and len({e['execution_id'] for e in events})==2
    assert all(e['backward_launched'] and e['forward_completed'] for e in events)
    previous={n:g.clone() for n,g in method.gprev.items()}
    corrections=method.corrections(state.model._contexts)
    expected={n:from_storage(c.detach(),rotation) for n,c in corrections.items()}
    restored=restore_snapshot(capture_snapshot(state),'cpu')
    second=train_step(state,batch);train_step(restored,batch)
    assert exact_cpu_state_hash(state)==exact_cpu_state_hash(restored)
    for n in previous:
        torch.testing.assert_close(method.committed_signal[n],expected[n])
        assert not torch.equal(method.gprev[n],previous[n])
        assert float(method.optimizer_states[n]['step'])==1
    changed=any(not torch.equal(p,theta[n]) for n,p in method.theta.items())
    assert changed==(variant=='online_target_training')
    assert method.theta_updates==(1 if changed else 0)
    assert second.work['backward']==8 and second.work['additional_loss']==0
    assert all(p.grad is not None for p in method.theta.values()) if changed else all(p.grad is None for p in method.theta.values())


def test_functional_mapping_read_only_and_microbatch_graphs_not_retained():
    from ralup.train import train_step
    from ralup.state import exact_cpu_state_hash
    from ralup.evaluate import summed_nll
    one,batch=fixture();two,_=fixture(microbatch=2)
    train_step(one,batch);train_step(two,batch)
    before=exact_cpu_state_hash(one)
    views,leaves=one.method.functional_views(one,one.schedule['point'])
    loss=summed_nll(one.model,batch,parameter_overrides=views)
    assert loss.requires_grad and exact_cpu_state_hash(one)==before
    assert set(leaves)==set(one.method.gprev)
    train_step(one,batch);train_step(two,batch)
    for n,p in one.model.named_parameters():torch.testing.assert_close(p,dict(two.model.named_parameters())[n],atol=2e-6,rtol=2e-4)
    assert all(g.grad_fn is None and not g.requires_grad for g in one.method.gprev.values())


@pytest.mark.parametrize('field',['gprev','rprev','cached_ranges','outer_states','theta_gradients','committed_signal'])
def test_snapshot_rejects_missing_scientific_owner(field):
    from ralup.train import train_step
    from ralup.state import capture_snapshot,IntegrityError
    state,batch=fixture();train_step(state,batch);train_step(state,batch)
    owner=getattr(state.method,field);owner.pop(next(iter(owner)))
    with pytest.raises(IntegrityError):capture_snapshot(state)


def test_cli_original_mechanism_and_fp4_are_separate(tmp_path):
    m=mq()
    from ralup.__main__ import main
    output=tmp_path/'suite.json'
    assert main(['verify-baselines','--suite','metaquant','--output',str(output)])==0
    report=json.loads(output.read_bytes())
    assert report['status']=='pass' and report['scientific_gate'] is None
    assert report['source_copy_status']=='blocked_missing_grant'
    assert report['original_mechanism']['factor_gradient_parity']=='pass'
    assert {v['variant'] for v in report['variants']}=={'frozen_theta','online_target_training'}


@pytest.mark.parametrize('rotation',['identity','full_hadamard'])
def test_h04_theta_gradient_matches_independent_factor_chain_in_physical_units(rotation):
    from ralup.train import train_step
    from ralup.evaluate import summed_nll
    from ralup.quantization.rotation import to_storage
    state,batch=fixture(rotation=rotation);train_step(state,batch)
    method=state.method;views,leaves=method.functional_views(state,state.schedule['point'])
    loss=summed_nll(state.model,batch,parameter_overrides=views)
    values=tuple(method.theta.values())
    actual=torch.autograd.grad(loss,values,retain_graph=True)
    current=torch.autograd.grad(loss,tuple(leaves.values()))
    factors=[p.detach().clone().requires_grad_() for p in values]
    independent=torch.zeros(())
    for n,g in zip(leaves,current):
        normalized=to_storage(method.parameters[n].detach(),rotation)/(6*state.model._contexts[n].scales)
        unfolded=normalized.reshape(-1,1)
        for p in factors:unfolded=torch.nn.functional.linear(unfolded,p)
        independent=independent-state.schedule['point'].lr*(g.detach()*method.gprev[n]*unfolded.reshape_as(g)).sum()
    expected=torch.autograd.grad(independent,factors)
    for a,e in zip(actual,expected):torch.testing.assert_close(a,e,atol=2e-10,rtol=2e-4)


def test_live_snapshot_rejects_graph_retention_and_factor_gradient_alias_break():
    from ralup.train import train_step
    from ralup.state import capture_snapshot,IntegrityError
    state,batch=fixture();train_step(state,batch)
    n=next(iter(state.method.gprev));state.method.gprev[n]=state.method.gprev[n].requires_grad_()*2
    with pytest.raises(IntegrityError):capture_snapshot(state)


@pytest.mark.parametrize('variant',['frozen_theta','online_target_training'])
def test_cli_deployment_variants_use_named_adapter_and_finite_checkpoint(tmp_path,capsys,variant):
    from pathlib import Path
    from ralup.__main__ import main
    from ralup.state import load_snapshot,restore_snapshot
    from ralup.records import ArtifactRef
    m=mq();template=json.loads((Path(__file__).parents[2]/'configs/smoke.json').read_bytes())
    name='metaquant_fp4_frozen' if variant=='frozen_theta' else 'metaquant_fp4_online_target_training'
    envelope={'config':template,'purpose':'smoke','pilot_target_budget':7,'seed':1010,
        'output':str(tmp_path/'run'),'checkpoint_directory':str(tmp_path/'checkpoint'),
        'method':{'name':name,'adaptation_label':f'project_FP4_physical_identity_STE_{variant}',
            'config':{**m.FIXTURE_CONFIG,'variant':variant},'learning_rate':.001}}
    path=tmp_path/'run.json';path.write_text(json.dumps(envelope))
    assert main(['run','--phase','explore','--config',str(path)])==0
    result=json.loads(capsys.readouterr().out)
    restored=restore_snapshot(load_snapshot(ArtifactRef(**result['checkpoint'])),'cpu')
    assert type(restored.method) is m.MetaQuantLMAdapter and restored.method.config['variant']==variant


def test_fast_context_forgery_rejected_and_failed_commit_rolls_back_theta_identity():
    from dataclasses import replace
    from ralup.train import train_step,TrainingFailure
    from ralup.state import capture_snapshot,IntegrityError
    state,batch=fixture();train_step(state,batch);n=next(iter(state.method.fast_contexts))
    c=state.method.fast_contexts[n];state.method.fast_contexts[n]=replace(c,storage_values=c.storage_values+1)
    with pytest.raises(IntegrityError):capture_snapshot(state)
    state.method.fast_contexts[n]=c
    identities={n:id(p) for n,p in state.method.theta.items()};before={n:p.detach().clone() for n,p in state.method.theta.items()}
    state.model.forward=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError('deliberate failed actual forward'))
    with pytest.raises(TrainingFailure):train_step(state,batch)
    assert state.ledger.events[-1]['loss_targets']==4 and state.ledger.events[-1]['backward']==0
    event=state.ledger.events[-1]['metaquant_work']['loss_pass_events'][0]
    assert event['target_bytes']==4 and not event['forward_completed'] and not event['backward_launched']
    assert identities=={n:id(p) for n,p in state.method.theta.items()}
    for n,p in state.method.theta.items():assert torch.equal(p,before[n])


def test_priming_failed_forward_charges_only_launched_exposures():
    from ralup.train import train_step,TrainingFailure
    state,batch=fixture()
    state.model.forward=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError('deliberate priming forward failure'))
    with pytest.raises(TrainingFailure):train_step(state,batch)
    work=state.ledger.events[-1]
    assert work['loss_targets']==4 and work['backward']==0
    assert work['metaquant_work']['priming_loss_target_bytes']==4
    assert work['metaquant_work']['priming_backward_target_bytes']==0


def test_correction_and_auxiliary_share_one_logical_gradient_clip():
    from ralup.train import train_step
    state,batch=fixture(variant='frozen_theta');state.clip_norm=.001
    train_step(state,batch)
    with torch.no_grad():
        for p in state.method.theta.values():p.fill_(1.)
    train_step(state,batch)
    norm=torch.stack([p.grad.square().sum() for p in state.model.parameters() if p.grad is not None]).sum().sqrt()
    assert float(norm)<=.001001


def test_framework_meta_initialization_preserves_enclosing_rng_streams():
    m=mq()
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.optim.base import ParameterGroup
    torch.manual_seed(44)
    cpu=torch.get_rng_state().clone()
    native=torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    m.MetaQuantLMAdapter(AdamWUpdateRule(),m.FIXTURE_CONFIG).bind({'w':torch.nn.Parameter(torch.ones(2,2))},(ParameterGroup('internal_projection',('w',)),))
    assert torch.equal(cpu,torch.get_rng_state())
    if native is not None:assert torch.equal(native,torch.mps.get_rng_state())
