import itertools
import pytest
import torch
import importlib

def ewgs():
    assert importlib.util.find_spec("ralup.baselines.ewgs") is not None, "M03 EWGS implementation is missing"
    return importlib.import_module("ralup.baselines.ewgs")


def test_detached_leaf_exhaustive_trace_population_std_and_hard_rounding_negative_control():
    m=ewgs();h=torch.tensor([[3.,1.],[1.,5.]],dtype=torch.float64)
    shadow=torch.tensor([.2,.8],dtype=torch.float64,requires_grad=True);code=torch.round(shadow*3)/3
    probes=torch.tensor(list(itertools.product([-1.,1.],repeat=2)),dtype=torch.float64)
    seen=[]
    def loss(leaf):
        seen.append((leaf.is_leaf,leaf.grad_fn,leaf.data_ptr()!=code.data_ptr()))
        return .5*leaf@h@leaf
    result=m.estimate_delta(loss,code,probes)
    assert all(a and b is None and c for a,b,c in seen)
    assert result['estimated_trace']==pytest.approx(torch.trace(h).item())
    assert result['hvp_backward_passes']==4 and result['extra_backward_passes']==5
    assert result['zero_hessian_inferred'] is False
    g=h@code.detach();assert result['delta']==pytest.approx(4/(3*g.std(correction=0).item()+1e-12))
    assert result['delta']>=0 and shadow.grad is None
    wrong_g=torch.autograd.grad(.5*code@h@code,shadow,create_graph=True)[0]
    assert not wrong_g.requires_grad
    with pytest.raises(RuntimeError,match="does not require grad"):
        torch.autograd.grad(wrong_g,shadow,torch.ones_like(shadow))
    wrong_hvp=torch.zeros_like(shadow) # hard-rounding Hessian convention is zero
    with pytest.raises(AssertionError):torch.testing.assert_close(wrong_hvp,h@torch.ones_like(shadow))
    assert wrong_hvp.abs().sum()==0


def test_normalized_physical_chain_applied_once():
    m=ewgs();q=torch.tensor([-.5,.2]);p=torch.tensor([[1.,1.],[-1.,1.]])
    scale=torch.tensor([2.,3.]);a=6*scale
    result=m.estimate_delta(lambda leaf:.5*(leaf*a).square().sum(),q,p)
    g=a.square()*q
    assert result['estimated_trace']==pytest.approx(a.square().sum().item())
    assert result['delta']==pytest.approx(a.square().mean().item()/(3*g.std(correction=0).item()+1e-12))


def test_fixed_shared_training_snapshot_resume_and_no_extra_passes():
    m=ewgs();state,batch=fixture(m.FIXTURE_CONFIG)
    from ralup.train import train_step
    from ralup.state import capture_snapshot,restore_snapshot
    result=train_step(state,batch)
    assert result.work['additional_loss']==0 and result.work['backward']==batch.target_count
    restored=restore_snapshot(capture_snapshot(state),'cpu')
    train_step(state,batch);train_step(restored,batch)
    for name,p in state.model.named_parameters():torch.testing.assert_close(p,dict(restored.model.named_parameters())[name],atol=0,rtol=0)
    assert restored.method.config==state.method.config
    assert torch.equal(restored.method.probe_rng_state,state.method.probe_rng_state)


def fixture(config):
    m=ewgs()
    import hashlib
    from ralup.data.articles import Article
    from ralup.data.blocks import tile_article
    from ralup.data.batches import LogicalBatch
    from ralup.models.transformer import ByteTransformer,ModelConfig
    from ralup.optim.base import default_parameter_groups
    from ralup.train import TrainingState
    torch.manual_seed(303);model=ByteTransformer(ModelConfig(1,8,2))
    model.configure_quantization({'name':'row_dynamic_max','backward':'ewgs'})
    method=m.EwgsAdapter(config).bind(dict(model.named_parameters()),default_parameter_groups(model))
    raw=b'abcdefg';blocks=tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))
    return TrainingState(model,method,microbatch=1),LogicalBatch(blocks,len(raw))


def test_adaptive_logical_mean_pass_events_and_cadence():
    m=ewgs();state,batch=fixture({**m.FIXTURE_CONFIG,'mode':'adaptive','pilot_relevance':True,'operation_support':'pass','probes':2,'cadence':2})
    from ralup.train import train_step
    first=train_step(state,batch)
    assert len(first.work['extra_pass_events'])==6*4 # forward, gradient, two HVPs per matrix
    assert first.work['additional_loss']==6*batch.target_count
    assert first.work['backward']==(1+6*3)*batch.target_count
    assert first.work['ewgs_work']['hvp_target_bytes']==6*2*batch.target_count
    assert first.work['ewgs_work']['gradient_only_target_bytes']==0
    assert first.work['ewgs_work']['loss_backward_target_bytes']==6*batch.target_count
    events=first.work['extra_pass_events']
    assert events[0]['execution_id']==events[1]['execution_id']
    assert len({e['execution_id'] for e in events})==6*3
    assert first.work['loss_targets']==7*batch.target_count
    second=train_step(state,batch)
    assert second.work['additional_loss']==0 and state.method.last_refresh_step==0


def test_adaptive_resume_scientific_hash_excludes_retained_timing():
    m=ewgs();state,batch=fixture({**m.FIXTURE_CONFIG,'mode':'adaptive','pilot_relevance':True,'operation_support':'pass'})
    from ralup.train import train_step
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    train_step(state,batch);restored=restore_snapshot(capture_snapshot(state),'cpu')
    train_step(state,batch);train_step(restored,batch)
    train_step(state,batch);train_step(restored,batch) # crosses the declared cadence
    assert exact_cpu_state_hash(state)==exact_cpu_state_hash(restored)
    assert state.ledger.events[0]['ewgs_work']['seconds']>0


def test_unknown_or_forged_estimator_state_rejects():
    m=ewgs();state,batch=fixture({**m.FIXTURE_CONFIG,'mode':'adaptive','pilot_relevance':True,'operation_support':'pass'})
    from ralup.train import train_step
    from ralup.state import capture_snapshot,IntegrityError
    train_step(state,batch)
    first=next(iter(state.method.refresh_telemetry))
    state.method.refresh_telemetry[first]['elements']=1
    with pytest.raises(IntegrityError):capture_snapshot(state)


def test_failure_preserves_launched_probe_work_and_rolls_back_rng():
    m=ewgs();state,batch=fixture({**m.FIXTURE_CONFIG,'mode':'adaptive','pilot_relevance':True,'operation_support':'pass'})
    from ralup.train import train_step,TrainingFailure
    rng=state.method.probe_rng_state.clone()
    # Concrete unsupported higher-order loss path; retain all actually launched work.
    state.model.forward=lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError('unsupported fixture forward'))
    with pytest.raises(TrainingFailure):train_step(state,batch)
    event=state.ledger.events[-1]
    assert event['status']=='failed' and event['additional_loss']==batch.target_count
    assert event['backward']==0 and event['loss_targets']==batch.target_count
    assert len(event['extra_pass_events'])==1 and event['ewgs_work']['seconds']>0
    assert torch.equal(rng,state.method.probe_rng_state)


def test_actual_mps_fixed_and_conditional_neural_higher_order_gate():
    import os,json
    if not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    m=ewgs();state,batch=fixture(m.FIXTURE_CONFIG);state.model.to('mps')
    from ralup.train import train_step,TrainingFailure
    from ralup.state import capture_snapshot,restore_snapshot
    first=train_step(state,batch);restored=restore_snapshot(capture_snapshot(state),'mps')
    train_step(state,batch);train_step(restored,batch)
    for name,p in state.model.named_parameters():torch.testing.assert_close(p,dict(restored.model.named_parameters())[name],atol=0,rtol=0)
    adaptive,batch=fixture({**m.FIXTURE_CONFIG,'mode':'adaptive','pilot_relevance':True,'operation_support':'pass'});adaptive.model.to('mps')
    try:
        result=train_step(adaptive,batch)
        print(json.dumps({'actual_mps_neural_higher_order':'pass','work':result.work}))
    except TrainingFailure as error:
        cause=error.__cause__
        assert isinstance(cause,RuntimeError) and any(s in str(cause).lower() for s in ('not implemented','not supported','does not support'))
        print(json.dumps({'actual_mps_neural_higher_order':'blocked','reason':str(cause),'work':adaptive.ledger.events[-1]}))


def test_cli_ewgs_method_and_fixture_suite(tmp_path,capsys):
    import json
    from pathlib import Path
    from ralup.__main__ import main
    from ralup.state import load_snapshot,restore_snapshot
    from ralup.records import ArtifactRef
    m=ewgs()
    template=json.loads((Path(__file__).parents[2]/'configs/smoke.json').read_bytes())
    envelope={'config':template,'purpose':'smoke','pilot_target_budget':7,'seed':303,
        'output':str(tmp_path/'run'),'checkpoint_directory':str(tmp_path/'checkpoint'),
        'method':{'name':'ewgs_fp4','adaptation_label':'project_FP4_population_std_detached_normalized_leaf','config':m.FIXTURE_CONFIG,'learning_rate':.001}}
    path=tmp_path/'ewgs.json';path.write_text(json.dumps(envelope))
    assert main(['run','--phase','explore','--config',str(path)])==0
    result=json.loads(capsys.readouterr().out)
    restored=restore_snapshot(load_snapshot(ArtifactRef(**result['checkpoint'])),'cpu')
    assert type(restored.method) is m.EwgsAdapter and restored.method.config==m.FIXTURE_CONFIG
    assert restored.model._recipe['backward']=='ewgs'
    suite=tmp_path/'suite.json'
    assert main(['verify-baselines','--suite','ewgs','--output',str(suite)])==0
    report=json.loads(suite.read_text())
    assert report['status']=='pass'
    assert report['variants'][1]['fp4_dimensionless_residual_clipping']=='pass'
    assert report['variants'][1]['analytic_hessian_agreement']=='pass'


def test_affine_hvp_accounting_and_trusted_v2_reconstruction(monkeypatch):
    m=ewgs();state,batch=fixture({**m.FIXTURE_CONFIG,'mode':'adaptive','pilot_relevance':True,'operation_support':'pass'})
    from ralup import evaluate
    from ralup.train import train_step
    from ralup.state import capture_snapshot,restore_snapshot,IntegrityError
    # Scoped affine scorer fixture: no Transformer forward/neural wave executes.
    # Estimator substitutes its real normalized leaf -> detached physical scale.
    def affine_target_losses(model,batch,*,contexts=None,parameter_overrides=None):
        assert parameter_overrides is None
        if contexts is not None:
            leaf_view=next(c for c in contexts.values() if type(c) is m.NormalizedLeafView)
            return (100+leaf_view.storage_values.sum()).reshape(1)
        return (100+sum(p.sum() for p in model.parameters())).reshape(1)
    monkeypatch.setattr(evaluate,'_target_losses',affine_target_losses)
    result=train_step(state,batch)
    work=result.work;events=work['extra_pass_events']
    assert work['additional_loss']==6*batch.target_count
    assert work['backward']==7*batch.target_count # ordinary plus six extra-loss backwards
    assert work['loss_targets']==7*batch.target_count
    assert len(events)==12 and len({e['execution_id'] for e in events})==6
    assert all(e['kind'] in ('extra_loss','extra_loss_backward') for e in events)
    assert work['ewgs_work']['hvp_target_bytes']==0
    assert work['ewgs_work']['gradient_only_target_bytes']==0
    assert work['ewgs_work']['loss_backward_target_bytes']==6*batch.target_count
    for telemetry in state.method.refresh_telemetry.values():
        assert telemetry['probes']==2 and telemetry['hvp_backward_passes']==0
        assert telemetry['extra_backward_passes']==1 and telemetry['zero_hessian_inferred'] is True
        assert telemetry['estimated_trace']==0 and telemetry['delta']==0
    snapshot=capture_snapshot(state);restored=restore_snapshot(snapshot,'cpu')
    assert restored.method.snapshot_schema['adapter_mutable_state']=='ewgs.normalized_leaf_probes_cadence.v2'
    assert restored.method.refresh_telemetry==state.method.refresh_telemetry
    # Both representations of actual launches must agree with the inferred branch.
    name=next(iter(restored.method.refresh_telemetry));telemetry=restored.method.refresh_telemetry[name]
    telemetry['hvp_backward_passes']=2;telemetry['extra_backward_passes']=3
    with pytest.raises(IntegrityError):capture_snapshot(restored)
    telemetry['hvp_backward_passes']=0;telemetry['extra_backward_passes']=1
    telemetry['zero_hessian_inferred']=False
    with pytest.raises(IntegrityError):capture_snapshot(restored)
