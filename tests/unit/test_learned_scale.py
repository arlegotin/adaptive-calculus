import importlib
import math
import pytest
import torch


def module():
    assert importlib.util.find_spec('ralup.baselines.learned_scale'), 'M06 learned-scale implementation missing'
    return importlib.import_module('ralup.baselines.learned_scale')


@pytest.mark.parametrize('format,maximum,expected',[
    ('uniform',3,[-3.,.4,0.,-.4,3.]),
    ('fp4',6,[-6.,.4,0.,-.4,6.]),
])
def test_analytic_surrogate_inside_outside_log_chain_normalizer(format,maximum,expected):
    m=module();s=2.;w=torch.tensor([[-20.,-1.8,0.,1.8,20.]],requires_grad=True)
    log=torch.tensor([[math.log(s)]],requires_grad=True)
    c={**m.VIEW_CONTRACT,'format':format,'minimum':-maximum,'maximum':maximum,'grouping':'tensor'}
    y=m.learned_scale_view(w,log,c);y.sum().backward()
    declared_scale_surrogate_grad=torch.tensor([[sum(expected)/math.sqrt(5*maximum)]])
    torch.testing.assert_close(log.grad,declared_scale_surrogate_grad*s)
    torch.testing.assert_close(w.grad,torch.tensor([[0.,1.,1.,1.,0.]]))
    # Asymmetry makes the scale factor and chain rule observable.
    w=torch.tensor([[-20.,-1.8,.3,1.8,10.]],requires_grad=True)
    log=torch.tensor([[math.log(s)]],requires_grad=True)
    upstream=torch.tensor([[1.,2.,3.,4.,5.]])
    y=m.learned_scale_view(w,log,c);(y*upstream).sum().backward()
    x=w.detach()/s;code=y.detach()/s
    local=torch.where(x<c['minimum'],float(c['minimum']),torch.where(x>c['maximum'],float(c['maximum']),code-x))
    declared=(upstream*local).sum().reshape(1,1)/math.sqrt(5*maximum)
    torch.testing.assert_close(log.grad,declared*s)
    assert not torch.allclose(log.grad,(upstream*code).sum().reshape(1,1)*s), 'hard quantizer finite difference is not the surrogate target'


def test_row_group_counts_and_physical_forward():
    m=module();w=torch.tensor([[1.8,.3],[-3.2,1.2]],requires_grad=True)
    log=torch.tensor([[math.log(2.)],[math.log(.5)]],requires_grad=True)
    y=m.learned_scale_view(w,log,m.VIEW_CONTRACT);y.sum().backward()
    torch.testing.assert_close(y,torch.tensor([[2.,0.],[-3.,1.]]))
    expected=torch.tensor([[(-.05)*2/math.sqrt(12)],[(-6-.4)*.5/math.sqrt(12)]])
    torch.testing.assert_close(log.grad,expected)


@pytest.mark.parametrize('format,low,high',[('uniform',-8,7),('fp4',-6,6)])
def test_exact_clip_endpoints_use_saturated_source_surrogate(format,low,high):
    m=module();w=torch.tensor([[float(low),float(high)]],requires_grad=True);p=torch.zeros(1,1,requires_grad=True)
    c={**m.VIEW_CONTRACT,'format':format,'minimum':low,'maximum':high,'grouping':'tensor'}
    m.learned_scale_view(w,p,c).sum().backward()
    torch.testing.assert_close(p.grad,torch.tensor([[(low+high)/math.sqrt(2*high)]]))
    assert torch.equal(w.grad,torch.zeros_like(w))


def test_contract_rejects_undeclared_range_and_geometry():
    m=module()
    with pytest.raises(ValueError):m.learned_scale_view(torch.ones(2,2),torch.zeros(1,1),m.VIEW_CONTRACT)
    with pytest.raises(ValueError):m.learned_scale_view(torch.ones(2,2),torch.zeros(2,1),{**m.VIEW_CONTRACT,'maximum':7})
    with pytest.raises(ValueError,match='full FP4 range'):
        m.validate_config({**m.FIXTURE_CONFIG,'view':{**m.VIEW_CONTRACT,'minimum':-4,'maximum':4}})


def fixture(grouping='row',config=None):
    import hashlib
    from ralup.models.transformer import ByteTransformer,ModelConfig
    from ralup.optim.base import default_parameter_groups,SchedulePoint
    from ralup.train import TrainingState
    from ralup.data.articles import Article
    from ralup.data.blocks import tile_article
    from ralup.data.batches import LogicalBatch
    with torch.random.fork_rng():
        torch.manual_seed(606);model=ByteTransformer(ModelConfig(1,8,2))
    m=module();c={**m.FIXTURE_CONFIG,'view':{**m.VIEW_CONTRACT,'grouping':grouping},**(config or {})}
    adapter=m.LearnedScaleAdapter(c).bind(dict(model.named_parameters()),default_parameter_groups(model))
    model.configure_quantization({'name':adapter.recipe,'rotation':'identity','backward':'ste'})
    raw=b'abcd';batch=LogicalBatch(tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest())),4)
    return TrainingState(model,adapter,schedule={'step':0,'point':SchedulePoint(.001,.1,0.)}),batch


@pytest.mark.parametrize('grouping',['row','tensor'])
def test_actual_scale_parameters_optimizer_export_evaluation_and_safe_replay(grouping):
    from ralup.train import train_step
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    from ralup.evaluate import score_blocks,evaluation_contexts
    from ralup.config import canonical_bytes
    state,batch=fixture(grouping);result=train_step(state,batch);method=state.method
    assert all(type(p) is torch.nn.Parameter for p in method.log_scales.values())
    assert all(s['step'].item()==1 for s in method.scale_optimizer_states.values())
    assert all(s['step'].item()==1 for s in method.optimizer_states.values())
    assert all(p.grad is method.scale_gradients[n] for n,p in method.log_scales.items())
    assert result.work['learned_scale_work']['extra_loss_passes']==0
    assert result.work['backward']==result.target_count==4 and result.work['loss_targets']==4
    export=method.export_scales();contexts=evaluation_contexts(state.model,{'partition':'fixture'})
    for n,s in export['scales'].items():
        assert s.dtype==torch.float32 and torch.equal(s,contexts[n].scales)
    before=exact_cpu_state_hash(state);score_blocks(state.model,batch.block_slices,evaluation_context={'partition':'fixture'})
    assert exact_cpu_state_hash(state)==before
    canonical_bytes(method.metadata)
    resumed=restore_snapshot(capture_snapshot(state),'cpu')
    assert all(type(p) is torch.nn.Parameter and p.requires_grad for p in resumed.method.log_scales.values())
    assert all(p is not method.log_scales[n] for n,p in resumed.method.log_scales.items())
    train_step(state,batch);train_step(resumed,batch)
    assert exact_cpu_state_hash(state)==exact_cpu_state_hash(resumed)


def test_matching_fixed_arm_consumes_actual_exported_fp32_scales():
    from ralup.train import train_step
    from ralup.models.transformer import ByteTransformer
    from ralup.evaluate import evaluation_contexts,score_blocks
    state,batch=fixture();train_step(state,batch);export=state.method.export_scales()
    fixed=ByteTransformer(state.model.config);fixed.load_state_dict(state.model.state_dict())
    runtime={n:{'previous_scales':s.clone(),'previous_zero_groups':state.model._runtime[n]['previous_zero_groups'].clone()} for n,s in export['scales'].items()}
    fixed.configure_quantization({'name':export['contract']['quantizer_recipe'],'rotation':'identity','backward':'ste'},runtime=runtime)
    learned=evaluation_contexts(state.model,{'partition':'fixture'});matched=evaluation_contexts(fixed,{'partition':'fixture'})
    for n in matched:
        assert torch.equal(matched[n].scales,learned[n].scales) and torch.equal(matched[n].storage_values,learned[n].storage_values)
    assert score_blocks(fixed,batch.block_slices,evaluation_context={'partition':'fixture'})==score_blocks(state.model,batch.block_slices,evaluation_context={'partition':'fixture'})


def test_scale_lr_independent_and_clipping_range_owned():
    from ralup.train import train_step
    a,batch=fixture(config={'scale_lr':0.});b,_=fixture(config={'scale_lr':.2,'log_scale_range':[-6.,-5.],'scale_clip_norm':.0001})
    a.method.prepare_contexts(a,0);initial={n:p.detach().clone() for n,p in a.method.log_scales.items()}
    train_step(a,batch);r=train_step(b,batch)
    for n,p in a.method.log_scales.items():assert torch.equal(p,initial[n])
    for p in b.method.log_scales.values():assert bool(((p>=-6)&(p<=-5)).all())
    norm=torch.stack([p.grad.square().sum() for p in b.method.log_scales.values()]).sum().sqrt()
    assert float(norm)<=.000101
    assert r.telemetry['learned_scale_contract']['scale_lr']==.2


def test_failed_scale_commit_rolls_back_every_owner_parameter_identity(monkeypatch):
    from ralup.train import train_step,TrainingFailure
    from ralup.state import capture_snapshot
    state,batch=fixture();train_step(state,batch)
    parameters=dict(state.method.log_scales);values={n:p.detach().clone() for n,p in parameters.items()};grads={n:p.grad.clone() for n,p in parameters.items()}
    original=state.method.finish_step
    def fail(state_,schedule):
        original(state_,schedule);raise RuntimeError('injected after learned scale and W commit')
    monkeypatch.setattr(type(state.method),'finish_step',lambda self,s,p:fail(s,p))
    with pytest.raises(TrainingFailure):train_step(state,batch)
    for n,p in parameters.items():
        assert state.method.log_scales[n] is p
        assert torch.equal(p,values[n]) and torch.equal(p.grad,grads[n])
        assert state.method.scale_optimizer_states[n]['step'].item()==1
    assert state.schedule['step']==1 and state.ledger.events[-1]['status']=='failed'
    with pytest.raises(ValueError):capture_snapshot(state)


@pytest.mark.parametrize('mutation',['replace','remove','add'])
def test_successful_scale_owner_mutation_is_rejected_and_rolled_back(monkeypatch,mutation):
    from ralup.train import train_step,TrainingFailure
    state,batch=fixture();train_step(state,batch)
    method=state.method;owners=dict(method.log_scales)
    values={n:p.detach().clone() for n,p in owners.items()}
    gradients={n:p.grad.clone() for n,p in owners.items()}
    weights={n:p.detach().clone() for n,p in state.model.named_parameters()}
    original=type(method).finish_step
    def mutate(self,state_,schedule):
        result=original(self,state_,schedule)
        name=next(iter(self.log_scales));p=self.log_scales[name]
        if mutation=='replace':
            replacement=torch.nn.Parameter(p.detach().clone())
            replacement.grad=p.grad.clone()
            self.log_scales[name]=replacement;self.scale_gradients[name]=replacement.grad
        elif mutation=='remove':del self.log_scales[name]
        else:self.log_scales['extra']=torch.nn.Parameter(p.detach().clone())
        return result
    monkeypatch.setattr(type(method),'finish_step',mutate)
    with pytest.raises(TrainingFailure,match='method-owned Parameter inventory or identity'):
        train_step(state,batch)
    assert set(method.log_scales)==set(owners)
    for n,p in owners.items():
        assert method.log_scales[n] is p
        assert torch.equal(p,values[n]) and torch.equal(p.grad,gradients[n])
        assert method.scale_optimizer_states[n]['step'].item()==1
    for n,p in state.model.named_parameters():assert torch.equal(p,weights[n])
    assert state.schedule['step']==1 and state.failed and not state.checkpointable
    event=state.ledger.events[-1]
    assert event['status']=='failed' and event['stream']==0
    assert event['backward']==event['loss_targets']==batch.target_count


def test_first_step_creates_scale_owners_then_preserves_their_identity():
    from ralup.train import train_step
    state,batch=fixture();assert state.method.transaction_parameters()=={}
    train_step(state,batch);owners=dict(state.method.transaction_parameters())
    assert set(owners)==set(state.method.groups[0].parameter_names)
    train_step(state,batch)
    for n,p in owners.items():assert state.method.log_scales[n] is p


def test_registry_public_cli_and_information_contract(tmp_path):
    from ralup.baselines.registry import default_registry
    from ralup.optim.base import ParameterGroup
    from ralup.__main__ import main
    registry=default_registry((ParameterGroup('internal_projection',('w',)),))
    record=registry.require('learned_fp4_scale',adaptation_label='project_FP4_log_scale_original_W')
    assert record.artifact_kind=='project_adaptation' and not registry.claim_eligibility(record.name)['eligible']
    assert 'arXiv:1902.08153v3' in record.source_revision
    assert main(['verify-baselines','--suite','scale-noise','--device','cpu','--output',str(tmp_path/'report.json')])==0


@pytest.mark.parametrize('name',['learned_fp4_scale','magma_matrix_mask'])
def test_public_run_cli_binds_real_control_and_trusted_checkpoint(tmp_path,capsys,name):
    import json
    from pathlib import Path
    from ralup.__main__ import main
    from ralup.baselines.registry import scale_noise_record
    from ralup.baselines.noise import FIXTURE_CONFIG as NOISE_CONFIG
    from ralup.optim.base import ParameterGroup
    from ralup.records import ArtifactRef
    from ralup.state import load_snapshot,restore_snapshot
    config=module().FIXTURE_CONFIG if name=='learned_fp4_scale' else NOISE_CONFIG
    record=scale_noise_record(name,(ParameterGroup('internal_projection',('w',)),),config)
    envelope={'config':json.loads(Path('configs/smoke.json').read_bytes()),'purpose':'smoke','pilot_target_budget':3,'seed':606,
              'output':str(tmp_path/'run'),'checkpoint_directory':str(tmp_path/'checkpoints'),
              'method':{'name':name,'adaptation_label':record.adaptation_label,'config':config,'learning_rate':.001}}
    path=tmp_path/'config.json';path.write_text(json.dumps(envelope))
    assert main(['run','--phase','explore','--config',str(path)])==0
    output=json.loads(capsys.readouterr().out);state=restore_snapshot(load_snapshot(ArtifactRef(**output['checkpoint'])),'cpu')
    assert state.schedule['step']==1
    assert state.method.log_scales if name=='learned_fp4_scale' else state.method.mask_states


def test_fresh_interpreter_trusted_scale_registration(tmp_path):
    import json,subprocess,sys
    from pathlib import Path
    from ralup.train import train_step
    from ralup.state import capture_snapshot,save_snapshot,exact_cpu_state_hash
    state,batch=fixture();train_step(state,batch);ref=save_snapshot(capture_snapshot(state),tmp_path/'snapshot')
    train_step(state,batch);expected=exact_cpu_state_hash(state)
    process=subprocess.run([sys.executable,str(Path('docs/harness/m06-replay.py').resolve()),str(ref.path),ref.sha256,'cpu'],capture_output=True,text=True)
    (tmp_path/'native-stdout.txt').write_text(process.stdout);(tmp_path/'native-stderr.txt').write_text(process.stderr)
    # Persist actual native child streams through the parent attempt recorder.
    print(process.stdout,end='');print(process.stderr,end='',file=sys.stderr)
    assert process.returncode==0 and not process.stderr
    assert json.loads(process.stdout)=={'scientific_hash':expected,'step':2}


def test_cpu_artifact_validation_and_exact_backend_restore_are_distinct():
    from ralup.train import train_step
    from ralup.state import capture_snapshot,restore_snapshot,IntegrityError
    state,batch=fixture();train_step(state,batch);snapshot=capture_snapshot(state)
    entry=snapshot.metadata['items']['method']['items'];entry['scale_backend']='mps'
    with pytest.raises(IntegrityError,match='cross-backend'):restore_snapshot(snapshot,'cpu')
    snapshot=capture_snapshot(state);entry=snapshot.metadata['items']['method']['items']
    key=next(iter(entry['log_scales']['items'].values()))['key'];snapshot.tensors[key].add_(.01)
    with pytest.raises(IntegrityError,match='declared-backend'):restore_snapshot(snapshot,'cpu')


@pytest.mark.parametrize('damage',['missing_gradient','bad_shape','negative_variance','stale_runtime','unknown_owner','hook','tensor_owner','log_mutation','cache_mutation'])
def test_trusted_learned_owners_reject_incomplete_or_unsafe_state(damage):
    from ralup.train import train_step
    from ralup.state import capture_snapshot,IntegrityError
    state,batch=fixture();train_step(state,batch);m=state.method;n=next(iter(m.log_scales));p=m.log_scales[n]
    if damage=='missing_gradient':p.grad=None;m.scale_gradients[n]=None
    if damage=='bad_shape':m.scale_optimizer_states[n]['exp_avg']=torch.zeros(1)
    if damage=='negative_variance':m.scale_optimizer_states[n]['exp_avg_sq'].fill_(-1)
    if damage=='stale_runtime':state.model._runtime[n]['previous_scales'].mul_(2)
    if damage=='unknown_owner':m.hidden_optimizer={}
    if damage=='hook':p.register_hook(lambda g:g)
    if damage=='tensor_owner':m.log_scales[n]=p.detach()
    if damage=='log_mutation':
        with torch.no_grad():p.add_(.01)
    if damage=='cache_mutation':m.physical_scales[n].mul_(2)
    with pytest.raises(IntegrityError):capture_snapshot(state)


@pytest.mark.parametrize('grouping',['row','tensor'])
def test_logical_sum_normalization_scale_gradients_and_damping_packing(grouping):
    import hashlib
    from ralup.train import train_step
    from ralup.data.articles import Article
    from ralup.data.blocks import tile_article
    from ralup.data.batches import LogicalBatch
    blocks=[b for raw in (b'abcd',b'efgh') for b in tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))]
    batch=LogicalBatch(blocks,8);a,_=fixture(grouping);b,_=fixture(grouping);b.microbatch=2
    train_step(a,batch);train_step(b,batch)
    for n,p in a.method.log_scales.items():
        torch.testing.assert_close(p.grad,b.method.log_scales[n].grad,atol=2e-7,rtol=2e-5)
        torch.testing.assert_close(p,b.method.log_scales[n],atol=2e-7,rtol=2e-5)


@pytest.mark.parametrize('name',['muon_project_fp32','muon_plus_fp32'])
def test_explicit_original_w_backbones_and_scale_replay(name):
    from ralup.baselines.trs import backbone_config,BACKBONES
    from ralup.train import train_step
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    config=backbone_config(BACKBONES[name]({}));state,batch=fixture(config={'backbone':config})
    train_step(state,batch);restored=restore_snapshot(capture_snapshot(state),'cpu')
    for n in state.method.groups[0].parameter_names:
        assert state.method.optimizer_states[n]['momentum'].shape==state.method.parameters[n].shape
    train_step(state,batch);train_step(restored,batch)
    assert exact_cpu_state_hash(state)==exact_cpu_state_hash(restored)
