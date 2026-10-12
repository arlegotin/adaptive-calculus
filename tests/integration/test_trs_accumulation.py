import hashlib,importlib,json,os,subprocess,sys
from pathlib import Path
import pytest
import torch
from ralup.train import TrainingState,train_step,TrainingFailure
from ralup.optim.base import default_parameter_groups,SchedulePoint
from ralup.optim.adamw import AdamWUpdateRule
from ralup.models.transformer import ByteTransformer,ModelConfig
from ralup.data.articles import Article
from ralup.data.blocks import tile_article
from ralup.data.batches import LogicalBatch


def trs():
    assert importlib.util.find_spec('ralup.baselines.trs') is not None,'M04 TRS implementation missing'
    return importlib.import_module('ralup.baselines.trs')


def fixture(microbatch=1,device='cpu',rule=None):
    m=trs();torch.manual_seed(404);model=ByteTransformer(ModelConfig(1,8,2)).to(device)
    model.configure_quantization({'name':'row_fixed_max'})
    base=rule or AdamWUpdateRule();config={**m.FIXTURE_CONFIG,'backbone':m.backbone_config(base)}
    method=m.TrsBackboneAdapter(base,config).bind(dict(model.named_parameters()),default_parameter_groups(model))
    blocks=[b for raw in (b'abcd',b'efgh',b'ijkl',b'mnop') for b in tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))]
    return TrainingState(model,method,microbatch=microbatch,schedule={'step':0,'point':SchedulePoint(.001,.1,0.)}),LogicalBatch(blocks,16)


def test_one_feedback_commit_per_logical_batch_and_validation_read_only():
    m=trs();one,batch=fixture(1);four,_=fixture(4)
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    from ralup.evaluate import score_blocks
    for step in range(3):
        a=train_step(one,batch);b=train_step(four,batch)
        assert one.histories['loss_updates']==four.histories['loss_updates']==step+1
        for name,s in one.method.trs_states.items():
            peer=four.method.trs_states[name]
            assert s.observations==peer.observations==step+1
            assert a.telemetry['trs'][name]['effective_lr']==b.telemetry['trs'][name]['effective_lr']
            assert s.r==peer.r and s.K==peer.K and torch.equal(s.previous_codes,peer.previous_codes)
        for name,p in one.model.named_parameters():torch.testing.assert_close(p,dict(four.model.named_parameters())[name],atol=3e-7,rtol=3e-5)
    before=exact_cpu_state_hash(one);score_blocks(one.model,batch.block_slices,evaluation_context={'partition':'fixture'})
    assert exact_cpu_state_hash(one)==before
    resumed=restore_snapshot(capture_snapshot(one),'cpu');train_step(one,batch);train_step(resumed,batch)
    assert exact_cpu_state_hash(one)==exact_cpu_state_hash(resumed)


def test_proposals_do_not_advance_feedback_and_decay_is_separate():
    state,batch=fixture();state.model.prepare_quantized_views(0,None);state.method.prepare_step(state,batch)
    for p in state.model.parameters():p.grad=torch.ones_like(p)
    proposals=state.method.collect_proposals(state.schedule['point'])
    assert not state.method.trs_states
    first=state.method.groups[0].parameter_names[0];p=state.method.parameters[first]
    torch.testing.assert_close(proposals[first].reference_after_decay,p.detach()*(1-.001*.1),atol=0,rtol=0)
    torch.testing.assert_close(proposals[first].new_optimizer_state['exp_avg'],torch.full_like(p,.1),atol=3e-8,rtol=1e-6)


def test_feedback_rolls_back_on_failure_after_commit(monkeypatch):
    state,batch=fixture();train_step(state,batch)
    from ralup.train import _clone,_equal
    before=_clone(state.method.trs_states);history=_clone(state.histories)
    import ralup.train as train
    monkeypatch.setattr(train,'_transition',lambda *args:(_ for _ in ()).throw(ValueError('after feedback')))
    with pytest.raises(TrainingFailure,match='after feedback'):train_step(state,batch)
    assert _equal(history,state.histories)
    for name,s in before.items():
        now=state.method.trs_states[name];assert (s.r,s.K,s.observations)==(now.r,now.K,now.observations)
        assert torch.equal(s.previous_codes,now.previous_codes)
    assert state.ledger.events[-1]['status']=='failed'


def test_complete_trusted_state_rejects_missing_forged_or_executable_owners():
    state,batch=fixture();train_step(state,batch)
    from ralup.state import capture_snapshot,IntegrityError,restore_snapshot
    for field in ('r','K','observations','previous_codes'):
        resumed=restore_snapshot(capture_snapshot(state),'cpu');name=next(iter(resumed.method.trs_states));s=resumed.method.trs_states[name]
        if field=='observations':s.observations=5
        elif field=='previous_codes':s.previous_codes=torch.zeros(1)
        else:setattr(s,field,float('nan'))
        with pytest.raises(IntegrityError):capture_snapshot(resumed)
    del state.method.trs_states[next(iter(state.method.trs_states))]
    with pytest.raises(IntegrityError):capture_snapshot(state)
    state,batch=fixture();state.method.extra=lambda:None
    with pytest.raises(IntegrityError):capture_snapshot(state)


def test_matrix_backbone_original_W_moments_and_safe_reconstruction():
    from ralup.optim.muon import MuonUpdateRule
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    state,batch=fixture(rule=MuonUpdateRule({}))
    from ralup.config import canonical_bytes
    m=trs();public=json.loads(canonical_bytes(state.method.metadata))
    assert public['trs_configuration']['backbone']['config']['auxiliary_betas']==[.9,.99]
    rebuilt=m.construct_backbone(json.loads(json.dumps(m.backbone_config(state.method.rule))))
    assert type(rebuilt) is MuonUpdateRule
    train_step(state,batch)
    resumed=restore_snapshot(capture_snapshot(state),'cpu');train_step(state,batch);train_step(resumed,batch)
    assert exact_cpu_state_hash(state)==exact_cpu_state_hash(resumed)
    for name in state.method.groups[0].parameter_names:assert state.method.optimizer_states[name]['momentum'].shape==state.method.parameters[name].shape


def test_cli_trs_config_target_schedule_and_public_canonical_json(tmp_path,capsys):
    m=trs()
    from ralup.__main__ import main
    from ralup.state import load_snapshot,restore_snapshot
    from ralup.records import ArtifactRef
    template=json.loads((Path(__file__).parents[2]/'configs/smoke.json').read_bytes())
    envelope={'config':template,'purpose':'smoke','pilot_target_budget':7,'seed':404,'output':str(tmp_path/'run'),'checkpoint_directory':str(tmp_path/'checkpoint'),
        'method':{'name':'trs_backbone_adapter','adaptation_label':'project_FP4_logical_step_decoupled_backbone','config':m.FIXTURE_CONFIG,'learning_rate':.001}}
    path=tmp_path/'config.json';path.write_text(json.dumps(envelope))
    assert main(['run','--phase','explore','--config',str(path)])==0
    result=json.loads(capsys.readouterr().out);state=restore_snapshot(load_snapshot(ArtifactRef(**result['checkpoint'])),'cpu')
    assert type(state.method) is m.TrsBackboneAdapter and state.method.config==m.FIXTURE_CONFIG
    assert state.method.trs_states and state.schedule['point'].lr==.001
    assert state.method.metadata['schedule_fields']==['lr','weight_decay','progress','target_rate']


def test_fresh_interpreter_trusted_feedback_reconstruction(tmp_path):
    from ralup.state import capture_snapshot,restore_snapshot,save_snapshot,exact_cpu_state_hash
    state,batch=fixture();train_step(state,batch)
    ref=save_snapshot(capture_snapshot(state),tmp_path/'snapshot')
    train_step(state,batch)
    script='''import sys
sys.path.insert(0,'tests/integration')
from test_trs_accumulation import fixture
from ralup.state import restore_snapshot,load_snapshot,exact_cpu_state_hash
from ralup.records import ArtifactRef
from ralup.train import train_step
_,batch=fixture()
state=restore_snapshot(load_snapshot(ArtifactRef(sys.argv[1],sys.argv[2])),'cpu')
train_step(state,batch)
print(exact_cpu_state_hash(state))
'''
    result=subprocess.run([sys.executable,'-c',script,str(ref.path),ref.sha256],capture_output=True,text=True,check=True)
    assert result.stderr=='' and result.stdout.strip()==exact_cpu_state_hash(state)


@pytest.mark.native
def test_actual_mps_feedback_accumulation_resume():
    if not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    one,batch=fixture(1,'mps');four,_=fixture(4,'mps')
    for _ in range(3):
        train_step(one,batch);train_step(four,batch)
        for name,s in one.method.trs_states.items():
            peer=four.method.trs_states[name];assert s.K==peer.K and s.r==peer.r
            assert torch.equal(s.previous_codes,peer.previous_codes)
        for name,p in one.model.named_parameters():torch.testing.assert_close(p,dict(four.model.named_parameters())[name],atol=3e-7,rtol=3e-5)
    restored=restore_snapshot(capture_snapshot(one),'mps');train_step(one,batch);train_step(restored,batch)
    assert exact_cpu_state_hash(one)==exact_cpu_state_hash(restored)
