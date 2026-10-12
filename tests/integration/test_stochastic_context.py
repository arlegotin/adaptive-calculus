"""Reject shadow residuals/redraws; test shared transaction and trusted resume."""
import hashlib,importlib,json,os,subprocess,sys
from pathlib import Path
import pytest
import torch
from ralup.models.transformer import ByteTransformer,ModelConfig
from ralup.optim.base import default_parameter_groups,SchedulePoint
from ralup.train import TrainingState,train_step,TrainingFailure
from ralup.data.articles import Article
from ralup.data.blocks import tile_article
from ralup.data.batches import LogicalBatch

def module():
    assert importlib.util.find_spec('ralup.baselines.qar') is not None,'M05 QAR missing'
    return importlib.import_module('ralup.baselines.qar')

def fixture(name='qar_fp4_adjacent',packing=1,device='cpu',rotation='identity',base=None):
    q=module();torch.manual_seed(505);model=ByteTransformer(ModelConfig(1,8,2)).to(device)
    config={**q.FIXTURE_CONFIG,'rotation':rotation}
    if base is not None:config['backbone']=q.backbone_config(base)
    if name=='qar_fp4_adjacent':method=q.QarAdjacentAdapter(config)
    else:
        from ralup.baselines.rounding import RoundingAdapter
        method=RoundingAdapter({**config,'mode':'shadow_forward' if name=='stochastic_shadow_forward' else 'endpoint'})
    model.configure_quantization({'name':'row_fixed_max','rotation':rotation,'backward':'ste'})
    method.bind(dict(model.named_parameters()),default_parameter_groups(model))
    rng=torch.Generator().manual_seed(777)  # Device-independent named CPU draws, then explicit transfer.
    blocks=[b for raw in (b'abcd',b'efgh',b'ijkl',b'mnop') for b in tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))]
    return TrainingState(model,method,microbatch=packing,rng_streams={'quantizer':rng},schedule={'step':0,'point':SchedulePoint(.001,.1,0.)}),LogicalBatch(blocks,16)

@pytest.mark.parametrize('name',['qar_fp4_adjacent','stochastic_shadow_forward','stochastic_endpoint'])
def test_one_realization_microbatch_cache_and_deterministic_evaluation(name):
    state,batch=fixture(name);seen=[]
    original=state.method.backward_microbatch
    def observe(s,b):
        seen.append({n:c.codes.clone() for n,c in s.model._contexts.items()})
        return original(s,b)
    state.method.backward_microbatch=observe
    result=train_step(state,batch);del state.method.backward_microbatch
    assert len(seen)==4
    for n in seen[0]:assert all(torch.equal(seen[0][n],v[n]) for v in seen[1:])
    from ralup.state import exact_cpu_state_hash
    from ralup.evaluate import score_blocks,evaluation_contexts
    before=exact_cpu_state_hash(state);rng=state.rng_streams['quantizer'].get_state().clone();global_rng=torch.get_rng_state().clone()
    score_blocks(state.model,batch.block_slices,evaluation_context={'partition':'fixture'})
    assert before==exact_cpu_state_hash(state) and torch.equal(rng,state.rng_streams['quantizer'].get_state()) and torch.equal(global_rng,torch.get_rng_state())
    assert all(c.metadata['forward_realization']=='nearest_even' for c in evaluation_contexts(state.model,{'partition':'fixture'}).values())
    assert result.work['routing_rounding_work']['probability_elements']>0
    assert result.telemetry['routing_rounding']['persistent_shadow']==(name=='stochastic_shadow_forward')

@pytest.mark.parametrize('rotation',['identity','full_hadamard'])
def test_endpoint_authority_original_W_moments_adjacent_steps_and_resume(rotation):
    q=module();state,batch=fixture(rotation=rotation)
    from ralup.quantization.rotation import from_storage
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    for _ in range(3):
        old={n:v['codes'].clone() for n,v in state.method.endpoint_states.items()}
        result=train_step(state,batch)
        for n,v in state.method.endpoint_states.items():
            assert torch.equal(state.method.parameters[n],from_storage(q.reconstruct(v['codes'],v['scales']),rotation))
            if old:assert int((v['codes'].to(torch.int64)-old[n].to(torch.int64)).abs().max())<=1
            assert state.method.optimizer_states[n]['exp_avg'].shape==state.method.parameters[n].shape
        assert result.telemetry['routing_rounding']['clipped_probability_count']>=0
    resumed=restore_snapshot(capture_snapshot(state),'cpu');train_step(state,batch);train_step(resumed,batch)
    assert exact_cpu_state_hash(state)==exact_cpu_state_hash(resumed)

@pytest.mark.parametrize('name',['stochastic_shadow_forward','stochastic_endpoint'])
def test_rounding_controls_full_state_seed_replay_and_distinct_authority(name):
    state,batch=fixture(name);train_step(state,batch)
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    assert bool(state.method.endpoint_states)==(name=='stochastic_endpoint')
    resumed=restore_snapshot(capture_snapshot(state),'cpu');train_step(state,batch);train_step(resumed,batch)
    assert exact_cpu_state_hash(state)==exact_cpu_state_hash(resumed)

def test_endpoint_missing_rng_and_untracked_shadow_are_rejected():
    from ralup.state import capture_snapshot,IntegrityError,restore_snapshot
    state,batch=fixture();del state.rng_streams['quantizer']
    with pytest.raises(TrainingFailure,match='named.*quantizer'):train_step(state,batch)
    state,batch=fixture();train_step(state,batch)
    state.method.shadow_residual=torch.ones(1)
    with pytest.raises(IntegrityError):capture_snapshot(state)
    del state.method.shadow_residual
    state.method.parameters[state.method.groups[0].parameter_names[0]].data.add_(.00001)
    with pytest.raises(IntegrityError,match='endpoint'):capture_snapshot(state)

@pytest.mark.parametrize('fault',['codes','scales','observations','config','missing'])
def test_trusted_state_rejects_forged_or_missing_endpoint_owners(fault):
    state,batch=fixture();train_step(state,batch)
    from ralup.state import capture_snapshot,IntegrityError
    name=next(iter(state.method.endpoint_states));v=state.method.endpoint_states[name]
    if fault=='codes':v['codes'].fill_(255)
    if fault=='scales':v['scales'].mul_(2)
    if fault=='observations':state.method.observations+=1
    if fault=='config':state.method.config['gamma']=-1
    if fault=='missing':del state.method.endpoint_states[name]
    with pytest.raises(IntegrityError):capture_snapshot(state)

def test_failed_commit_rolls_back_endpoint_moments_and_named_rng(monkeypatch):
    state,batch=fixture();train_step(state,batch)
    from ralup.train import _clone,_equal
    before=_clone(state.method.endpoint_states);moments=_clone(state.method.optimizer_states);rng=state.rng_streams['quantizer'].get_state().clone()
    import ralup.train as t
    monkeypatch.setattr(t,'_transition',lambda *args:(_ for _ in ()).throw(ValueError('after endpoint')))
    with pytest.raises(TrainingFailure,match='after endpoint'):train_step(state,batch)
    assert _equal(state.method.endpoint_states,before) and _equal(state.method.optimizer_states,moments)
    assert torch.equal(rng,state.rng_streams['quantizer'].get_state()) and state.ledger.events[-1]['status']=='failed'

def test_matrix_backbone_moments_keep_original_W_and_reconstruct():
    from ralup.optim.muon_plus import MuonPlusUpdateRule
    state,batch=fixture(rotation='full_hadamard',base=MuonPlusUpdateRule({}))
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    from ralup.config import canonical_bytes
    json.loads(canonical_bytes(state.method.metadata));train_step(state,batch)
    resumed=restore_snapshot(capture_snapshot(state),'cpu');train_step(state,batch);train_step(resumed,batch)
    assert exact_cpu_state_hash(state)==exact_cpu_state_hash(resumed)
    assert all(state.method.optimizer_states[n]['momentum'].shape==state.method.parameters[n].shape for n in state.method.groups[0].parameter_names)

@pytest.mark.parametrize('name',['qar_fp4_adjacent','stochastic_shadow_forward','stochastic_endpoint'])
def test_cli_registration_config_and_complete_state(tmp_path,capsys,name):
    q=module()
    from ralup.baselines.registry import routing_record
    from ralup.__main__ import main
    from ralup.state import restore_snapshot,load_snapshot
    from ralup.records import ArtifactRef
    state,_=fixture(name);record=routing_record(name,state.method.groups,state.method.config)
    config=json.loads((Path(__file__).parents[2]/'configs/smoke.json').read_bytes())
    envelope={'config':config,'purpose':'smoke','pilot_target_budget':7,'seed':505,'output':str(tmp_path/'run'),'checkpoint_directory':str(tmp_path/'checkpoint'),
        'method':{'name':name,'adaptation_label':record.adaptation_label,'config':state.method.config,'learning_rate':.001}}
    path=tmp_path/'config.json';path.write_text(json.dumps(envelope))
    assert main(['run','--phase','explore','--config',str(path)])==0
    result=json.loads(capsys.readouterr().out);resumed=restore_snapshot(load_snapshot(ArtifactRef(**result['checkpoint'])),'cpu')
    assert resumed.method.config==state.method.config and resumed.method.observations==1

@pytest.mark.native
def test_actual_mps_routing_rounding_cached_sample_resume_and_accumulation():
    if not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    for name in ('qar_fp4_adjacent','stochastic_shadow_forward','stochastic_endpoint'):
        one,batch=fixture(name,1,'mps');four,_=fixture(name,4,'mps')
        for _ in range(2):
            train_step(one,batch);train_step(four,batch)
            for n in one.model._contexts:assert torch.equal(one.model._contexts[n].codes,four.model._contexts[n].codes)
            for n,p in one.model.named_parameters():torch.testing.assert_close(p,dict(four.model.named_parameters())[n],atol=3e-7,rtol=3e-5)
        resumed=restore_snapshot(capture_snapshot(one),'mps');train_step(one,batch);train_step(resumed,batch)
        assert exact_cpu_state_hash(one)==exact_cpu_state_hash(resumed)


def test_scales_and_parent_hashes_are_frozen_before_endpoint_conversion():
    state,batch=fixture(rotation='full_hadamard')
    from ralup.quantization.context import prepare_context
    expected={n:prepare_context(state.method.parameters[n],state.model._recipe,0,None).scales for n in state.method.groups[0].parameter_names}
    parent={n:hashlib.sha256(state.method.parameters[n].detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n in expected}
    first=train_step(state,batch)
    for n,scale in expected.items():
        assert torch.equal(state.method.endpoint_states[n]['scales'],scale)
        assert state.method.scale_initialization[n]['original_W_sha256']==parent[n]
        assert state.method.scale_initialization[n]['source']=='pre_conversion_parent_storage_row_max'
    assert first.work['routing_rounding_work']['endpoint_initialization_elements']==sum(state.method.parameters[n].numel() for n in expected)
    train_step(state,batch)
    assert all(torch.equal(state.method.endpoint_states[n]['scales'],scale) for n,scale in expected.items())
    from ralup.state import capture_snapshot,IntegrityError
    state.method.scale_initialization[next(iter(expected))]['scales_sha256']='0'*64
    with pytest.raises(IntegrityError):capture_snapshot(state)

@pytest.mark.parametrize('name',['qar_fp4_adjacent','stochastic_shadow_forward','stochastic_endpoint'])
def test_cpu_packing_parity_and_independent_named_rng(name):
    one,batch=fixture(name,1);four,_=fixture(name,4)
    global_rng=torch.get_rng_state().clone();one.rng_streams['meta']=torch.Generator().manual_seed(12);meta=one.rng_streams['meta'].get_state().clone()
    for _ in range(2):
        train_step(one,batch);train_step(four,batch)
        for n in one.model._contexts:assert torch.equal(one.model._contexts[n].codes,four.model._contexts[n].codes)
        for n,p in one.model.named_parameters():torch.testing.assert_close(p,dict(four.model.named_parameters())[n],atol=3e-7,rtol=3e-5)
    assert torch.equal(global_rng,torch.get_rng_state()) and torch.equal(meta,one.rng_streams['meta'].get_state())


def test_fresh_interpreter_rounding_trusted_reconstruction(tmp_path):
    state,batch=fixture('stochastic_endpoint');train_step(state,batch)
    from ralup.state import capture_snapshot,save_snapshot,exact_cpu_state_hash
    ref=save_snapshot(capture_snapshot(state),tmp_path/'snapshot');train_step(state,batch)
    script='''import sys
sys.path.insert(0,'tests/integration')
from test_stochastic_context import fixture
from ralup.state import load_snapshot,restore_snapshot,exact_cpu_state_hash
from ralup.records import ArtifactRef
from ralup.train import train_step
_,batch=fixture('stochastic_endpoint')
state=restore_snapshot(load_snapshot(ArtifactRef(sys.argv[1],sys.argv[2])),'cpu')
train_step(state,batch)
print(exact_cpu_state_hash(state))
'''
    result=subprocess.run([sys.executable,'-c',script,str(ref.path),ref.sha256],capture_output=True,text=True,check=True)
    assert result.stderr=='' and result.stdout.strip()==exact_cpu_state_hash(state)


def test_verifier_public_report_labels_probabilities_and_replay(tmp_path,capsys):
    module()
    from ralup.__main__ import main
    path=tmp_path/'report.json'
    assert main(['verify-baselines','--suite','routing-rounding','--device','cpu','--output',str(path)])==0
    cli=json.loads(capsys.readouterr().out);report=json.loads(path.read_bytes())
    assert cli['status']=='pass' and report['fp4_uniform_theory_transfer'] is False
    assert set(v['name'] for v in report['variants'])=={'qar_fp4_adjacent','stochastic_shadow_forward','stochastic_endpoint'}
    assert all(v['seed_replay']=='pass' and v['deterministic_evaluation_read_only'] for v in report['variants'])
    assert report['uniform_fixture']['probabilities'][0]==pytest.approx([0.,.2])

@pytest.mark.parametrize('name',['qar_fp4_adjacent','stochastic_shadow_forward','stochastic_endpoint'])
def test_existing_warm_parent_context_cannot_bypass_method_initialization(name):
    state,batch=fixture(name)
    state.model.prepare_quantized_views(0,None)
    result=train_step(state,batch)
    assert state.method.scale_initialization and state.method.observations==1
    assert bool(state.method.endpoint_states)==(name!='stochastic_shadow_forward')
    assert result.work['routing_rounding_work']['probability_elements']>0


def test_rotated_gradient_moments_are_original_W_not_storage():
    state,batch=fixture(rotation='full_hadamard');train_step(state,batch)
    from ralup.quantization.rotation import to_storage
    distinguishable=False
    for n in state.method.groups[0].parameter_names:
        gradient=state.method.parameters[n].grad
        torch.testing.assert_close(state.method.optimizer_states[n]['exp_avg'],gradient*.1,atol=2e-9,rtol=1e-6)
        distinguishable=distinguishable or not torch.allclose(gradient,to_storage(gradient,'full_hadamard'),atol=1e-7,rtol=1e-4)
    assert distinguishable
