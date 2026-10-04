"""Same-device replay with partial carry, original-W moments and cached scales."""
import hashlib
import json
import os
import random

import pytest
import torch

from ralup.data.articles import Article
from ralup.data.cursor import BatchCursor
from ralup.models.transformer import ByteTransformer, ModelConfig
from ralup.optim.adamw import AdamWUpdateRule
from ralup.optim.base import MethodAdapter, SchedulePoint, default_parameter_groups
from ralup.train import TrainingState, select_logical_batch, train_step


def fixture(device,recipe):
    torch.manual_seed(607)
    random.seed(608)
    if device == 'mps': torch.mps.manual_seed(609)
    model = ByteTransformer(ModelConfig(1,8,2)).to(device)
    model.configure_quantization({'name':recipe,'rotation':'full_hadamard'})
    raw = b'abcdefghijklmnopq'
    cursor = BatchCursor.from_articles([Article(raw,[],hashlib.sha256(raw).hexdigest())],seed=33)
    method = MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule(betas=(.8,.95)))
    streams = {'initialization':torch.Generator().manual_seed(60),'data':random.Random(61),
               'quantizer':torch.Generator(device=device).manual_seed(62),'meta':random.Random(63),'timing':random.Random(64)}
    return TrainingState(model,method,cursor=cursor,rng_streams=streams,microbatch=2,clip_norm=.2,
                         schedule={'step':0,'point':SchedulePoint(.001,.02,0.)})


def advance(state,count):
    for _ in range(count):
        step = state.schedule['step']
        state.schedule['point'] = SchedulePoint(.001/(1+step),.02,step/10)
        # Draw all named/global streams to expose missing restoration independently
        # of today's deterministic quantizer (future stochastic methods own draws).
        for rng in state.rng_streams.values():
            if isinstance(rng,random.Random): rng.random()
            else: torch.rand(2,generator=rng,device=rng.device)
        random.random()
        torch.rand(2)
        if next(state.model.parameters()).device.type == 'mps': torch.rand(2,device='mps')
        train_step(state,select_logical_batch(state,5))


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('recipe',['row_fixed_max','row_dynamic_max','quest_fp4_row_reference'])
def test_uninterrupted_equals_atomic_restored_continuation(tmp_path,device,recipe):
    from ralup.state import capture_snapshot, restore_snapshot, save_snapshot, load_snapshot, exact_cpu_state_hash
    if device == 'mps':
        if not torch.backends.mps.is_available(): pytest.skip('Actual MPS unavailable')
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
        # Prior authenticated H04 duplicate evidence declared zero variation.
        evidence = json.loads((__import__('pathlib').Path(__file__).parents[2]/'docs/numerics/h04-evidence.json').read_text())
        assert evidence['tolerances']['duplicate_gradient_atol'] == 0
    uninterrupted = fixture(device,recipe)
    advance(uninterrupted,6)
    expected = exact_cpu_state_hash(uninterrupted)
    interrupted = fixture(device,recipe)
    advance(interrupted,3)
    assert interrupted.cursor.state['carry'] is not None
    assert interrupted.histories['loss_updates'] == 3
    assert all(s['step'] == 3 for s in interrupted.optimizer_state.values())
    elapsed = interrupted.ledger.elapsed['seconds']
    ref = save_snapshot(capture_snapshot(interrupted),tmp_path/'checkpoints')
    # Deliberately disturb process RNG and Parameters before safe reconstruction.
    torch.rand(99); random.random()
    resumed = restore_snapshot(load_snapshot(ref),device)
    assert resumed.ledger.elapsed['seconds'] == elapsed
    assert resumed.process_audit[-1]['kind'] == 'resume'
    assert resumed.optimizer_state is resumed.method.optimizer_states
    identities = {name:id(p) for name,p in resumed.model.named_parameters()}
    advance(resumed,3)
    assert exact_cpu_state_hash(resumed) == expected
    assert resumed.cursor.state == uninterrupted.cursor.state
    assert identities == {name:id(p) for name,p in resumed.model.named_parameters()}
    assert resumed.ledger.stream == {'targets':30,'unique':17,'updates':6}


def test_weight_only_restore_fails_trajectory_equivalence():
    from ralup.state import exact_cpu_state_hash
    uninterrupted = fixture('cpu','row_fixed_max'); advance(uninterrupted,6)
    interrupted = fixture('cpu','row_fixed_max'); advance(interrupted,3)
    weights_only = fixture('cpu','row_fixed_max')
    weights_only.model.load_state_dict(interrupted.model.state_dict())
    weights_only.cursor = BatchCursor(interrupted.cursor.state)
    weights_only.schedule = interrupted.schedule.copy()
    advance(weights_only,3)
    assert exact_cpu_state_hash(weights_only) != exact_cpu_state_hash(uninterrupted)
    assert any(not torch.equal(p,dict(uninterrupted.model.named_parameters())[n]) for n,p in weights_only.model.named_parameters())


def test_cli_explicit_snapshot_and_resume_retains_costs_and_lineage(tmp_path,capsys):
    from pathlib import Path
    from ralup.__main__ import main
    from ralup.state import load_snapshot
    from ralup.records import ArtifactRef
    template = json.loads((Path(__file__).parents[2]/'configs/smoke.json').read_bytes())
    envelope = {'config':template,'purpose':'smoke','pilot_target_budget':17,'seed':606,
                'output':str(tmp_path/'first'),'checkpoint_directory':str(tmp_path/'checkpoints')}
    path = tmp_path/'launch.json'
    path.write_text(json.dumps(envelope))
    assert main(['run','--phase','explore','--config',str(path)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first['checkpoint_published'] is True
    ref = first['checkpoint']
    past = load_snapshot(ArtifactRef(**ref))
    envelope.update({'output':str(tmp_path/'resumed'),'resume':ref})
    path.write_text(json.dumps(envelope))
    assert main(['run','--phase','explore','--config',str(path)]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed['stream_targets'] == 17 and resumed['steps'] == 1
    metadata = json.loads((tmp_path/'resumed/run.json').read_text())
    original = json.loads((tmp_path/'first/run.json').read_text())
    assert metadata['lineage'] == {'parent':original['run_id'],'resume':ref['sha256']}
    from ralup.state import restore_snapshot
    restored = restore_snapshot(load_snapshot(ArtifactRef(**resumed['checkpoint'])),'cpu')
    old = restore_snapshot(past,'cpu')
    assert restored.ledger.passes == old.ledger.passes
    assert restored.ledger.elapsed == old.ledger.elapsed
    assert len(restored.process_audit) > len(old.process_audit)


@pytest.mark.parametrize('device',['cpu','mps'])
def test_new_interpreter_continuation_matches_uninterrupted(tmp_path,device):
    from pathlib import Path
    import subprocess
    import sys
    from ralup.state import capture_snapshot, save_snapshot, exact_cpu_state_hash
    if device == 'mps':
        if not torch.backends.mps.is_available(): pytest.skip('Actual MPS unavailable')
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    uninterrupted = fixture(device,'row_fixed_max'); advance(uninterrupted,6)
    expected = exact_cpu_state_hash(uninterrupted)
    interrupted = fixture(device,'row_fixed_max'); advance(interrupted,3)
    ref = save_snapshot(capture_snapshot(interrupted),tmp_path/'snapshots')
    script = '''import sys
sys.path.insert(0,"tests/integration")
from test_resume import advance
from ralup.state import restore_snapshot, load_snapshot, exact_cpu_state_hash
from ralup.records import ArtifactRef
state = restore_snapshot(load_snapshot(ArtifactRef(sys.argv[1],sys.argv[2])),sys.argv[3])
advance(state,3)
print(exact_cpu_state_hash(state))
'''
    result = subprocess.run([sys.executable,'-c',script,str(ref.path),ref.sha256,device],
                            cwd=Path(__file__).parents[2],capture_output=True,text=True,check=True)
    assert result.stderr == ''
    assert result.stdout.strip() == expected


def test_cli_records_actual_initialization_and_independent_named_streams(tmp_path,capsys):
    from pathlib import Path
    from ralup.__main__ import main
    from ralup.state import load_snapshot, restore_snapshot
    from ralup.records import ArtifactRef
    template = json.loads((Path(__file__).parents[2]/'configs/smoke.json').read_bytes())
    path = tmp_path/'launch.json'
    path.write_text(json.dumps({'config':template,'purpose':'smoke','pilot_target_budget':3,'seed':606,
        'output':str(tmp_path/'run'),'checkpoint_directory':str(tmp_path/'checkpoints')}))
    assert main(['run','--phase','explore','--config',str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    state = restore_snapshot(load_snapshot(ArtifactRef(**result['checkpoint'])),'cpu')
    assert set(state.rng_streams) == {'initialization','quantizer','meta','timing'}
    assert 'data' in state.cursor.state['rng_states']
    with torch.random.fork_rng():
        torch.manual_seed(606)
        ByteTransformer(ModelConfig(1,8,2))
        expected_draw_state = torch.get_rng_state()
    assert torch.equal(state.rng_streams['initialization'].get_state(),expected_draw_state)
    timing_before = state.rng_streams['timing'].getstate()
    data_before = state.cursor.state['rng_states']['data']
    state.rng_streams['meta'].random()
    assert state.rng_streams['timing'].getstate() == timing_before
    assert state.cursor.state['rng_states']['data'] == data_before


def test_adapter_cannot_mutate_snapshot_configuration_during_commit():
    from ralup.train import TrainingFailure
    state = fixture('cpu','row_fixed_max')
    state.configuration = {'fixture':{'horizon':3}}
    finish = state.method.finish_step
    def broken(training,point):
        result = finish(training,point)
        training.configuration['fixture']['horizon'] = 9
        return result
    state.method.finish_step = broken
    with pytest.raises(TrainingFailure): train_step(state,select_logical_batch(state,5))
    assert state.configuration == {'fixture':{'horizon':3}}
