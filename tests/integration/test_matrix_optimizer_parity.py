"""Mixed original-W matrix/auxiliary ownership and safe complete continuation."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

from ralup.optim.base import SchedulePoint, default_parameter_groups


def matrix_fixture(name='muon_project_fp32', device='cpu', recipe='row_dynamic_max'):
    from test_resume import fixture
    from ralup.baselines.registry import build_matrix_method
    state = fixture(device,recipe)
    state.method = build_matrix_method(name,dict(state.model.named_parameters()),default_parameter_groups(state.model),{})
    state.optimizer_state = state.method.optimizer_states
    return state


def test_matrix_adapter_routes_embeddings_and_heads_to_independent_no_decay_adamw():
    assert importlib.util.find_spec('ralup.optim.matrix_adapter') is not None, 'M01 named-group adapter missing'
    from ralup.optim.matrix_adapter import MatrixMethodAdapter
    from ralup.optim.muon import MuonUpdateRule
    from ralup.models.transformer import ByteTransformer, ModelConfig
    model = ByteTransformer(ModelConfig(1,8,2))
    groups = default_parameter_groups(model)
    rule = MuonUpdateRule({'auxiliary_lr':.007,'auxiliary_betas':(.8,.95)})
    adapter = MatrixMethodAdapter(dict(model.named_parameters()),groups,rule)
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    for p in model.parameters(): p.grad = torch.full_like(p,.3)
    proposals = adapter.collect_proposals(SchedulePoint(.02,.4,0.))
    for n in groups[0].parameter_names:
        assert set(proposals[n].new_optimizer_state) == {'momentum'}
        torch.testing.assert_close(proposals[n].reference_after_decay,before[n]*(1-.02*.4))
    for n in groups[1].parameter_names:
        assert set(proposals[n].new_optimizer_state) == {'step','exp_avg','exp_avg_sq'}
        torch.testing.assert_close(proposals[n].reference_after_decay,before[n],atol=0,rtol=0)
        torch.testing.assert_close(proposals[n].gradient_displacement,torch.full_like(before[n],-.007),rtol=2e-6,atol=2e-8)
    identities = {n:id(p) for n,p in model.named_parameters()}
    adapter.commit_proposals(proposals)
    assert identities == {n:id(p) for n,p in model.named_parameters()}


@pytest.mark.parametrize('name',['muon_project_fp32','muon_plus_row_col_fp32'])
@pytest.mark.parametrize('device',['cpu','mps'])
def test_quantized_duplicate_and_snapshot_continuation(tmp_path,name,device):
    if device == 'mps':
        if not torch.backends.mps.is_available(): pytest.skip('Actual MPS unavailable')
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    from test_resume import advance
    from ralup.state import capture_snapshot,save_snapshot,load_snapshot,restore_snapshot,exact_cpu_state_hash
    full = matrix_fixture(name,device); advance(full,4)
    expected = exact_cpu_state_hash(full)
    part = matrix_fixture(name,device); advance(part,2)
    ref = save_snapshot(capture_snapshot(part),tmp_path/'checkpoints')
    resumed = restore_snapshot(load_snapshot(ref),device); advance(resumed,2)
    assert exact_cpu_state_hash(resumed) == expected
    assert resumed.optimizer_state is resumed.method.optimizer_states
    for group in resumed.method.groups:
        for n in group.parameter_names:
            fields = {'momentum'} if group.name == 'internal_projection' else {'step','exp_avg','exp_avg_sq'}
            assert set(resumed.optimizer_state[n]) == fields
    script = '''import sys
sys.path.insert(0,"tests/integration")
from test_resume import advance
from ralup.state import restore_snapshot,load_snapshot,exact_cpu_state_hash
from ralup.records import ArtifactRef
s=restore_snapshot(load_snapshot(ArtifactRef(sys.argv[1],sys.argv[2])),sys.argv[3]);advance(s,2)
print(exact_cpu_state_hash(s))
'''
    result = subprocess.run([sys.executable,'-c',script,str(ref.path),ref.sha256,device],capture_output=True,text=True,check=True)
    assert result.stderr == '' and result.stdout.strip() == expected


@pytest.mark.parametrize('mutation',['extra_rule','unknown_schema','wrong_geometry','matrix_to_aux','missing_moment'])
def test_snapshot_rejects_unknown_rule_and_cross_owner_state(mutation):
    from test_resume import advance
    from ralup.state import capture_snapshot,_validated,_Encoder,IntegrityError
    state = matrix_fixture();advance(state,1)
    snap = capture_snapshot(state); payload = _validated(snap)
    if mutation == 'extra_rule': payload['rule']['config'] = {**payload['rule']['config'],'untrusted':1}
    if mutation == 'unknown_schema': payload['rule']['schema'] = 'evil.import.rule'
    n = state.method.groups[0].parameter_names[0]
    if mutation == 'wrong_geometry': payload['method']['optimizer_states'][n]['momentum'] = torch.ones(1)
    if mutation == 'matrix_to_aux': payload['method']['optimizer_states'][n] = state.optimizer_state[state.method.groups[1].parameter_names[0]]
    if mutation == 'missing_moment': payload['method']['optimizer_states'][n] = {}
    encoder = _Encoder();snap.metadata=encoder.encode(payload);snap.tensors=encoder.tensors
    with pytest.raises(IntegrityError): _validated(snap)


def test_registry_distinguishes_variants_rights_and_upstream_departures():
    from ralup.baselines.registry import default_registry
    from ralup.models.transformer import ByteTransformer,ModelConfig
    registry = default_registry(default_parameter_groups(ByteTransformer(ModelConfig(1,8,2))))
    muon = registry.require('muon_project_fp32',adaptation_label='project_muon_fp32_original_W')
    assert 'f98f1cacc0263b04290753e32be8d498c1efc806' in muon.source_revision
    for order in ('none','row','col','row_col','col_row'):
        name = f'muon_plus_{order}_fp32'
        method = registry.require(name,adaptation_label=f'paper_muon_plus_fp32_{order}_original_W')
        assert method.update_rule_settings['normalization'] == order
        assert method.license_status == 'blocked'
        assert registry.claim_eligibility(name)['blocked_is_defeated_baseline'] is False


def test_cli_muon_verification_and_explicit_smoke_method(tmp_path,capsys):
    from ralup.__main__ import main
    from pathlib import Path
    assert main(['verify-baselines','--suite','muon','--output',str(tmp_path/'fixtures.json')]) == 0
    report = json.loads((tmp_path/'fixtures.json').read_bytes());capsys.readouterr()
    assert report['status'] == 'pass' and len(report['variants']) == 6
    config = json.loads(Path('configs/smoke.json').read_bytes())
    envelope = {'config':config,'purpose':'smoke','pilot_target_budget':17,'output':str(tmp_path/'run'),
                'method':{'name':'muon_plus_col_row_fp32','adaptation_label':'paper_muon_plus_fp32_col_row_original_W',
                          'config':{'auxiliary_lr':.002},'learning_rate':.005},
                'checkpoint_directory':str(tmp_path/'snapshots')}
    path=tmp_path/'envelope.json';path.write_text(json.dumps(envelope))
    assert main(['run','--phase','explore','--config',str(path)]) == 0
    result=json.loads(capsys.readouterr().out)
    from ralup.state import restore_snapshot,load_snapshot
    from ralup.records import ArtifactRef
    state=restore_snapshot(load_snapshot(ArtifactRef(**result['checkpoint'])),'cpu')
    assert state.method.rule.configuration['normalization'] == 'col_row'
    assert state.schedule['point'].lr == .005
    assert json.loads((tmp_path/'run/run.json').read_bytes())['method'] == 'muon_plus_col_row_fp32'


def test_cli_muon_actual_mps_cross_device_values(tmp_path):
    if not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    from ralup.baselines.muon_checks import check_muon
    report=json.loads(check_muon('mps',tmp_path/'muon-mps.json').path.read_bytes())
    assert report['status'] == 'pass', report['blockers']
    assert len(report['variants']) == 6
