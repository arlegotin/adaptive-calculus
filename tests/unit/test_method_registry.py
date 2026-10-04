"""Registry cannot silently promote fixtures or unnamed adaptations."""
import dataclasses
import json

import pytest


def test_adamw_registry_names_state_groups_upstream_and_fixture_limit():
    from ralup.baselines.registry import default_registry
    from ralup.models.transformer import ByteTransformer,ModelConfig
    from ralup.optim.base import default_parameter_groups
    model = ByteTransformer(ModelConfig(1,8,2))
    registry = default_registry(default_parameter_groups(model))
    method = registry.require('adamw_scalar_fp32',adaptation_label='original_W_scalar_fixture')
    assert method.family == 'AdamW'
    assert method.source_revision.startswith('torch==2.14.1;git=')
    assert method.ordered_parameter_groups[1].weight_decay == 0.
    assert method.state_schema['optimizer_states'] == 'named.original_W.adamw.v1'
    assert method.required_operations
    assert method.licenses
    assert method.parity_status == 'unverified'
    assert not registry.claim_eligibility(method.name)['eligible']
    with pytest.raises(ValueError,match='unregistered'):
        registry.require('soap',adaptation_label='project')
    with pytest.raises(ValueError,match='adaptation'):
        registry.require(method.name)


def test_registry_blocks_missing_fields_duplicate_names_and_unmatched_adaptation():
    from ralup.baselines.registry import MethodRegistry,default_registry
    from ralup.optim.base import ParameterGroup
    method = default_registry((ParameterGroup('internal',('w',)),)).require('adamw_scalar_fp32',adaptation_label='original_W_scalar_fixture')
    with pytest.raises(ValueError,match='revision'):
        dataclasses.replace(method,source_revision='')
    with pytest.raises(ValueError,match='schema'):
        dataclasses.replace(method,state_schema={})
    registry = MethodRegistry()
    registry.register(method)
    with pytest.raises(ValueError,match='registered'):
        registry.register(method)
    with pytest.raises(ValueError,match='adaptation'):
        registry.require(method.name,adaptation_label='fp4_lm_adapter')


def test_verified_fixture_and_blocked_operations_never_authorize_scientific_claims():
    from ralup.baselines.registry import MethodRegistry,default_registry
    from ralup.optim.base import ParameterGroup
    method = default_registry((ParameterGroup('internal',('w',)),)).require('adamw_scalar_fp32',adaptation_label='original_W_scalar_fixture')
    registry = MethodRegistry()
    registry.register(dataclasses.replace(method,parity_status='pass'))
    assert 'fixture' in ' '.join(registry.claim_eligibility(method.name)['blockers'])
    registry2 = MethodRegistry()
    registry2.register(dataclasses.replace(method,artifact_kind='project_adaptation',parity_status='pass',operations_status='blocked'))
    assert 'operations' in ' '.join(registry2.claim_eligibility(method.name)['blockers'])


def test_cli_adamw_fixture_records_exact_cpu_and_no_scientific_gate(tmp_path):
    from ralup.__main__ import main
    output = tmp_path/'adamw.json'
    assert main(['verify-baselines','--suite','adamw','--output',str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['status'] == 'pass'
    assert report['comparison']['same_device_bitwise']
    assert report['comparison']['cases'] >= 4
    assert report['scientific_gate'] is None
    assert not report['claim_eligibility']['eligible']
    assert report['upstream']['scalar_source_sha256']


def test_actual_mps_adamw_fixture(tmp_path):
    import os
    import torch
    from ralup.baselines.checks import check_adamw
    if not torch.backends.mps.is_available():
        pytest.skip('Actual MPS unavailable; sandbox skip is not device evidence')
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    report = json.loads(check_adamw('mps',tmp_path/'mps.json').path.read_text())
    assert report['status'] == 'pass', report['blockers']
    assert report['comparison']['same_device_bitwise']


def test_registry_and_adapter_metadata_record_actual_optimizer_settings():
    import torch
    from ralup.baselines.registry import default_registry
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.optim.base import MethodAdapter,ParameterGroup
    groups = (ParameterGroup('internal',('w',)),)
    rule = AdamWUpdateRule(betas=(.95,.999),eps=1e-4)
    record = default_registry(groups,rule=rule).require('adamw_scalar_fp32',adaptation_label='original_W_scalar_fixture')
    assert record.update_rule_settings['betas'] == (.95,.999)
    assert record.update_rule_settings['eps'] == 1e-4
    adapter = MethodAdapter({'w':torch.nn.Parameter(torch.tensor([2.]))},groups,rule)
    assert adapter.metadata['update_rule_settings'] == record.update_rule_settings
    assert adapter.metadata['schedule_fields'] == ('lr','weight_decay','progress')
    assert adapter.metadata['extra_passes'] == ()
