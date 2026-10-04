"""Fail closed on incomplete, unauthenticated or unregistered checkpoint state."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from ralup.config import canonical_bytes
from ralup.records import ArtifactRef, sha256_file


def fixture():
    from ralup.models.transformer import ByteTransformer, ModelConfig
    from ralup.optim.base import MethodAdapter, default_parameter_groups
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.train import TrainingState
    with torch.random.fork_rng():
        torch.manual_seed(607)
        model = ByteTransformer(ModelConfig(1,8,2))
    model.configure_quantization({'name':'row_fixed_max'})
    return TrainingState(model,MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule()))


def test_capture_owns_every_mutable_tensor_rng_and_metadata(tmp_path):
    from ralup.state import capture_snapshot, restore_snapshot, exact_cpu_state_hash
    import random
    state = fixture()
    state.histories['nested'] = {'tensor':torch.tensor([2.]),'values':[3]}
    state.rng_streams = {'meta':random.Random(17),'quantizer':torch.Generator().manual_seed(19)}
    snapshot = capture_snapshot(state)
    before = exact_cpu_state_hash(restore_snapshot(snapshot,'cpu'))
    state.histories['nested']['tensor'].add_(4)
    state.histories['nested']['values'].append(9)
    state.rng_streams['meta'].random()
    torch.rand(3,generator=state.rng_streams['quantizer'])
    with torch.no_grad():
        next(state.model.parameters()).add_(7)
    restored = restore_snapshot(snapshot,'cpu')
    assert exact_cpu_state_hash(restored) == before
    restored.histories['nested']['tensor'].add_(9)
    assert exact_cpu_state_hash(restore_snapshot(snapshot,'cpu')) == before


@pytest.mark.parametrize('stage',['pending','work','proposal','failed'])
def test_partial_step_and_failed_state_cannot_be_captured(stage):
    from ralup.state import capture_snapshot, IncompleteStep
    state = fixture()
    if stage == 'pending': state._pending = object()
    if stage == 'work': state._work = {}
    if stage == 'proposal': state.method.proposals['unfinished'] = object()
    if stage == 'failed': state.failed = True
    with pytest.raises(IncompleteStep): capture_snapshot(state)


@pytest.mark.parametrize('owner',['method','rule','training'])
def test_unknown_mutable_owner_is_never_silently_omitted(owner):
    from ralup.state import capture_snapshot, IntegrityError
    state = fixture()
    target = {'method':state.method,'rule':state.method.rule,'training':state}[owner]
    target.future_state = {'tensor':torch.ones(2)}
    with pytest.raises(IntegrityError,match='registered|Unknown'): capture_snapshot(state)


@pytest.mark.parametrize('damage',['corrupt','truncated','unknown','hash','shape','extra'])
def test_corrupt_or_unknown_payload_is_unusable(tmp_path,damage):
    from ralup.state import capture_snapshot, save_snapshot, load_snapshot, IntegrityError
    ref = save_snapshot(capture_snapshot(fixture()),tmp_path/'snapshots')
    manifest_path = ref.path
    manifest = json.loads(manifest_path.read_bytes())
    tensors = manifest_path.parent/'tensors.safetensors'
    if damage == 'corrupt': tensors.write_bytes(tensors.read_bytes()+b'broken')
    elif damage == 'truncated': tensors.write_bytes(tensors.read_bytes()[:50])
    elif damage == 'hash': ref = ArtifactRef(ref.path,'0'*64)
    else:
        if damage == 'unknown': manifest['schema'] = 'unknown.v999'
        if damage == 'shape': next(iter(manifest['tensors'].values()))['shape'] = [999]
        if damage == 'extra': manifest['unexpected_owner'] = {}
        manifest_path.write_bytes(canonical_bytes(manifest))
        ref = ArtifactRef(ref.path,sha256_file(ref.path))
    with pytest.raises(IntegrityError): load_snapshot(ref)


def test_external_identities_are_authenticated_at_capture_load_and_restore(tmp_path):
    from ralup.state import capture_snapshot, save_snapshot, load_snapshot, restore_snapshot, IntegrityError
    state = fixture()
    source = tmp_path/'source.json'
    source.write_text('{"fixture":true}')
    state.identities = {'data':ArtifactRef(source,sha256_file(source))}
    snapshot = capture_snapshot(state)
    ref = save_snapshot(snapshot,tmp_path/'snapshots')
    source.write_text('changed')
    for action in (lambda:capture_snapshot(state),lambda:load_snapshot(ref),lambda:restore_snapshot(snapshot,'cpu')):
        with pytest.raises(IntegrityError,match='hash'): action()


@pytest.mark.parametrize('failure',['fsync','rename','pointer_rename'])
def test_failed_publication_recovers_previous_verified_generation(tmp_path,monkeypatch,failure):
    import ralup.state as checkpoint
    state = fixture()
    root = tmp_path/'snapshots'
    first = checkpoint.save_snapshot(checkpoint.capture_snapshot(state),root)
    original_sync,original_replace = checkpoint.os.fsync,checkpoint.os.replace
    def broken_sync(fd): raise OSError('injected fsync interruption')
    def broken_replace(source,target):
        if failure == 'rename' or Path(target).name == 'latest.json':
            raise OSError('injected rename interruption')
        return original_replace(source,target)
    if failure == 'fsync': monkeypatch.setattr(checkpoint.os,'fsync',broken_sync)
    else: monkeypatch.setattr(checkpoint.os,'replace',broken_replace)
    with pytest.raises(OSError): checkpoint.save_snapshot(checkpoint.capture_snapshot(state),root)
    monkeypatch.setattr(checkpoint.os,'fsync',original_sync)
    monkeypatch.setattr(checkpoint.os,'replace',original_replace)
    recovered = checkpoint.recover_snapshot(root)
    checkpoint.load_snapshot(recovered)
    assert recovered == first


def test_retention_keeps_last_two_valid_and_declared_milestones(tmp_path):
    from ralup.state import capture_snapshot, save_snapshot, load_snapshot, recover_snapshot
    root = tmp_path/'snapshots'
    state = fixture()
    snapshot = capture_snapshot(state)
    snapshot.milestone = True
    milestone = save_snapshot(snapshot,root)
    refs = []
    for _ in range(4): refs.append(save_snapshot(capture_snapshot(state),root))
    assert milestone.path.exists()
    assert not refs[0].path.exists() and not refs[1].path.exists()
    for ref in refs[-2:]: load_snapshot(ref)
    refs[-1].path.write_text('corrupt')
    assert recover_snapshot(root) == refs[-2]


def test_nonfinite_and_executable_values_cannot_publish(tmp_path):
    from ralup.state import capture_snapshot, save_snapshot, IntegrityError
    state = fixture()
    state.histories['code'] = lambda:None
    with pytest.raises(IntegrityError): capture_snapshot(state)
    state.histories['code'] = float('nan')
    with pytest.raises(IntegrityError): capture_snapshot(state)
    del state.histories['code']
    with torch.no_grad(): next(state.model.parameters()).fill_(float('nan'))
    with pytest.raises(IntegrityError): capture_snapshot(state)


def test_compatible_container_with_wrong_weight_dtype_is_rejected(tmp_path):
    from ralup.state import capture_snapshot, restore_snapshot, save_snapshot, IntegrityError
    snapshot = capture_snapshot(fixture())
    weights = snapshot.metadata['items']['weights']['items']
    key = weights['output.weight']['key']
    snapshot.tensors[key] = snapshot.tensors[key].to(torch.float64)
    with pytest.raises(IntegrityError,match='dtype'): restore_snapshot(snapshot,'cpu')
    with pytest.raises(IntegrityError,match='dtype'): save_snapshot(snapshot,tmp_path/'snapshots')


def test_corrupted_older_manifest_cannot_be_rehashed_into_recovery(tmp_path):
    from ralup.state import capture_snapshot, save_snapshot, recover_snapshot, IntegrityError
    root = tmp_path/'snapshots'
    first = save_snapshot(capture_snapshot(fixture()),root)
    manifest = json.loads(first.path.read_bytes())
    manifest['milestone'] = True
    first.path.write_bytes(canonical_bytes(manifest))
    (root/'latest.json').write_text('interrupted')
    with pytest.raises(IntegrityError): recover_snapshot(root)


def test_complete_unpublished_orphan_is_not_recovery_candidate(tmp_path):
    from ralup.state import capture_snapshot, save_snapshot, recover_snapshot, IntegrityError
    import shutil
    root = tmp_path/'snapshots'
    published = save_snapshot(capture_snapshot(fixture()),root)
    orphan = root/'g-99999999999999999999-orphan'
    orphan.mkdir()
    for name in ('manifest.json','tensors.safetensors'): shutil.copyfile(published.path.parent/name,orphan/name)
    shutil.rmtree(published.path.parent)
    with pytest.raises(IntegrityError): recover_snapshot(root)


def test_direct_coverage_and_prior_failed_costs_survive_restore():
    from ralup.data.articles import Article
    from ralup.data.blocks import tile_article
    from ralup.data.batches import LogicalBatch
    from ralup.train import train_step
    from ralup.state import capture_snapshot, restore_snapshot
    state = fixture()
    raw = b'abc'
    block = tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))[0]
    batch = LogicalBatch([block],3)
    train_step(state,batch)
    state.ledger.record({'step':1,'status':'failed','stream':0,'unique':0,'loss_targets':3,
                         'additional_loss':2,'backward':3,'query':2,'forward_context':3,
                         'elapsed_seconds':.25,'failure':'prior discarded attempt','nonfinite_count':1})
    before = dict(state.ledger.passes)
    restored = restore_snapshot(capture_snapshot(state),'cpu')
    assert restored.ledger.passes == before
    assert restored.ledger.elapsed == state.ledger.elapsed
    assert restored.ledger.events[-1]['failure'] == 'prior discarded attempt'
    train_step(restored,batch)
    assert restored.ledger.stream == {'targets':6,'unique':3,'updates':2}
    assert restored.ledger.passes['backward'] == 9


def test_completed_step_cannot_omit_one_parameters_moments():
    from ralup.data.articles import Article
    from ralup.data.blocks import tile_article
    from ralup.data.batches import LogicalBatch
    from ralup.train import train_step
    from ralup.state import capture_snapshot, IntegrityError
    state = fixture()
    raw = b'abc'
    train_step(state,LogicalBatch(tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest())),3))
    state.method.optimizer_states['output.weight'] = {}
    with pytest.raises(IntegrityError,match='moment|completed'): capture_snapshot(state)


@pytest.mark.parametrize('sync_number',[3,4,5,6,7,8,9])
def test_fsync_interruption_at_each_publication_boundary_leaves_only_valid_recovery(tmp_path,monkeypatch,sync_number):
    import ralup.state as checkpoint
    root = tmp_path/'snapshots'
    first = checkpoint.save_snapshot(checkpoint.capture_snapshot(fixture()),root)
    original = checkpoint.os.fsync
    calls = 0
    def interrupt(fd):
        nonlocal calls
        calls += 1
        if calls == sync_number: raise OSError('interrupted durability boundary')
        return original(fd)
    monkeypatch.setattr(checkpoint.os,'fsync',interrupt)
    with pytest.raises(OSError): checkpoint.save_snapshot(checkpoint.capture_snapshot(fixture()),root)
    monkeypatch.setattr(checkpoint.os,'fsync',original)
    recovered = checkpoint.recover_snapshot(root)
    checkpoint.load_snapshot(recovered)
    if sync_number <= 5: assert recovered == first
    else: assert recovered != first


def test_registered_method_extra_owner_roundtrips_and_unknown_nested_schema_fails(tmp_path):
    from ralup.optim.base import MethodAdapter
    from ralup.state import capture_snapshot, restore_snapshot, save_snapshot, load_snapshot, register_method_schema, IntegrityError
    class RegisteredMethod(MethodAdapter):
        def __init__(self,parameters,groups,rule):
            super().__init__(parameters,groups,rule)
            self.features = {'normalizer':torch.tensor([3.,7.]),'pending_refresh':False}
            self.snapshot_schema['adapter_mutable_state'] = 'fixture_features.v1'
    register_method_schema('fixture_registered_method.v1',RegisteredMethod,
                           ('optimizer_states','proposals','snapshot_schema','features'),RegisteredMethod)
    state = fixture()
    state.method = RegisteredMethod(dict(state.model.named_parameters()),state.method.groups,state.method.rule)
    state.optimizer_state = state.method.optimizer_states
    snapshot = capture_snapshot(state)
    restored = restore_snapshot(snapshot,'cpu')
    assert isinstance(restored.method,RegisteredMethod)
    assert torch.equal(restored.method.features['normalizer'],torch.tensor([3.,7.]))
    restored.method.features['normalizer'].add_(8)
    assert torch.equal(restore_snapshot(snapshot,'cpu').method.features['normalizer'],torch.tensor([3.,7.]))
    ref = save_snapshot(snapshot,tmp_path/'snapshots')
    manifest = json.loads(ref.path.read_bytes())
    manifest['metadata']['items']['model_config']['schema'] = 'unregistered.v1'
    ref.path.write_bytes(canonical_bytes(manifest))
    with pytest.raises(IntegrityError): load_snapshot(ArtifactRef(ref.path,sha256_file(ref.path)))


def test_embedded_configuration_hash_is_checked_before_restore():
    from ralup.state import capture_snapshot, restore_snapshot, IntegrityError
    state = fixture()
    state.configuration = {'horizon':7}
    snapshot = capture_snapshot(state)
    snapshot.metadata['items']['configuration']['items']['horizon'] = 8
    with pytest.raises(IntegrityError,match='Configuration hash'): restore_snapshot(snapshot,'cpu')
