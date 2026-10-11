"""Derived parents preserve original roles and import only authenticated model state."""
import hashlib
import json
import random

import pytest
import torch

from ralup.config import canonical_bytes
from ralup.data.articles import Article
from ralup.data.cursor import BatchCursor
from ralup.models.transformer import ByteTransformer, ModelConfig
from ralup.optim.adamw import AdamWUpdateRule
from ralup.optim.base import MethodAdapter, default_parameter_groups
from ralup.records import ArtifactRef, sha256_file
from ralup.state import IntegrityError, capture_snapshot, save_snapshot, exact_cpu_state_hash
from ralup.train import TrainingState, select_logical_batch, train_step


def write(path, value):
    path.write_bytes(canonical_bytes(value))
    return ArtifactRef(path, sha256_file(path))


def fresh(partition='target_train'):
    torch.manual_seed(817)
    model = ByteTransformer(ModelConfig(1, 8, 2))
    raw = b'partition separated model-only parent fixture'
    cursor = BatchCursor.from_articles([Article(raw, [], hashlib.sha256(raw).hexdigest())], seed=23)
    return TrainingState(model, MethodAdapter(dict(model.named_parameters()), default_parameter_groups(model), AdamWUpdateRule()),
        cursor=cursor, configuration={'source_partition': partition},
        rng_streams={'torch': torch.Generator().manual_seed(819), 'python': random.Random(820)})


def original(tmp_path):
    from ralup.calibrate import import_warmstart
    warm = fresh()
    warm.configuration['artifact_role'] = 'warmstart'
    train_step(warm, select_logical_batch(warm, 4))
    wref = save_snapshot(capture_snapshot(warm), tmp_path/'warm')
    state = import_warmstart(wref, fresh(), partition='target_train', continuation_targets=8)
    state.configuration.update(artifact_role='continuation', resolved={'fixture': True})
    train_step(state, select_logical_batch(state, 4))
    ref = save_snapshot(capture_snapshot(state), tmp_path/'original')
    selection = write(tmp_path/'selection.json', {'scope': 'CPU fixture selection', 'checkpoint': {'path':str(ref.path),'sha256':ref.sha256}})
    receipt = write(tmp_path/'finish.json', {'exit_code':0, 'elapsed_seconds':1.25,
        'completed_objects':[{'path':str(p.path.relative_to(tmp_path)), 'sha256':p.sha256, 'bytes':p.path.stat().st_size}
                             for p in (ref, wref, selection)]})
    return state, ref, selection, receipt


def derived(tmp_path):
    from ralup.warmstart import create_derived_warmstart
    state, ref, selection, receipt = original(tmp_path)
    record = create_derived_warmstart(ref, tmp_path/'derived.json', completed_attempts=[receipt], selection_evidence=[selection])
    return state, ref, record, receipt


def test_derived_model_only_preserves_owners_cursor_rng_cost_and_original_role(tmp_path):
    from ralup.warmstart import import_derived_warmstart
    from ralup.calibrate import import_warmstart
    parent, original_ref, record, receipt = derived(tmp_path)
    target = fresh()
    ids = {n:id(p) for n,p in target.model.named_parameters()}
    result = import_derived_warmstart(record, target, partition='target_train', continuation_targets=7)
    assert result is target
    assert ids == {n:id(p) for n,p in target.model.named_parameters()}
    assert all(torch.equal(w, parent.model.state_dict()[n]) for n,w in target.model.state_dict().items())
    assert target.cursor.state == {**parent.cursor.state, 'run_target_budget':15}
    assert target.schedule['step'] == 0 and not target.histories and not target.quantizer_state
    assert all(not v for v in target.optimizer_state.values())
    assert torch.equal(target.rng_streams['torch'].get_state(), parent.rng_streams['torch'].get_state())
    assert target.rng_streams['python'].getstate() == parent.rng_streams['python'].getstate()
    assert target.ledger.stream == parent.ledger.stream
    assert target.ledger.passes == parent.ledger.passes
    assert target.ledger.elapsed['seconds'] >= parent.ledger.elapsed['seconds']
    assert target.ledger.events[-1]['status'] == 'derived_model_only_import'
    ancestry = target.configuration['warmstart_parent']['derived_record']
    assert ancestry['original_role'] == 'continuation'
    assert ancestry['project_cost']['completed_seconds'] == 1.25
    assert sha256_file(original_ref.path) == original_ref.sha256
    with pytest.raises(IntegrityError, match='Generated warmstart'):
        import_warmstart(original_ref, fresh(), partition='target_train', continuation_targets=7)


@pytest.mark.parametrize('field,value', [('schema','unknown'), ('original_role','warmstart'),
    ('cost',{}), ('source_partition','meta_train'), ('model_parameters',[]), ('resets',[]), ('unknown',True)])
def test_rehashed_derived_semantic_tamper_rejects_before_mutation(tmp_path, field, value):
    from ralup.warmstart import import_derived_warmstart
    _, _, ref, _ = derived(tmp_path)
    data = json.loads(ref.path.read_bytes()); data[field] = value
    bad = write(tmp_path/'bad.json', data)
    target = fresh(); before = exact_cpu_state_hash(target)
    with pytest.raises(IntegrityError):
        import_derived_warmstart(bad, target, partition='target_train', continuation_targets=7)
    assert exact_cpu_state_hash(target) == before


def test_missing_duplicate_or_tampered_cost_receipts_reject(tmp_path):
    from ralup.warmstart import create_derived_warmstart, import_derived_warmstart
    _, original_ref, record, receipt = derived(tmp_path)
    for refs in ([], [receipt,receipt]):
        with pytest.raises(IntegrityError, match='cost|receipt|attempt'):
            create_derived_warmstart(original_ref, tmp_path/'invalid.json', completed_attempts=refs, selection_evidence=[])
    receipt.path.write_bytes(receipt.path.read_bytes()+b' ')
    with pytest.raises(IntegrityError, match='hash'):
        import_derived_warmstart(record, fresh(), partition='target_train', continuation_targets=7)


def test_derived_partition_model_and_freshness_are_strict(tmp_path):
    from ralup.warmstart import import_derived_warmstart
    _, _, ref, _ = derived(tmp_path)
    wrong = fresh('meta_train')
    with pytest.raises(IntegrityError, match='partition'):
        import_derived_warmstart(ref, wrong, partition='meta_train', continuation_targets=7)
    wrong = fresh(); wrong.model = ByteTransformer(ModelConfig(1,16,2))
    with pytest.raises(IntegrityError, match='model'):
        import_derived_warmstart(ref, wrong, partition='target_train', continuation_targets=7)
    used = fresh(); train_step(used, select_logical_batch(used,4))
    with pytest.raises(IntegrityError, match='fresh'):
        import_derived_warmstart(ref, used, partition='target_train', continuation_targets=7)


def test_search_rejects_unknown_parent_kind_before_construction():
    from ralup.search import create_trial_state
    with pytest.raises(ValueError, match='parent_kind'):
        create_trial_state({'parent_kind':'silent_full_restore'})


def test_derived_import_rejects_different_data_even_with_matching_partition(tmp_path):
    from ralup.warmstart import import_derived_warmstart
    _, _, ref, _ = derived(tmp_path)
    target = fresh()
    raw = b'different source with the same partition label'
    target.cursor = BatchCursor.from_articles([Article(raw, [], hashlib.sha256(raw).hexdigest())], seed=23)
    before = exact_cpu_state_hash(target)
    with pytest.raises(IntegrityError, match='data'):
        import_derived_warmstart(ref, target, partition='target_train', continuation_targets=7)
    assert exact_cpu_state_hash(target) == before
