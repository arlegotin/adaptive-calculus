import hashlib
import json

import pytest
import torch

from ralup.data.articles import Article
from ralup.data.cursor import BatchCursor
from ralup.models.transformer import ByteTransformer, ModelConfig
from ralup.optim.adamw import AdamWUpdateRule
from ralup.optim.base import MethodAdapter, SchedulePoint, default_parameter_groups
from ralup.records import ArtifactRef
from ralup.state import capture_snapshot, save_snapshot, exact_cpu_state_hash, IntegrityError
from ralup.train import TrainingState, train_step, select_logical_batch


def fixture():
    torch.manual_seed(808)
    model = ByteTransformer(ModelConfig(1, 8, 2))
    model.configure_quantization({'name': 'row_fixed_max', 'rotation': 'identity'})
    raw = b'abcdefghijklmnopq'
    cursor = BatchCursor.from_articles([Article(raw, [], hashlib.sha256(raw).hexdigest())], seed=23)
    method = MethodAdapter(dict(model.named_parameters()), default_parameter_groups(model), AdamWUpdateRule())
    return TrainingState(model, method, cursor=cursor, schedule={'step': 0, 'point': SchedulePoint(.001, 0., 0.)},
                         configuration={'source_partition': 'meta_train'},
                         rng_streams={'quantizer': torch.Generator().manual_seed(83)})


def test_disposable_warmup_restores_scientific_state_and_charges_cost(tmp_path):
    from ralup.calibrate import measure_capacity
    state = fixture()
    before = exact_cpu_state_hash(state)
    report = measure_capacity(state, [1, 2], repeats=2, logical_targets=5, output=tmp_path)
    assert exact_cpu_state_hash(state) == before
    assert report['post_warmup_initial_hash'] == report['original_initial_hash'] == before
    assert report['cost']['passes']['backward'] >= 25  # warmup plus 2 sizes x 2 steps
    assert report['cost']['warmup_seconds'] > 0
    assert report['step_seconds'] and report['reset_seconds'] and report['checkpoint_seconds']
    assert report['queries'][0]['targets'] == report['cost']['passes']['query'] == 5


def test_oom_retry_uses_identical_scored_offsets_and_retains_failed_work(tmp_path, monkeypatch):
    from ralup.calibrate import step_with_oom_retry
    state = fixture()
    state.microbatch = 2
    original = state.method.backward_microbatch
    failed = False
    def oom(training, batch):
        nonlocal failed
        if not failed:
            failed = True
            training.record_work(additional_loss=3, query=3)
            raise RuntimeError('MPS out of memory')
        return original(training, batch)
    monkeypatch.setattr(state.method, 'backward_microbatch', oom)
    result, attempts = step_with_oom_retry(state, 5)
    assert attempts[0]['target_offsets'] == attempts[1]['target_offsets']
    assert state.microbatch == 1
    assert state.ledger.passes['query'] == 3
    assert state.ledger.stream['targets'] == result.target_count == 5
    assert state.ledger.events[0]['status'] == 'failed'


def test_model_only_historical_import_preserves_cursor_rng_and_all_parent_cost(tmp_path):
    from ralup.calibrate import import_warmstart
    state = fixture()
    train_step(state, select_logical_batch(state, 5))
    ref = save_snapshot(capture_snapshot(state), tmp_path / 'parent')
    fresh = fixture()
    parent_cursor = state.cursor.state
    restored = import_warmstart(ref, fresh, partition='meta_train', continuation_targets=7)
    assert restored is fresh
    assert all(torch.equal(p, dict(state.model.named_parameters())[n]) for n, p in fresh.model.named_parameters())
    assert fresh.cursor.state == {**parent_cursor, 'run_target_budget': 12}
    assert fresh.histories == fresh.quantizer_state == {}
    assert fresh.schedule['step'] == 0
    assert all(not value for value in fresh.optimizer_state.values())
    assert fresh.ledger.passes == state.ledger.passes
    assert fresh.configuration['warmstart_parent']['checkpoint']['sha256'] == ref.sha256
    assert fresh.configuration['warmstart_parent']['source_identity']
    assert torch.equal(fresh.rng_streams['quantizer'].get_state(), state.rng_streams['quantizer'].get_state())
    other = import_warmstart(ref, fixture(), partition='meta_train', continuation_targets=7)
    assert other.cursor.state == fresh.cursor.state
    assert select_logical_batch(other, 7) == select_logical_batch(fresh, 7)


def test_wrong_partition_ancestry_and_wrong_model_reject(tmp_path):
    from ralup.calibrate import import_warmstart
    ref = save_snapshot(capture_snapshot(fixture()), tmp_path / 'parent')
    with pytest.raises(IntegrityError, match='partition'):
        import_warmstart(ref, fixture(), partition='target_train', continuation_targets=7)
    fresh = fixture()
    fresh.model = ByteTransformer(ModelConfig(1, 16, 2))
    with pytest.raises(IntegrityError, match='model'):
        import_warmstart(ref, fresh, partition='meta_train', continuation_targets=7)
    fresh = fixture()
    fresh.model.blocks[0].attention_norm.eps = .1
    with pytest.raises(IntegrityError, match='parameterization'):
        import_warmstart(ref, fresh, partition='meta_train', continuation_targets=7)


def test_historical_model_import_does_not_waive_full_resume_runtime_check(tmp_path, monkeypatch):
    import ralup.state as snapshots
    from ralup.calibrate import import_warmstart
    state = fixture()
    ref = save_snapshot(capture_snapshot(state), tmp_path / 'parent')
    identities = snapshots._automatic_identities()
    key = next(iter(identities))
    identities[key] = ArtifactRef(identities[key].path, '0'*64)
    monkeypatch.setattr(snapshots, '_automatic_identities', lambda: identities)
    with pytest.raises(IntegrityError, match='incompatible'):
        snapshots.restore_snapshot(snapshots.load_snapshot(ref), 'cpu')
    imported = import_warmstart(ref, fixture(), partition='meta_train', continuation_targets=7)
    assert all(torch.equal(p, dict(state.model.named_parameters())[n]) for n, p in imported.model.named_parameters())


def test_public_calibration_and_separate_warmstart_cli(tmp_path, capsys):
    from pathlib import Path
    from ralup.__main__ import main
    from ralup.calibrate import import_warmstart
    template = json.loads((Path(__file__).parents[2] / 'configs/smoke.json').read_bytes())
    template['calibration'] = {'output': str(tmp_path/'runs'), 'seed': 808, 'repeats': 2,
                               'pilot_target_budget': 10, 'warmstart_target_budget': 5,
                               'continuation_target_budget': 7, 'doublings': 2}
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'config': template, 'purpose': 'calibration', 'microbatches': [1, 2]}))
    assert main(['calibrate', '--config', str(path)]) == 0
    ref = ArtifactRef(**json.loads(capsys.readouterr().out))
    report = json.loads(ref.path.read_bytes())
    assert report['b0'] is None and report['synthetic'] is True
    from ralup.calibrate import reconcile_calibration
    from ralup.records import sha256_file
    reconciled = reconcile_calibration(ref, ref.path.with_name('reconciled.json'))
    corrected = json.loads(reconciled.path.read_bytes())
    assert sha256_file(ref.path) == ref.sha256
    assert corrected['aggregate_parent']['sha256'] == ref.sha256
    assert corrected['cost']['stream']['unique'] < corrected['cost']['trajectory_unique_sum']
    assert corrected['cost']['events'][-1]['status'] == 'global_coverage_reconciliation'
    assert main(['run', '--phase', 'explore', '--suite', 'warmstart', '--config', str(path)]) == 0
    refs = json.loads(capsys.readouterr().out)
    for partition, value in refs.items():
        record = json.loads(ArtifactRef(**value).path.read_bytes())
        assert record['source_partition'] == partition
        fresh = fixture()
        fresh.cursor = None  # Import authenticates the parent fixture's source itself.
        fresh.configuration['source_partition'] = partition
        imported = import_warmstart(ArtifactRef(**value), fresh, partition=partition, continuation_targets=7)
        assert imported.configuration['warmstart_parent']['parent_cost']['project_seconds'] == record['project_seconds']
        assert imported.ledger.events[-1]['status'] == 'checkpoint_write'


def test_corrupt_producer_archive_rejects_model_import(tmp_path):
    from ralup.calibrate import import_warmstart
    ref = save_snapshot(capture_snapshot(fixture()), tmp_path / 'parent')
    file = next(p for p in (ref.path.parent/'producer').rglob('*') if p.is_file())
    file.write_bytes(file.read_bytes() + b'\n# corrupt\n')
    with pytest.raises(IntegrityError):
        import_warmstart(ref, fixture(), partition='meta_train', continuation_targets=7)


def test_imported_current_snapshot_retains_authenticated_archived_source_refs(tmp_path, monkeypatch):
    import ralup.state as snapshots
    from ralup.calibrate import import_warmstart
    refs, files = snapshots._archive_producer()
    key = next(name for name in files if name.endswith('.py'))
    files[key] += b'\n# prior producer fixture\n'
    refs[key] = ArtifactRef(refs[key].path, hashlib.sha256(files[key]).hexdigest())
    with monkeypatch.context() as patch:
        patch.setattr(snapshots, '_archive_producer', lambda: (refs, files))
        ref = save_snapshot(capture_snapshot(fixture()), tmp_path/'old')
    with pytest.raises(IntegrityError, match='incompatible'):
        snapshots.restore_snapshot(snapshots.load_snapshot(ref), 'cpu')
    imported = import_warmstart(ref, fixture(), partition='meta_train', continuation_targets=7)
    # A continuation snapshot must authenticate stable archived historical bytes,
    # rather than claiming the current source path has its prior digest.
    current = save_snapshot(capture_snapshot(imported), tmp_path/'current')
    assert snapshots.restore_snapshot(snapshots.load_snapshot(current), 'cpu').cursor.state == imported.cursor.state


def test_memory_cap_blocks_warmup_before_any_training(tmp_path):
    from ralup.calibrate import measure_capacity
    state = fixture()
    with pytest.raises(RuntimeError, match='cap'):
        measure_capacity(state, [1, 2], repeats=2, logical_targets=5, output=tmp_path,
                         caps={'resident_cap_bytes': 1, 'driver_cap_bytes': None})
    assert state.cursor.cumulative_target_count == 0
    assert state.schedule['step'] == 0


def test_oom_capacity_remeasures_all_arms_with_common_smaller_packing(tmp_path, monkeypatch):
    from ralup.calibrate import measure_capacity
    original = MethodAdapter.backward_microbatch
    def oom(self, state, batch):
        if state.microbatch > 1: raise RuntimeError('out of memory')
        return original(self, state, batch)
    monkeypatch.setattr(MethodAdapter, 'backward_microbatch', oom)
    report = measure_capacity(fixture(), [1, 2], repeats=2, logical_targets=5, output=tmp_path,
                              recipes=['row_fixed_max', 'row_dynamic_max'])
    assert report['selected_packing'] == 1
    remeasured = [arm for arm in report['arms'] if arm.get('oom_common_packing_remeasurement')]
    assert {arm['recipe'] for arm in remeasured} == {'row_fixed_max', 'row_dynamic_max'}
    assert all(arm['microbatch_actual'] == 1 and len(arm['full_step_seconds']) == 2 for arm in remeasured)
