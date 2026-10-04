"""Catch retiled carries, repeated epoch offsets and RNG loss on restore."""
import json

import pytest

from ralup.data.batches import next_logical_batch, pack_microbatches
from ralup.data.cursor import BatchCursor
from test_byte_blocks import article


def selected(batch):
    return [(b.document_id, offset) for b in batch.block_slices for offset in b.scored_offsets()]


@pytest.mark.parametrize('seed', [0, 4])
def test_split_carry_preserves_original_inputs_labels_positions(seed):
    cursor = BatchCursor.from_articles([article(257)], seed=seed)
    first = next_logical_batch(cursor, 131)
    assert first.target_count == 131
    carried = first.block_slices[-1]
    second = next_logical_batch(cursor, 126)
    continuation = second.block_slices[0]
    assert continuation.inputs == carried.inputs
    assert continuation.labels == carried.labels
    assert continuation.positions == carried.positions
    assert continuation.target_start == carried.target_start
    assert continuation.scored_offsets()[0] == carried.scored_offsets()[-1] + 1
    assert sorted(offset for _, offset in selected(first) + selected(second)) == list(range(257))
    assert cursor.unique_target_count == 257
    assert cursor.cumulative_target_count == 257


def test_json_restoration_matches_carry_and_future_epoch_rng():
    original = BatchCursor.from_articles([article(257), article(129)], seed=17)
    next_logical_batch(original, 131)
    saved = json.loads(json.dumps(original.state))
    resumed = BatchCursor(saved)
    for _ in range(9):
        left, right = next_logical_batch(original, 131), next_logical_batch(resumed, 131)
        assert selected(left) == selected(right)
        assert left == right
    assert original.state == resumed.state
    assert original.unique_target_count == 386
    assert original.cumulative_target_count == 1310
    assert original.repeated_target_count == 924


def test_final_run_budget_has_short_denominator_and_stops():
    cursor = BatchCursor.from_articles([article(257)], seed=4, run_target_budget=263)
    assert next_logical_batch(cursor, 131).target_count == 131
    assert next_logical_batch(cursor, 131).target_count == 131
    last = next_logical_batch(cursor, 131)
    assert last.target_count == 1
    assert next_logical_batch(cursor, 131).target_count == 0
    assert cursor.epoch == 1
    assert cursor.unique_target_count == 257
    assert cursor.repeated_target_count == 6


def test_complete_epoch_and_independent_identical_documents():
    cursor = BatchCursor.from_articles([article(257), article(257)], seed=11)
    first = next_logical_batch(cursor, 514)
    assert len(set(selected(first))) == 514
    assert sorted(o for d, o in selected(first) if d == 'fixture:0') == list(range(257))
    assert sorted(o for d, o in selected(first) if d == 'fixture:1') == list(range(257))
    second = next_logical_batch(cursor, 514)
    assert set(selected(second)) == set(selected(first))
    assert cursor.epoch == 1
    assert cursor.unique_target_count == 514
    assert cursor.cumulative_target_count == 1028


def test_physical_packing_does_not_change_logical_schedule():
    one = BatchCursor.from_articles([article(257), article(129)], seed=9)
    four = BatchCursor.from_articles([article(257), article(129)], seed=9)
    for _ in range(8):
        left, right = next_logical_batch(one, 131), next_logical_batch(four, 131)
        packing_one_selected_offsets = [pair for micro in pack_microbatches(left, 1) for pair in selected(micro)]
        packing_four_selected_offsets = [pair for micro in pack_microbatches(right, 4) for pair in selected(micro)]
        assert packing_one_selected_offsets == packing_four_selected_offsets
        assert sum(m.target_count for m in pack_microbatches(left, 1)) == left.target_count
    assert one.state == four.state


@pytest.mark.parametrize('budget', [0, -1, True, 1.5])
def test_invalid_budgets_do_not_move_cursor(budget):
    cursor = BatchCursor.from_articles([article(1)], seed=1)
    saved = cursor.state
    with pytest.raises(ValueError):
        next_logical_batch(cursor, budget)
    assert cursor.state == saved


def test_state_corruption_rejected():
    cursor = BatchCursor.from_articles([article(257)], seed=4)
    next_logical_batch(cursor, 131)
    saved = cursor.state
    saved['carry']['score_start'] = 129
    with pytest.raises(ValueError):
        BatchCursor(saved)


def frozen_fixture(tmp_path):
    from ralup.data.partitions import prepare_bootstrap
    rows = tmp_path / 'rows.json'
    rows.write_text(json.dumps({'rows': ['= A =', 'first original article ' * 40, '= B =', 'independent other article ' * 30]}))
    config = tmp_path / 'source.json'
    config.write_text(json.dumps({'kind': 'synthetic', 'rows_path': str(rows), 'output': str(tmp_path / 'provenance')}))
    return prepare_bootstrap(config)


def test_indexed_cursor_authenticates_and_reads_only_current_blocks(tmp_path):
    manifest = frozen_fixture(tmp_path)
    partition = next(k for k, v in manifest.payload['partitions'].items() if v['documents'])
    cursor = BatchCursor.from_manifest(manifest.path, manifest.sha256, partition=partition, seed=4)
    assert cursor.materialized_blocks == 0
    first = next_logical_batch(cursor, 131)
    assert first.target_count == 131
    assert cursor.materialized_blocks <= 3
    assert cursor.source_bytes_read <= cursor.materialized_blocks * 256
    assert cursor.state['source']['kind'] == 'indexed'
    assert all(b.document_id.startswith(cursor.state['source']['hashes']['source_sha256'] + ':synthetic:') for b in first.block_slices)
    resumed = BatchCursor(json.loads(json.dumps(cursor.state)))
    assert next_logical_batch(cursor, 131) == next_logical_batch(resumed, 131)


@pytest.mark.parametrize('artifact', ['manifest', 'source', 'content.bin', 'rows.bin', 'articles.jsonl'])
def test_indexed_corruption_fails_before_block_construction(tmp_path, artifact):
    from pathlib import Path
    manifest = frozen_fixture(tmp_path)
    if artifact == 'manifest':
        path = manifest.path
    elif artifact == 'source':
        path = Path(manifest.payload['source']['path'])
    else:
        path = Path(manifest.payload['preparation']['path']).parent / ('synthetic.' + artifact)
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match='hash|size'):
        BatchCursor.from_manifest(manifest.path, manifest.sha256, partition='target-training', seed=4)


def test_official_test_selector_is_closed_before_source_io():
    with pytest.raises(ValueError, match='protected|partition'):
        BatchCursor.from_manifest('does-not-exist', '0' * 64, partition='test', seed=4)


def test_default_logical_batch_has_8192_scored_bytes():
    cursor = BatchCursor.from_articles([article(9000)], seed=13)
    assert next_logical_batch(cursor).target_count == 8192


def test_data_cli_reports_fixture_scope_and_pending_harness(tmp_path, capsys):
    from ralup.__main__ import main
    output = tmp_path / 'checks.json'
    assert main(['test-harness', '--suite', 'data', '--output', str(output)]) == 0
    response = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_bytes())
    assert response['status'] == 'pass'
    assert report['synthetic'] is True
    assert report['scientific_gate'] is None
    assert report['remaining_harness'] == ['H04-H08 remaining harness']
    assert report['checks']['final_short_denominator'] == 1
    assert report['checks']['default_logical_target_count'] == 8192


def test_indexed_source_change_after_initialization_is_rejected(tmp_path):
    from pathlib import Path
    manifest = frozen_fixture(tmp_path)
    partition = next(k for k, v in manifest.payload['partitions'].items() if v['documents'])
    cursor = BatchCursor.from_manifest(manifest.path, manifest.sha256, partition=partition, seed=4)
    path = Path(manifest.payload['preparation']['path']).parent / 'synthetic.content.bin'
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match='changed'):
        next_logical_batch(cursor, 131)


@pytest.mark.parametrize('field', ['schema_version', 'unique_target_count'])
def test_boolean_cursor_metadata_is_not_an_integer(field):
    cursor = BatchCursor.from_articles([article(1)], seed=1)
    next_logical_batch(cursor, 1)
    saved = cursor.state
    saved[field] = True
    with pytest.raises(ValueError):
        BatchCursor(saved)


def test_boolean_carried_block_index_is_rejected():
    cursor = BatchCursor.from_articles([article(257)], seed=4)
    next_logical_batch(cursor, 131)
    saved = cursor.state
    assert saved['carry']['block_index'] == 0
    saved['carry']['block_index'] = False
    with pytest.raises(ValueError):
        BatchCursor(saved)
