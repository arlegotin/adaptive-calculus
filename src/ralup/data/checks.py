"""Bounded data engineering fixtures; no model, neural result or WP02 gate."""

import hashlib
import json
from pathlib import Path

from ..config import canonical_bytes
from ..records import ArtifactRef, sha256_file
from .articles import Article
from .batches import next_logical_batch, pack_microbatches
from .blocks import tile_article
from .cursor import BatchCursor


def _article(length):
    raw = bytes(index % 256 for index in range(length))
    return Article(raw, [], hashlib.sha256(raw).hexdigest())


def _selected(batch):
    return [(b.document_id, offset) for b in batch.block_slices for offset in b.scored_offsets()]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _checks():
    lengths = [1, 127, 128, 129, 255, 256, 257]
    for length in lengths:
        blocks = tile_article(_article(length))
        _require(sorted(offset for b in blocks for offset in b.scored_offsets()) == list(range(length)), 'Byte coverage differs')
        _require(blocks[0].inputs[0] == 256, 'Missing input boundary')
        for block in blocks:
            _require(all(label < 256 for label in block.labels), 'Boundary target')
            _require(block.scored_offsets() == tuple(range(block.target_start, block.target_end)), 'Context target')
    cursor = BatchCursor.from_articles([_article(257)], seed=4, run_target_budget=263)
    first = next_logical_batch(cursor, 131)
    restored = BatchCursor(cursor.state)
    second = next_logical_batch(cursor, 131)
    _require(second == next_logical_batch(restored, 131), 'Cursor restoration differs')
    carried, continued = first.block_slices[-1], second.block_slices[0]
    _require((carried.inputs, carried.labels, carried.positions, carried.target_start)
             == (continued.inputs, continued.labels, continued.positions, continued.target_start), 'Carry retiled context/position IDs')
    final = next_logical_batch(cursor, 131)
    _require(final.target_count == 1 and next_logical_batch(cursor, 131).target_count == 0, 'Final denominator differs')
    _require(cursor.unique_target_count == 257 and cursor.repeated_target_count == 6, 'Unique/repeated accounting differs')
    one = BatchCursor.from_articles([_article(257), _article(129)], seed=17)
    four = BatchCursor.from_articles([_article(257), _article(129)], seed=17)
    next_logical_batch(one, 131)
    next_logical_batch(four, 131)
    resumed = BatchCursor(one.state)
    for _ in range(9):
        left, right = next_logical_batch(one, 131), next_logical_batch(four, 131)
        _require(left == next_logical_batch(resumed, 131), 'Named data RNG restoration differs across epochs')
        packed_one = [offset for b in pack_microbatches(left, 1) for offset in _selected(b)]
        packed_four = [offset for b in pack_microbatches(right, 4) for offset in _selected(b)]
        _require(packed_one == packed_four, 'Physical packing changes logical selection')
    default = next_logical_batch(BatchCursor.from_articles([_article(9000)], seed=13))
    _require(default.target_count == 8192, 'Initial logical byte budget differs')
    return {'boundary_lengths': lengths, 'complete_once_only_coverage': True,
            'fixed_context_position_carry': True, 'named_data_rng_restoration': True,
            'epoch_rollover': True, 'physical_packing_invariance': True,
            'unique_targets': 257, 'repeated_targets': 6,
            'final_short_denominator': final.target_count,
            'default_logical_target_count': default.target_count}


def check_data(output: Path) -> ArtifactRef:
    report = {'suite': 'data', 'synthetic': True, 'scientific_gate': None,
              'scope': 'H03 byte correctness fixtures only; no model scoring or full WP02',
              'remaining_harness': ['H04-H08 remaining harness'], 'blockers': []}
    try:
        report['checks'] = _checks()
        report['status'] = 'pass'
    except ValueError as error:
        report['status'] = 'fail'
        report['blockers'] = [str(error)]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(report) + b'\n')
    return ArtifactRef(output, sha256_file(output))
