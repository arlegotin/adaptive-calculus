"""Bounded real H03 indexed evidence, run from the repository root with .venv."""
import hashlib
import json
from pathlib import Path
import resource
import time

from ralup.config import canonical_bytes
from ralup.data.batches import next_logical_batch, pack_microbatches
from ralup.data.cursor import BatchCursor
from ralup.records import sha256_file

started = time.perf_counter()
accepted = json.loads(Path('docs/data/h02-evidence.json').read_bytes())
reference = accepted['manifest']
path = Path(reference['path'])
assert path.stat().st_size == reference['bytes']
assert sha256_file(path) == reference['sha256']
init_started = time.perf_counter()
cursor = BatchCursor.from_manifest(path, reference['sha256'], partition='target-training',
                                   seed=20261004, run_target_budget=16515)
init_seconds = time.perf_counter() - init_started
assert cursor.materialized_blocks == cursor.source_bytes_read == 0
permutation_bytes = cursor._permutation.itemsize * len(cursor._permutation)
state = cursor.state
restoration_started = time.perf_counter()
restored = BatchCursor(json.loads(json.dumps(state)))
restore_seconds = time.perf_counter() - restoration_started
# Independent article-index offsets, not cursor lookup or tile_article.
index = {}
preparation = json.loads(Path(accepted['preparation']['path']).read_bytes())
content_path = Path(accepted['preparation']['path']).parent / 'train.content.bin'
for line in (content_path.parent / 'train.articles.jsonl').read_bytes().splitlines():
    record = json.loads(line)
    index['train:' + str(record['article_index'])] = record
selection_digest = hashlib.sha256()
selected = set()  # Bounded 16,515 selected targets only, never a corpus coverage set.
target_counts = []
with content_path.open('rb') as stream:
    for budget in (8192, 8192, 8192):
        batch = next_logical_batch(cursor, budget)
        replayed = next_logical_batch(restored, budget)
        assert batch == replayed
        one = [(b.document_id, o) for m in pack_microbatches(batch, 1) for b in m.block_slices for o in b.scored_offsets()]
        four = [(b.document_id, o) for m in pack_microbatches(batch, 4) for b in m.block_slices for o in b.scored_offsets()]
        assert one == four
        target_counts.append(batch.target_count)
        assert len(one) == batch.target_count
        for block in batch.block_slices:
            identity = block.document_id.removeprefix(state['source']['hashes']['source_sha256'] + ':')
            record = index[identity]
            start = block.target_start
            end = block.target_end
            assert start % 128 == 0 and end == min(start + 128, record['content_end'] - record['content_start'])
            context_start = max(0, start - 128)
            stream.seek(record['content_start'] + context_start)
            raw = stream.read(end - context_start)
            sequence = ((256,) if start == 0 else tuple(raw[:start-context_start])) + tuple(raw[start-context_start:])
            assert block.inputs == sequence[:-1] and block.labels == sequence[1:]
            assert block.positions == tuple(range(len(block.inputs)))
            for offset in block.scored_offsets():
                key = block.document_id, offset
                assert key not in selected
                selected.add(key)
                selection_digest.update(canonical_bytes({'document_id': block.document_id, 'offset': offset}) + b'\n')
assert target_counts == [8192, 8192, 131]
assert next_logical_batch(cursor).target_count == 0
assert cursor.unique_target_count == cursor.cumulative_target_count == 16515
assert cursor.repeated_target_count == 0
assert cursor.state == restored.state
assert sha256_file(path) == reference['sha256']
development = BatchCursor.from_manifest(path, reference['sha256'], partition='development', seed=20261004,
                                        run_target_budget=131)
assert len(development._documents) == accepted['audit']['protected_by_split']['validation'] == 61
assert len({d['id'] for d in development._documents}) == 61
assert all(d['split'] == 'validation' for d in development._documents)
assert development.total_target_bytes == 1145547
development_batch = next_logical_batch(development)
assert development_batch.target_count == 131
assert all(':validation:' in b.document_id for b in development_batch.block_slices)
report = {'task': 'H03', 'scope': 'bounded real indexed data correctness; no model scoring or scientific training gate',
          'manifest': reference, 'hashes': state['source']['hashes'], 'partition': 'target-training',
          'source_documents': len(cursor._documents), 'source_bytes': cursor.total_target_bytes,
          'block_references': cursor.block_count, 'permutation_bytes': permutation_bytes,
          'materialized_blocks_at_initialization': 0, 'tensor_sequences_at_initialization': 0,
          'materialized_selected_slices': cursor.materialized_blocks, 'selected_source_bytes_read': cursor.source_bytes_read,
          'development': {'documents': len(development._documents), 'source_bytes': development.total_target_bytes,
                          'selected_targets': development_batch.target_count,
                          'independent_protected_document_ids': True, 'official_test_targets_constructed': 0},
          'target_counts': target_counts, 'unique_target_count': cursor.unique_target_count,
          'repeated_target_count': cursor.repeated_target_count,
          'selected_offsets_sha256': selection_digest.hexdigest(),
          'cursor_state_sha256': hashlib.sha256(canonical_bytes(cursor.state)).hexdigest(),
          'serialized_cursor_bytes': len(canonical_bytes(cursor.state)),
          'resources': {'initialization_seconds': round(init_seconds, 4),
                        'restoration_seconds': round(restore_seconds, 4),
                        'wall_seconds': round(time.perf_counter() - started, 4),
                        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'rss_scope': 'macOS ru_maxrss bytes; whole process including authenticated manifest parsing, two target-training cursors and one development cursor; no tensor allocation'},
          'checks': ['all pinned source/preparation/derived and frozen H02 hashes authenticated',
                     'only current selected blocks materialized; no article strings or all tensors',
                     'independent raw-index shift/context/position/offset comparison',
                     'JSON cursor restoration exact', 'packing one/four preserves byte schedule',
                     'final run batch 131 exact targets', 'frozen corpus manifest unchanged', 'all 61 official development documents retained independently; test targets closed'],
          'scientific_gate': None, 'remaining_harness': ['H04-H08 remaining harness']}
output = Path('docs/data/h03-indexed-evidence.json')
output.write_bytes(canonical_bytes(report) + b'\n')
print(json.dumps(report, indent=2))
