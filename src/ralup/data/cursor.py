"""Safe resumable byte cursor over compact permutations and authenticated indexes.

No physical packing choice affects selection. Corpus content remains on disk;
block ordinal lookup uses per-document cumulative counts, not tensor sequences.
"""

from array import array
import base64
from bisect import bisect_right
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys

from ..config import canonical_bytes
from .articles import Article
from .blocks import CONTEXT_BYTES, TARGET_BYTES, make_block
from .partitions import (PARTITIONS, _authenticate, _indexed_documents,
                         _load_preparation)

ALLOWED_PARTITIONS = (*PARTITIONS, 'development')


def _hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _tuple_tree(value):
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, list) else value


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid cursor integer')
    return value


def _load_indexed(source):
    """Authenticate before using H02 documents; do not reconstruct or deduplicate."""
    partition = source['partition']
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError('Invalid partition; official test targets are protected until E03')
    manifest_path = Path(source['path'])
    _authenticate(manifest_path, source['sha256'])
    payload = json.loads(manifest_path.read_bytes())
    preparation = payload['preparation']
    preparation_path = Path(preparation['path'])
    _authenticate(preparation_path, preparation['sha256'])
    preparation_payload = json.loads(preparation_path.read_bytes())
    report = _load_preparation(preparation_path, preparation_payload['config'], payload['source_config_sha256'])
    if payload['source'] != report['source'] or payload['source_files'] != {k: v['files'] for k, v in report['splits'].items()}:
        raise ValueError('Frozen source references mismatch')
    hashes = {'data_manifest_sha256': source['sha256'],
              'source_sha256': _hash({k: payload[k] for k in ('preparation', 'source', 'source_files', 'revision')}),
              'exclusion_sha256': _hash({'documents': payload['audit']['excluded_documents'],
                                         'hashes': payload['audit']['excluded_component_hashes']}),
              'partition_sha256': _hash({'partitions': payload['partitions'], 'algorithm': payload['partition_algorithm']})}
    if 'hashes' in source and source['hashes'] != hashes:
        raise ValueError('Frozen source/exclusion/partition hashes mismatch')
    # The accepted real frozen identity has independently recorded object hashes.
    if not payload['synthetic']:
        from .source import ROOT
        accepted = json.loads((ROOT / 'docs/data/h02-evidence.json').read_bytes())
        if source['sha256'] != accepted['manifest']['sha256']:
            raise ValueError('Pinned corpus manifest is not the accepted frozen H02 identity')
        if any(hashes[k] != accepted['frozen_hashes'][k] for k in hashes):
            raise ValueError('Accepted H02 frozen hashes mismatch')
    indexed = {doc['id']: doc for doc in _indexed_documents(preparation_path, report)}
    retained = payload['audit']['retained_documents']
    protected = payload['audit']['protected_documents']
    chosen = [d for d in protected if d['split'] == 'validation'] if partition == 'development' else [d for d in retained if d['partition'] == partition]
    seen = set()
    for document in chosen:
        identity = document['id']
        base = indexed.get(identity)
        if (identity in seen or base is None
                or any(document[k] != value for k, value in base.items())
                or (partition == 'development' and document['split'] != 'validation')
                or (partition != 'development' and document['protected'])):
            raise ValueError('Frozen document identity/index mismatch')
        seen.add(identity)
    if partition != 'development':
        item = payload['partitions'][partition]
        if (item['document_ids'] != [d['id'] for d in chosen]
                or item['content_hashes'] != [d['content_sha256'] for d in chosen]
                or item['documents'] != len(chosen)
                or item['bytes'] != sum(d['bytes'] for d in chosen)):
            raise ValueError('Frozen partition membership mismatch')
    # Keep only compact metadata for selected documents; release the 51 MB audit.
    documents = [{k: d[k] for k in ('id', 'content_sha256', 'content_path', 'content_start', 'bytes',
                                   'source_row_start', 'source_row_stop', 'source_shards', 'split')}
                 for d in chosen]
    return documents, hashes


class BatchCursor:
    """Restore from JSON-compatible metadata; fixture content is explicitly local.

    `from_manifest` accepts meta-train/meta-development/target-training or
    development (official validation). Official test has no target interface.
    State includes frozen data hashes, epoch, permutation position, carried score
    interval and the named `data` RNG. H07 owns atomic full-state publication.
    """

    @classmethod
    def from_articles(cls, articles: list[Article], *, seed: int,
                      run_target_budget: int | None = None):
        items = []
        for ordinal, article in enumerate(articles):
            if hashlib.sha256(article.content).hexdigest() != article.content_sha256:
                raise ValueError('Article content hash mismatch')
            items.append({'id': f'fixture:{ordinal}', 'content_hex': article.content.hex(),
                          'content_sha256': article.content_sha256})
        return cls._new({'kind': 'articles', 'articles': items}, seed, run_target_budget)

    @classmethod
    def from_manifest(cls, path: Path, sha256: str, *, partition: str, seed: int,
                      run_target_budget: int | None = None):
        return cls._new({'kind': 'indexed', 'path': str(Path(path).resolve()),
                         'sha256': sha256, 'partition': partition}, seed, run_target_budget)

    @classmethod
    def _new(cls, source, seed, budget):
        _integer(seed)
        if budget is not None:
            _integer(budget)
        instance = cls.__new__(cls)
        instance._source = source
        instance._open_source()
        instance._rng = random.Random(seed)
        instance._permutation = array('I', range(instance.block_count))
        instance._rng.shuffle(instance._permutation)
        instance.epoch = instance.permutation_position = 0
        instance._carry = None
        instance.cumulative_target_count = 0
        instance._run_target_budget = budget
        return instance

    def _open_source(self):
        self.materialized_blocks = self.source_bytes_read = 0
        if self._source['kind'] == 'indexed':
            self._documents, hashes = _load_indexed(self._source)
            self._source = {**self._source, 'hashes': hashes}
            self._content_stats = {d['content_path']: self._stat_key(Path(d['content_path']).stat())
                                   for d in self._documents}
        elif self._source['kind'] == 'articles':
            self._documents = []
            for item in self._source['articles']:
                content = bytes.fromhex(item['content_hex'])
                if hashlib.sha256(content).hexdigest() != item['content_sha256']:
                    raise ValueError('Fixture article hash mismatch')
                self._documents.append({**item, 'content': content, 'bytes': len(content)})
            if len({d['id'] for d in self._documents}) != len(self._documents):
                raise ValueError('Duplicate fixture document identity')
        else:
            raise ValueError('Invalid cursor source kind')
        self._stops = array('Q')
        count = 0
        for doc in self._documents:
            count += (doc['bytes'] + TARGET_BYTES - 1) // TARGET_BYTES
            self._stops.append(count)
        if count >= 2**32:
            raise ValueError('Block count exceeds compact uint32 permutation capacity')
        self.block_count = count
        self.total_target_bytes = sum(d['bytes'] for d in self._documents)

    def __init__(self, state: dict):
        # JSON round-trip also owns the input and rejects non-JSON/executable data.
        state = json.loads(canonical_bytes(state))
        if _integer(state['schema_version']) != 1:
            raise ValueError('Unsupported cursor schema')
        self._source = state['source']
        self._open_source()
        self._permutation = array('I')
        try:
            self._permutation.frombytes(base64.b64decode(state['permutation_u32le'], validate=True))
        except (ValueError, TypeError) as error:
            raise ValueError('Invalid cursor permutation encoding') from error
        if sys.byteorder != 'little':
            self._permutation.byteswap()
        if len(self._permutation) != self.block_count:
            raise ValueError('Cursor permutation length mismatch')
        seen = bytearray(self.block_count)
        for block in self._permutation:
            if block >= self.block_count or seen[block]:
                raise ValueError('Cursor permutation is not without replacement')
            seen[block] = 1
        self.epoch = _integer(state['epoch'])
        self.permutation_position = _integer(state['permutation_position'])
        if self.permutation_position > self.block_count:
            raise ValueError('Invalid permutation position')
        self._carry = state['carry']
        partial = 0
        if self._carry is not None:
            if self.permutation_position >= self.block_count:
                raise ValueError('Carry outside epoch')
            block_index = self._permutation[self.permutation_position]
            _, start, end = self._reference(block_index)
            score_start = _integer(self._carry['score_start'])
            if (_integer(self._carry['block_index']) != block_index or not start < score_start < end
                    or _integer(self._carry['score_end']) != end):
                raise ValueError('Invalid carried score interval')
            partial = score_start - start
        self.cumulative_target_count = _integer(state['cumulative_target_count'])
        epoch_targets = partial
        for position in range(self.permutation_position):
            _, start, end = self._reference(self._permutation[position])
            epoch_targets += end - start
        if self.cumulative_target_count != self.epoch * self.total_target_bytes + epoch_targets:
            raise ValueError('Cursor target counters mismatch')
        if (_integer(state['unique_target_count']) != self.unique_target_count
                or _integer(state['repeated_target_count']) != self.repeated_target_count):
            raise ValueError('Cursor coverage counters mismatch')
        self._run_target_budget = state['run_target_budget']
        if self._run_target_budget is not None:
            _integer(self._run_target_budget)
            if self.cumulative_target_count > self._run_target_budget:
                raise ValueError('Cursor exceeds final run budget')
        self._rng = random.Random()
        try:
            self._rng.setstate(_tuple_tree(state['rng_states']['data']))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError('Invalid named data RNG state') from error

    @property
    def unique_target_count(self):
        # Exact because an epoch is a permutation of disjoint byte intervals.
        return min(self.cumulative_target_count, self.total_target_bytes)

    @property
    def repeated_target_count(self):
        return self.cumulative_target_count - self.unique_target_count

    @property
    def remaining_run_targets(self):
        return None if self._run_target_budget is None else self._run_target_budget - self.cumulative_target_count

    @property
    def state(self) -> dict:
        permutation = array('I', self._permutation)
        if sys.byteorder != 'little':
            permutation.byteswap()
        state = {'schema_version': 1, 'source': self._source,
                 'epoch': self.epoch, 'permutation_position': self.permutation_position,
                 'carry': self._carry, 'permutation_u32le': base64.b64encode(permutation.tobytes()).decode('ascii'),
                 'rng_states': {'data': json.loads(json.dumps(self._rng.getstate()))},
                 'cumulative_target_count': self.cumulative_target_count,
                 'unique_target_count': self.unique_target_count,
                 'repeated_target_count': self.repeated_target_count,
                 'run_target_budget': self._run_target_budget}
        return json.loads(canonical_bytes(state))

    def _reference(self, ordinal):
        document_index = bisect_right(self._stops, ordinal)
        previous = self._stops[document_index - 1] if document_index else 0
        document = self._documents[document_index]
        start = (ordinal - previous) * TARGET_BYTES
        return document, start, min(start + TARGET_BYTES, document['bytes'])

    def take(self, budget):
        """Consume one score slice, retaining its original context/position IDs."""
        _integer(budget, 1)
        if not self.block_count or self.remaining_run_targets == 0:
            raise ValueError('Exhausted cursor')
        if self.permutation_position == self.block_count:
            self.epoch += 1
            self.permutation_position = 0
            self._rng.shuffle(self._permutation)
        ordinal = self._permutation[self.permutation_position]
        document, start, end = self._reference(ordinal)
        score_start = self._carry['score_start'] if self._carry else start
        selected = min(budget, end - score_start)
        if self.remaining_run_targets is not None:
            selected = min(selected, self.remaining_run_targets)
        context_start = max(0, start - CONTEXT_BYTES)
        if self._source['kind'] == 'indexed':
            with Path(document['content_path']).open('rb') as stream:
                stream.seek(document['content_start'] + context_start)
                expected_stat = self._content_stats[document['content_path']]
                if self._stat_key(os.fstat(stream.fileno())) != expected_stat:
                    raise ValueError('Authenticated indexed content changed since initialization')
                raw = stream.read(end - context_start)
                if self._stat_key(os.fstat(stream.fileno())) != expected_stat:
                    raise ValueError('Authenticated indexed content changed during read')
            if len(raw) != end - context_start:
                raise ValueError('Indexed source byte slice truncated')
            self.source_bytes_read += len(raw)
        else:
            raw = document['content'][context_start:end]
        document_id = document['id']
        if self._source['kind'] == 'indexed':
            document_id = self._source['hashes']['source_sha256'] + ':' + document_id
        block = make_block(document['content_sha256'], start, end,
                           raw[:start-context_start], raw[start-context_start:], document_id=document_id)
        mask = tuple(score and score_start <= offset < score_start + selected
                     for score, offset in zip(block.score_mask, self._label_offsets(block)))
        block = replace(block, score_mask=mask)
        self.materialized_blocks += 1
        self.cumulative_target_count += selected
        if score_start + selected == end:
            self._carry = None
            self.permutation_position += 1
        else:
            self._carry = {'block_index': ordinal, 'score_start': score_start + selected, 'score_end': end}
        return block

    @staticmethod
    def _stat_key(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @staticmethod
    def _label_offsets(block):
        first = 0 if block.target_start == 0 else min(CONTEXT_BYTES, block.target_start) - 1
        return range(block.target_start - first, block.target_start - first + len(block.labels))
