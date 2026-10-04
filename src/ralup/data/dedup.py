"""Non-scoring duplicate audit; normalized tuples never replace raw content."""

from collections import defaultdict
import hashlib
import struct
import unicodedata

import numpy as np

from .articles import Article

CONTRACT = {'normalization': ['NFKC', 'lowercase', 'whitespace'], 'shingle_tokens': 13,
            'minhash_functions': 128, 'bands': 16, 'rows_per_band': 8,
            'jaccard_threshold': 0.8}
SEED = b'ralup-minhash-v1'
ALGORITHM = {
    'seed_hex': SEED.hex(),
    'coefficients': 'SHA256(seed || uint32_le(i)), i=0..127; a=uint64_le(digest[0:8])|1, b=uint64_le(digest[8:16])',
    'shingle_hash': 'first uint64_le(SHA256(b"ralup-shingle-v1" || concat(uint32_le(UTF8_token_length), UTF8_token)))',
    'permutation': '(a*x+b) modulo 2**64; odd a is a bijection; minimum per function',
    'band_key': 'band ordinal and 8 consecutive uint64_le minima',
    'verification': 'exact normalized token-tuple set intersection/union; accept iff 5*intersection >= 4*union',
    'limitation': 'LSH may miss near duplicates; affine permutations and 64-bit shingle hashing are candidate heuristics. Exact verification uses full tuples. No complete absence of public-data contamination is claimed.',
}


def _contract(config):
    if config and config != CONTRACT:
        raise ValueError('Duplicate audit config differs from the S09 contract')


def _tokens(content):
    return unicodedata.normalize('NFKC', content.decode('utf-8')).lower().split()


def shingles(content: bytes) -> set[tuple[str, ...]]:
    tokens = _tokens(content)
    if len(tokens) < 13:
        return {tuple(tokens)}
    return {tuple(tokens[i:i+13]) for i in range(len(tokens)-12)}


def signature(content: bytes):
    """Chunk arithmetic at 4096 shingles; retain only one document's tokens."""
    coefficients = [hashlib.sha256(SEED + struct.pack('<I', i)).digest() for i in range(128)]
    a = np.array([int.from_bytes(c[:8], 'little') | 1 for c in coefficients], dtype=np.uint64)
    b = np.array([int.from_bytes(c[8:16], 'little') for c in coefficients], dtype=np.uint64)
    tokens = [t.encode('utf-8') for t in _tokens(content)]
    framed = [struct.pack('<I', len(t)) + t for t in tokens]
    count = max(1, len(tokens)-12)
    result = np.full(128, 2**64-1, dtype=np.uint64)
    for start in range(0, count, 4096):
        values = []
        for index in range(start, min(start+4096, count)):
            raw = b''.join(framed[index:index+13]) if len(tokens) >= 13 else b''.join(framed)
            values.append(int.from_bytes(hashlib.sha256(b'ralup-shingle-v1' + raw).digest()[:8], 'little'))
        x = np.array(values, dtype=np.uint64)
        minima = (x[:, None] * a[None, :] + b[None, :]).min(axis=0)
        np.minimum(result, minima, out=result)
    return result


def exact_similarity(left, right):
    intersection = len(left & right)
    union = len(left) + len(right) - intersection
    return intersection, union, 5 * intersection >= 4 * union


class _Components:
    def __init__(self, count):
        self.parent = list(range(count))

    def root(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def audit_indexed(documents: list[dict], read_content, config: dict) -> dict:
    """Audit authenticated metadata and load bytes only for signatures/candidates."""
    _contract(config)
    components = _Components(len(documents))
    hashes = defaultdict(list)
    for index, document in enumerate(documents):
        hashes[document['content_sha256']].append(index)
    exact_groups = []
    representatives = []
    for raw_hash, indices in sorted(hashes.items()):
        representatives.append(indices[0])
        if len(indices) > 1:
            exact_groups.append({'content_sha256': raw_hash, 'documents': [documents[i]['id'] for i in indices]})
            for i in indices[1:]:
                components.join(indices[0], i)
    buckets = defaultdict(list)
    candidate_pairs = set()
    for index in sorted(representatives):
        sig = signature(read_content(documents[index]))
        for band in range(16):
            key = (band, sig[band*8:(band+1)*8].astype('<u8', copy=False).tobytes())
            for other in buckets[key]:
                # Protected-protected edges matter: they can bridge training nodes.
                candidate_pairs.add((other, index))
            buckets[key].append(index)
    del buckets
    candidates = []
    for left, right in sorted(candidate_pairs):
        left_set = shingles(read_content(documents[left]))
        right_set = shingles(read_content(documents[right]))
        intersection, union, accepted = exact_similarity(left_set, right_set)
        del left_set, right_set
        candidates.append({'left': documents[left]['id'], 'right': documents[right]['id'],
                           'intersection': intersection, 'union': union,
                           'jaccard': intersection / union, 'accepted': accepted})
        if accepted:
            components.join(left, right)
    groups = defaultdict(list)
    for index in range(len(documents)):
        groups[components.root(index)].append(index)
    retained, excluded, summaries = [], [], []
    for indices in groups.values():
        protected = [i for i in indices if documents[i]['protected']]
        training = [i for i in indices if not documents[i]['protected']]
        canonical = min(training, key=lambda i: (documents[i]['content_sha256'], documents[i]['id'])) if training else None
        summaries.append({'documents': [documents[i]['id'] for i in indices],
                          'protected': [documents[i]['id'] for i in protected],
                          'canonical_hash': documents[canonical]['content_sha256'] if canonical is not None and not protected else None})
        for i in training:
            if protected or i != canonical:
                excluded.append({**documents[i], 'reason': 'protected_component' if protected else 'duplicate_component',
                                 'canonical_hash': None if protected else documents[canonical]['content_sha256'],
                                 'component': [documents[j]['id'] for j in indices]})
            else:
                retained.append(documents[i])
    retained.sort(key=lambda d: d['id'])
    excluded.sort(key=lambda d: d['id'])
    protected_documents = [d for d in documents if d['protected']]
    retained_hashes = sorted({d['content_sha256'] for d in retained})
    protected_hashes = sorted({d['content_sha256'] for d in protected_documents})
    if not set(retained_hashes).isdisjoint(protected_hashes):
        raise ValueError('Retained training/protected exact overlap')
    return {'algorithm': ALGORITHM, 'contract': CONTRACT, 'documents': documents,
            'exact_groups': exact_groups, 'candidates': candidates, 'components': summaries,
            'retained_documents': retained, 'excluded_documents': excluded,
            'protected_documents': protected_documents, 'retained_training_hashes': retained_hashes,
            'protected_hashes': protected_hashes,
            'excluded_component_hashes': sorted({d['content_sha256'] for d in excluded}),
            'exact_retained_protected_overlap': 0}


def audit_duplicates(train: list[Article], protected: list[Article], config: dict) -> dict:
    """Public fixture/small-corpus API, preserving distinct document ordinals."""
    documents, contents = [], {}
    for split, articles, is_protected in [('train', train, False), ('protected', protected, True)]:
        for index, article in enumerate(articles):
            if hashlib.sha256(article.content).hexdigest() != article.content_sha256:
                raise ValueError('Article content hash mismatch')
            identity = f'{split}:{index}'
            contents[identity] = article.content
            documents.append({'id': identity, 'split': split, 'article_index': index,
                              'content_sha256': article.content_sha256, 'bytes': len(article.content),
                              'source_rows': article.source_rows, 'protected': is_protected})
    return audit_indexed(documents, lambda d: contents[d['id']], config)
