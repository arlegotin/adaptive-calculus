import hashlib

import pytest

from ralup.data.articles import Article


def article(content):
    raw = content.encode() if isinstance(content, str) else content
    return Article(raw, [], hashlib.sha256(raw).hexdigest())


def test_exact_overlap_preserves_independent_protected_documents():
    from ralup.data.dedup import audit_duplicates
    a, b = article('same'), article('other')
    result = audit_duplicates([a, a, b], [a, a], {})
    assert result['retained_training_hashes'] == [b.content_sha256]
    assert len(result['documents']) == 5
    assert [d['id'] for d in result['documents']] == ['train:0', 'train:1', 'train:2', 'protected:0', 'protected:1']
    assert len(result['protected_documents']) == 2
    assert set(result['retained_training_hashes']).isdisjoint(result['protected_hashes'])
    assert result['excluded_documents'][0]['reason'] == 'protected_component'


def test_connected_chain_excludes_indirect_training_neighbor():
    from ralup.data.dedup import audit_duplicates, shingles
    # Sliding windows overlap at >=.8 only for adjacent nodes; endpoints <.8.
    tokens = [f't{i}' for i in range(250)]
    protected = article(' '.join(tokens[:200]))
    a = article(' '.join(tokens[12:212]))
    b = article(' '.join(tokens[24:224]))
    ps, bs = shingles(protected.content), shingles(b.content)
    assert len(ps & bs) * 5 < len(ps | bs) * 4
    result = audit_duplicates([a, b], [protected], {})
    assert {a.content_sha256, b.content_sha256} <= set(result['excluded_component_hashes'])
    assert not result['retained_training_hashes']
    assert result['candidates'] == audit_duplicates([a, b], [protected], {})['candidates']
    assert any(pair['accepted'] for pair in result['candidates'])


def test_normalization_is_audit_only_and_short_tuple_is_complete():
    from ralup.data.dedup import audit_duplicates, shingles
    original = article('Ａ  É\té\n')
    raw_unicode_bytes = original.content
    protected = article('a é é')
    assert shingles(original.content) == {('a', 'é', 'é')}
    assert shingles(b'') == {()}
    result = audit_duplicates([original], [protected], {})
    assert original.content_sha256 in result['excluded_component_hashes']
    assert raw_unicode_bytes == original.content
    assert original.content != protected.content


def test_canonical_hash_is_lexicographic_and_exact_duplicates_collapse():
    from ralup.data.dedup import audit_duplicates
    a, b = article('A b c'), article('a B c')
    result = audit_duplicates([a, b, a], [], {})
    assert result['retained_training_hashes'] == [min(a.content_sha256, b.content_sha256)]
    assert len(result['retained_documents']) == 1
    assert len(result['exact_groups']) == 1


@pytest.mark.parametrize('value, expected', [
    (0, 'meta-train'), ((2**64 - 1)//5, 'meta-train'),
    ((2**64 + 4)//5, 'meta-development'), (2**62 - 1, 'meta-development'),
    (2**62, 'target-training'), (2**64-1, 'target-training'),
])
def test_partition_exact_rational_thresholds(monkeypatch, value, expected):
    from ralup.data import partitions
    monkeypatch.setattr(partitions, '_partition_hash64', lambda content: value)
    assert partitions.partition_article(b'unmodified') == expected


def test_partition_uses_raw_bytes_and_domain():
    from ralup.data.partitions import partition_article
    raw = 'é'.encode()
    value = int.from_bytes(hashlib.sha256(b'ralup-split-v1' + raw).digest()[:8], 'big')
    expected = 'meta-train' if 5*value < 2**64 else ('meta-development' if 4*value < 2**64 else 'target-training')
    assert partition_article(raw) == expected


def test_exact_jaccard_threshold_and_stable_signatures():
    from ralup.data.dedup import exact_similarity, signature
    assert exact_similarity({('a',), ('b',), ('c',), ('d',)}, {('a',), ('b',), ('c',), ('d',), ('e',)}) == (4, 5, True)
    assert exact_similarity({('a',), ('b',), ('c',)}, {('a',), ('b',), ('c',), ('d',)}) == (3, 4, False)
    assert signature(b'a b c').tolist() == signature(b'a b c').tolist()
    assert len(signature(b'a b c')) == 128


def test_invalid_contract_rejected():
    from ralup.data.dedup import audit_duplicates
    with pytest.raises(ValueError, match='contract'):
        audit_duplicates([], [], {'bands': 8})
    with pytest.raises(ValueError, match='hash'):
        audit_duplicates([Article(b'wrong', [], '0'*64)], [], {})


def test_candidate_hash_collisions_do_not_replace_exact_tuple_jaccard(monkeypatch):
    import numpy as np
    from ralup.data import dedup
    monkeypatch.setattr(dedup, 'signature', lambda content: np.zeros(128, dtype=np.uint64))
    a, b = article('completely unrelated'), article('other tuple')
    result = dedup.audit_duplicates([a], [b], {})
    assert len(result['candidates']) == 1
    assert result['candidates'][0]['intersection'] == 0
    assert result['candidates'][0]['accepted'] is False
    assert result['retained_training_hashes'] == [a.content_sha256]


def test_signatures_do_not_depend_on_python_process_hash_seed():
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from ralup.data.dedup import signature
    command = [sys.executable, '-c', "import json; from ralup.data.dedup import signature; print(json.dumps(signature(b'a b c').tolist()))"]
    root = Path(__file__).parents[2]
    results = [subprocess.check_output(command, cwd=root, env={**os.environ, 'PYTHONPATH': str(root/'src'), 'PYTHONHASHSEED': seed}, text=True) for seed in ('1', '98765')]
    assert json.loads(results[0]) == json.loads(results[1]) == signature(b'a b c').tolist()
