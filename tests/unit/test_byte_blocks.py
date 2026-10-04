"""Catch dropped boundary bytes, context scoring and wrongly shifted labels."""
import hashlib

import pytest

from ralup.data.articles import Article
from ralup.data.blocks import tile_article


def article(length):
    content = bytes(i % 256 for i in range(length))
    return Article(content, [], hashlib.sha256(content).hexdigest())


@pytest.mark.parametrize('length', [1, 127, 128, 129, 255, 256, 257])
def test_every_raw_byte_once_without_context_or_boundary_targets(length):
    source = article(length)
    blocks = tile_article(source)
    offsets = []
    assert blocks[0].inputs[0] == 256
    for block in blocks:
        assert len(block.inputs) == len(block.labels) == len(block.positions) == len(block.score_mask)
        assert len(block.inputs) <= 256
        assert block.positions == tuple(range(len(block.inputs)))
        scored = block.scored_offsets()
        offsets.extend(scored)
        assert scored == tuple(range(block.target_start, block.target_end))
        assert 0 < block.target_end - block.target_start <= 128
        assert all(label != 256 for label in block.labels)
        assert [label for label, mask in zip(block.labels, block.score_mask) if mask] == list(source.content[block.target_start:block.target_end])
        if block.target_start:
            context = source.content[max(0, block.target_start - 128):block.target_start]
            sequence = context + source.content[block.target_start:block.target_end]
            assert block.inputs == tuple(sequence[:-1])
            assert block.labels == tuple(sequence[1:])
            assert not any(block.score_mask[:len(context)-1])
    assert sorted(offsets) == list(range(length))


def test_empty_article_has_no_targets_and_bad_hash_is_rejected():
    assert tile_article(article(0)) == []
    with pytest.raises(ValueError, match='hash'):
        tile_article(Article(b'a', [], 'bad'))
