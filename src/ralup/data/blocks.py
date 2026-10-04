"""S10 byte blocks: input-only boundary, shifted labels and exact target masks."""

from dataclasses import dataclass
import hashlib

from .articles import Article

BOUNDARY_TOKEN = 256
CONTEXT_BYTES = 128
TARGET_BYTES = 128


@dataclass(frozen=True)
class TargetBlock:
    article_hash: str
    target_start: int
    target_end: int
    inputs: tuple[int, ...]
    labels: tuple[int, ...]
    positions: tuple[int, ...]
    score_mask: tuple[bool, ...]
    # Content hash is dedup identity; this distinguishes independent documents.
    document_id: str = ''

    def scored_offsets(self) -> tuple[int, ...]:
        first_label = 0 if self.target_start == 0 else min(CONTEXT_BYTES, self.target_start) - 1
        return tuple(self.target_start + index - first_label
                     for index, score in enumerate(self.score_mask) if score)


def make_block(article_hash: str, target_start: int, target_end: int,
               context: bytes, targets: bytes, *, document_id: str = '') -> TargetBlock:
    """Materialize one original block; context must come from the same document."""
    if (target_start < 0 or not 0 < target_end - target_start <= TARGET_BYTES
            or len(targets) != target_end - target_start
            or len(context) != min(CONTEXT_BYTES, target_start)):
        raise ValueError('Invalid byte block interval/context')
    prefix = tuple(context) if target_start else (BOUNDARY_TOKEN,)
    sequence = prefix + tuple(targets)
    inputs, labels = sequence[:-1], sequence[1:]
    first_label = len(prefix) - 1
    return TargetBlock(article_hash, target_start, target_end, inputs, labels,
                       tuple(range(len(inputs))),
                       tuple(i >= first_label for i in range(len(labels))),
                       document_id or article_hash)


def tile_article(article: Article) -> list[TargetBlock]:
    """Public fixture/one-article API; real streams materialize individual blocks."""
    if hashlib.sha256(article.content).hexdigest() != article.content_sha256:
        raise ValueError('Article content hash mismatch')
    return [make_block(article.content_sha256, start, min(start + TARGET_BYTES, len(article.content)),
                       article.content[max(0, start - CONTEXT_BYTES):start],
                       article.content[start:start + TARGET_BYTES])
            for start in range(0, len(article.content), TARGET_BYTES)]
