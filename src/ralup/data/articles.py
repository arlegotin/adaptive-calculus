"""Exact UTF-8 article construction and compact, reversible provenance storage."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import struct
from typing import Iterable, Iterator

from ..config import canonical_bytes
from ..records import sha256_file

# One record per source row, in order: absolute content offset, raw byte length,
# appended LF flag. Original rows are slices of content, excluding that LF.
ROW_RECORD = struct.Struct("<QQB")


@dataclass(frozen=True)
class Article:
    content: bytes
    source_rows: list[dict]
    content_sha256: str


def is_article_heading(text: str) -> bool:
    tokens = text.split()
    return (len(tokens) >= 3 and tokens[0] == tokens[-1] == "="
            and "=" not in tokens[1:-1])


def iter_articles(rows: Iterable[str]) -> Iterator[Article]:
    """Keep source order, including a preamble, buffering only one article."""
    content = bytearray()
    offsets = []
    for index, text in enumerate(rows):
        if not isinstance(text, str):
            raise ValueError("Source rows must be strings")
        if is_article_heading(text) and offsets:
            result = bytes(content)
            yield Article(result, offsets, hashlib.sha256(result).hexdigest())
            content, offsets = bytearray(), []
        raw = text.encode("utf-8")
        start = len(content)
        content.extend(raw)
        appended = not raw.endswith(b"\n")
        offsets.append({"row_index": index, "start": start, "end": len(content),
                        "appended_lf": appended, "raw_sha256": hashlib.sha256(raw).hexdigest()})
        if appended:
            content.append(10)
    if offsets:
        result = bytes(content)
        yield Article(result, offsets, hashlib.sha256(result).hexdigest())


def reconstruct_articles(rows: Iterable[str]) -> list[Article]:
    """Public in-memory interface; corpus preparation uses the streaming form."""
    return list(iter_articles(rows))


def _audit_row(digest, raw: bytes):
    # Length framing preserves row boundaries, including zero-length rows.
    digest.update(struct.pack("<Q", len(raw)))
    digest.update(raw)


def write_reconstruction(rows: Iterable[str], destination: Path, *, split: str) -> dict:
    """Write local non-scoring provenance, then independently reverse every row.

    Storage: content once, compact binary row index, and article JSONL index.
    No target blocks, partitions or model/scoring interface are constructed.
    """
    if not re.fullmatch(r"[a-z]+", split):
        raise ValueError("Invalid split name")
    destination = Path(destination)
    content_path = destination / f"{split}.content.bin"
    row_path = destination / f"{split}.rows.bin"
    article_path = destination / f"{split}.articles.jsonl"
    if any(path.exists() or path.is_symlink() for path in (content_path, row_path, article_path)):
        raise ValueError("Refusing existing reconstruction generation; use a fresh destination")
    destination.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256()
    raw_bytes = row_count = article_count = content_bytes = 0
    source_count = 0

    def audited_rows():
        nonlocal raw_bytes, source_count
        for text in rows:
            if not isinstance(text, str):
                raise ValueError("Source rows must be strings")
            raw = text.encode("utf-8")
            _audit_row(source_digest, raw)
            raw_bytes += len(raw)
            source_count += 1
            yield text

    with content_path.open("xb") as content_file, row_path.open("xb") as row_file, article_path.open("xb") as article_file:
        for article in iter_articles(audited_rows()):
            first_row = article.source_rows[0]["row_index"]
            for row in article.source_rows:
                raw = article.content[row["start"]:row["end"]]
                row_count += 1
                row_file.write(ROW_RECORD.pack(content_bytes + row["start"], len(raw), row["appended_lf"]))
            article_file.write(canonical_bytes({
                "article_index": article_count, "content_start": content_bytes,
                "content_end": content_bytes + len(article.content),
                "source_row_start": first_row, "source_row_stop": row_count,
                "is_preamble": first_row == 0 and not is_article_heading(
                    article.content[:article.source_rows[0]["end"]].decode("utf-8")),
                "content_sha256": article.content_sha256,
            }) + b"\n")
            content_file.write(article.content)
            content_bytes += len(article.content)
            article_count += 1
    reconstruction_digest = hashlib.sha256()
    with content_path.open("rb") as content_file, row_path.open("rb") as row_file:
        expected_offset = 0
        for _ in range(row_count):
            offset, length, appended = ROW_RECORD.unpack(row_file.read(ROW_RECORD.size))
            if offset != expected_offset or appended not in (0, 1):
                raise ValueError("Invalid reversible row offset/flag")
            raw = content_file.read(length)
            if len(raw) != length or (appended and content_file.read(1) != b"\n"):
                raise ValueError("Invalid reversible row bytes/separator")
            _audit_row(reconstruction_digest, raw)
            expected_offset += length + appended
        if content_file.read(1) or row_file.read(1):
            raise ValueError("Unexpected reconstruction tail")
    if row_count != source_count or reconstruction_digest.digest() != source_digest.digest():
        raise ValueError("Raw-row reconstruction audit mismatch")
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in (content_path, row_path, article_path)]
    return {"purpose": "non_scoring_provenance_audit", "rows": row_count,
            "articles": article_count, "raw_utf8_bytes": raw_bytes,
            "derived_content_bytes": content_bytes, "row_index_bytes": row_path.stat().st_size,
            "derived_total_bytes": sum(f["bytes"] for f in files),
            "source_row_audit_sha256": source_digest.hexdigest(),
            "reconstruction_audit_sha256": reconstruction_digest.hexdigest(),
            "audit_format": "sha256(concat(uint64_le(raw_length), raw_utf8))",
            "row_index_format": "<QQB: content_offset, raw_length, appended_lf; ordinal implicit",
            "files": files}
