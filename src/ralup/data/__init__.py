"""Pinned, non-scoring source provenance and byte-preserving articles."""

from .articles import Article, is_article_heading, reconstruct_articles
from .source import fetch_source
from .dedup import audit_duplicates
from .partitions import CorpusManifest, partition_article, prepare_corpus

__all__ = ["Article", "fetch_source", "is_article_heading", "reconstruct_articles",
           "audit_duplicates", "CorpusManifest", "partition_article", "prepare_corpus"]
