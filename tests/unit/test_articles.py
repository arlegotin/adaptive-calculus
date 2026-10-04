import hashlib
import json
from pathlib import Path

import pytest

from ralup.data.articles import is_article_heading, reconstruct_articles, write_reconstruction
from ralup.data.source import fetch_source, iter_source_rows, prepare_data
from ralup.records import sha256_file

FIXTURE = Path(__file__).parents[1] / "fixtures/wikitext_rows.json"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


@pytest.mark.parametrize("text, expected", [
    (" = Title = ", True), ("= = Section = =", False),
    ("== Section ==", False), ("= =", False), ("=", False),
    (" = 尾 title =\r\n", True), ("= A=B =", True),
])
def test_heading_tokens(text, expected):
    assert is_article_heading(text) is expected


def test_exact_articles_and_reversible_offsets():
    rows = json.loads(FIXTURE.read_text())["rows"]
    articles = reconstruct_articles(iter(rows))
    assert [a.content for a in articles] == [
        "Preamble é\n\n".encode(),
        " = Title = \né\r\nalready\n= = Section = =\n== Section ==\n".encode(),
        " = Title = \n尾\n".encode(),
    ]
    recovered = []
    for article in articles:
        assert article.content_sha256 == hashlib.sha256(article.content).hexdigest()
        for row in article.source_rows:
            raw = article.content[row["start"]:row["end"]]
            assert hashlib.sha256(raw).hexdigest() == row["raw_sha256"]
            assert article.content[row["end"]:row["end"] + row["appended_lf"]] == (b"\n" if row["appended_lf"] else b"")
            recovered.append(raw.decode())
    assert recovered == rows
    assert [a.source_rows[0]["row_index"] for a in articles] == [0, 2, 7]
    assert reconstruct_articles(["é", ""])[0].content == "é\n\n".encode()
    assert reconstruct_articles([]) == []


def test_streaming_manifest_audits_recovered_raw_bytes(tmp_path):
    rows = json.loads(FIXTURE.read_text())["rows"]
    result = write_reconstruction(iter(rows), tmp_path, split="synthetic")
    assert result["rows"] == len(rows)
    assert result["articles"] == 3
    assert result["raw_utf8_bytes"] == sum(len(r.encode()) for r in rows)
    assert result["reconstruction_audit_sha256"] == result["source_row_audit_sha256"]
    assert result["derived_content_bytes"] == sum(len(r.encode()) + (not r.endswith("\n")) for r in rows)
    assert result["row_index_bytes"] == 17 * len(rows)
    assert result["purpose"] == "non_scoring_provenance_audit"
    assert not any("raw_bytes" in line for line in (tmp_path / "synthetic.articles.jsonl").read_text().splitlines())


@pytest.mark.parametrize("args", [
    ("other/wikitext", "wikitext-103-raw-v1", REVISION),
    ("Salesforce/wikitext", "wikitext-2-raw-v1", REVISION),
    ("Salesforce/wikitext", "wikitext-103-raw-v1", "main"),
])
def test_fetch_rejects_unpinned_identity_before_any_io(tmp_path, args):
    with pytest.raises(ValueError, match="pinned"):
        fetch_source(*args, tmp_path)
    assert not list(tmp_path.iterdir())


def test_fetch_rejects_corrupt_cache_without_network(tmp_path):
    (tmp_path / "test-00000-of-00001.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash|size"):
        fetch_source("Salesforce/wikitext", "wikitext-103-raw-v1", REVISION, tmp_path)
    assert not (tmp_path / "source-manifest.json").exists()


def test_parquet_order_and_schema_validation(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    paths = [tmp_path / f"shard{i}.parquet" for i in range(2)]
    pq.write_table(pa.table({"text": [" = A = ", "first"]}), paths[0])
    pq.write_table(pa.table({"text": ["second", " = B = "]}), paths[1])
    rows = iter_source_rows(paths, expected_rows=4)
    assert [a.content for a in reconstruct_articles(rows)] == [b" = A = \nfirst\nsecond\n", b" = B = \n"]
    with pytest.raises(ValueError, match="row count"):
        list(iter_source_rows(paths, expected_rows=3))
    pq.write_table(pa.table({"other": ["bad"]}), paths[0])
    with pytest.raises(ValueError, match="schema"):
        list(iter_source_rows(paths))
    pq.write_table(pa.table({"text": [None]}, schema=pa.schema([("text", pa.string())])), paths[0])
    with pytest.raises(ValueError, match="string"):
        list(iter_source_rows(paths))


def test_synthetic_pipeline_reports_missing_stages(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"kind": "synthetic", "rows_path": str(FIXTURE), "output": str(tmp_path / "output")}))
    ref = prepare_data(config)
    result = json.loads(ref.path.read_text())
    assert ref.sha256 == sha256_file(ref.path)
    assert result["status"] == "incomplete"
    assert result["missing_stages"] == ["H02 deduplication and partitions", "H03 byte batches", "H04-H08 remaining harness"]
    assert result["scientific_gate"] is None
    assert result["splits"]["synthetic"]["rows"] == 9


def test_cli_reports_incomplete_source_only_pipeline(tmp_path, capsys):
    from ralup.__main__ import main
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"kind": "synthetic", "rows_path": str(FIXTURE), "output": str(tmp_path / "output")}))
    assert main(["prepare-data", "--config", str(config)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "incomplete"
    assert report["missing_stages"][0].startswith("H03")


def test_same_size_corrupt_asset_is_rejected(tmp_path):
    (tmp_path / "test-00000-of-00001.parquet").write_bytes(b"x" * 732610)
    with pytest.raises(ValueError, match="hash"):
        fetch_source("Salesforce/wikitext", "wikitext-103-raw-v1", REVISION, tmp_path)


@pytest.mark.parametrize("change", [{"kind": "production"}, {"extra": "field"}, {"output": ""}])
def test_invalid_source_config_rejected(tmp_path, change):
    settings = {"kind": "synthetic", "rows_path": str(FIXTURE), "output": str(tmp_path / "output")}
    settings.update(change)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="config"):
        prepare_data(config)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("header", ["x-repo-commit", "x-linked-etag", "x-linked-size"])
def test_download_rejects_wrong_remote_identity(tmp_path, monkeypatch, header):
    import urllib.error
    from ralup.data import source
    headers = {"x-repo-commit": REVISION, "x-linked-etag": '"' + "0" * 64 + '"', "x-linked-size": "3"}
    headers[header] = "wrong"
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 302, "redirect", headers, None)
    monkeypatch.setattr(source.urllib.request, "build_opener", lambda *_: Opener())
    with pytest.raises(ValueError, match="revision/asset"):
        source._download({"url": "https://example.invalid/pinned", "sha256": "0" * 64, "bytes": 3}, tmp_path / "asset.parquet", REVISION)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("body", [b"abc", b"bad"])
def test_download_authenticates_bytes_and_preserves_unrelated_partial(tmp_path, monkeypatch, body):
    import io
    import urllib.error
    from ralup.data import source
    expected = hashlib.sha256(b"abc").hexdigest()
    headers = {"x-repo-commit": REVISION, "x-linked-etag": '"' + expected + '"', "x-linked-size": "3"}
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 302, "redirect", headers, None)
    monkeypatch.setattr(source.urllib.request, "build_opener", lambda *_: Opener())
    monkeypatch.setattr(source.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(body))
    path = tmp_path / "asset.parquet"
    partial = tmp_path / "asset.parquet.part"
    partial.write_bytes(b"unrelated")
    asset = {"name": path.name, "url": "https://example.invalid/pinned", "sha256": expected, "bytes": 3}
    if body == b"abc":
        source._download(asset, path, REVISION)
        assert path.read_bytes() == b"abc"
    else:
        with pytest.raises(ValueError, match="hash"):
            source._download(asset, path, REVISION)
        assert not path.exists()
    assert partial.read_bytes() == b"unrelated"
    assert list(tmp_path.glob("*.part")) == [partial]


def test_source_audit_is_computed_before_article_transformation(tmp_path, monkeypatch):
    import unicodedata
    from ralup.data import articles
    original = articles.iter_articles
    monkeypatch.setattr(articles, "iter_articles", lambda rows: original(unicodedata.normalize("NFC", row) for row in rows))
    with pytest.raises(ValueError, match="audit mismatch"):
        articles.write_reconstruction(["é"], tmp_path, split="synthetic")
