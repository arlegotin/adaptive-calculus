"""Authenticate immutable non-executable assets; prepare provenance only."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

from ..config import canonical_bytes
from ..records import ArtifactRef, sha256_file
from .articles import write_reconstruction

ROOT = Path(__file__).resolve().parents[3]
PIN_MANIFEST = ROOT / "vendor/source/wikitext/manifest.json"
PIN_MANIFEST_SHA256 = "fefdc80bde11a22eaeeefe3ae1c4d4a71be9871759b12323ade9b93d202a1b1f"
IDENTITY = ("Salesforce/wikitext", "wikitext-103-raw-v1", "b08601e04326c79dfdd32d625aee71d232d685c3")
MISSING_STAGES = ["H02 deduplication and partitions", "H03 byte batches", "H04-H08 remaining harness"]


def _pins() -> dict:
    if sha256_file(PIN_MANIFEST) != PIN_MANIFEST_SHA256:
        raise ValueError("Pinned provenance manifest hash mismatch")
    pins = json.loads(PIN_MANIFEST.read_bytes())
    for notice in [pins["card"], *pins["legal_notices"]["records"], *pins["provenance_files"]]:
        path = ROOT / notice["path"]
        if sha256_file(path) != notice["sha256"] or ("bytes" in notice and path.stat().st_size != notice["bytes"]):
            raise ValueError("Pinned source/legal notice hash or size mismatch")
    return pins


def _verify(path: Path, asset: dict):
    if path.stat().st_size != asset["bytes"] or sha256_file(path) != asset["sha256"]:
        raise ValueError(f"Source asset hash or size mismatch: {asset['name']}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _download(asset: dict, path: Path, revision: str):
    # Authenticate repository revision before following signed transport redirects.
    request = urllib.request.Request(asset["url"], method="HEAD")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=60) as response:
            headers = dict(response.headers)
    except urllib.error.HTTPError as response:
        if response.code not in (301, 302, 303, 307, 308):
            raise
        headers = dict(response.headers)
        response.close()
    lowered = {k.lower(): v for k, v in headers.items()}
    if (lowered.get("x-repo-commit") != revision
            or lowered.get("x-linked-etag", "").strip('"') != asset["sha256"]
            or lowered.get("x-linked-size") != str(asset["bytes"])):
        raise ValueError("Remote pinned revision/asset metadata mismatch")
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        count = 0
        with os.fdopen(descriptor, "wb") as stream, urllib.request.urlopen(asset["url"], timeout=60) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
                digest.update(chunk)
                count += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if count != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
            raise ValueError("Downloaded asset hash or size mismatch")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return lowered


def fetch_source(repository: str, configuration: str, revision: str, destination: Path) -> ArtifactRef:
    """Authenticate all pinned cache bytes, downloading only absent assets.

    An invalid existing cache is rejected, never silently overwritten. Original
    verified download/HEAD evidence remains in the permanent provenance bundle.
    """
    if (repository, configuration, revision) != IDENTITY:
        raise ValueError("Source identity must match the pinned repository/configuration/revision")
    pins = _pins()
    destination = Path(destination)
    output = destination / "source-manifest.json"
    if output.exists() or output.is_symlink():
        # Reuse the authenticated observation itself, not a newly timestamped
        # report about the same assets. An incomplete cache cannot repair it.
        try:
            report = json.loads(output.read_bytes())
            if ((report['repository'], report['configuration'], report['revision']) != IDENTITY
                    or report['provenance_manifest_sha256'] != PIN_MANIFEST_SHA256
                    or len(report['assets']) != len(pins['assets'])):
                raise ValueError('Source observation manifest identity mismatch')
            for observed, asset in zip(report['assets'], pins['assets']):
                path = destination / asset['name']
                _verify(path, asset)
                if (any(observed.get(key) != value for key, value in asset.items())
                        or Path(observed['local_path']).resolve() != path.resolve()
                        or observed['observed_bytes'] != asset['bytes']
                        or observed['observed_sha256'] != asset['sha256']
                        or observed['status'] not in ('authenticated_cache_reuse', 'verified_download')):
                    raise ValueError('Source observation manifest asset mismatch')
            if (report['observed_total_bytes'] != sum(a['bytes'] for a in pins['assets'])
                    or report['new_download_bytes'] != sum(a['observed_bytes'] for a in report['assets']
                                                           if a['status'] == 'verified_download')
                    or any(report[key] != pins[key] for key in
                           ('intended_use', 'protected_test_policy', 'license_discrepancy'))):
                raise ValueError('Source observation manifest accounting mismatch')
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Existing source observation manifest cannot be authenticated: {error}') from error
        return ArtifactRef(output, sha256_file(output))
    for asset in pins["assets"]:
        path = destination / asset["name"]
        if path.exists():
            _verify(path, asset)
    destination.mkdir(parents=True, exist_ok=True)
    observed = []
    for asset in pins["assets"]:
        path = destination / asset["name"]
        reused = path.exists()
        head = asset["pinned_head"] if reused else _download(asset, path, revision)
        observed.append({**asset, "local_path": str(path), "observed_bytes": path.stat().st_size,
                         "observed_sha256": sha256_file(path),
                         "status": "authenticated_cache_reuse" if reused else "verified_download",
                         "revision_authentication": "retained_pinned_head_and_asset_hash" if reused else "live_pinned_head_and_asset_hash",
                         "authenticated_head": head})
    report = {"repository": repository, "configuration": configuration, "revision": revision,
              "provenance_manifest_sha256": PIN_MANIFEST_SHA256,
              "observed_utc": datetime.now(timezone.utc).isoformat(), "assets": observed,
              "observed_total_bytes": sum(a["observed_bytes"] for a in observed),
              "new_download_bytes": sum(a["observed_bytes"] for a in observed if a["status"] == "verified_download"),
              "intended_use": pins["intended_use"], "protected_test_policy": pins["protected_test_policy"],
              "license_discrepancy": pins["license_discrepancy"]}
    with output.open('xb') as stream:
        stream.write(canonical_bytes(report) + b"\n")
    return ArtifactRef(output, sha256_file(output))


def iter_source_rows(paths, *, expected_rows: int | None = None):
    """Iterate caller-specified deterministic shard order without shard resets."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    count = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        if schema.names != ["text"] or schema.field("text").type != pa.string():
            raise ValueError(f"Unexpected source schema: {path}")
        for batch in parquet.iter_batches(batch_size=4096, columns=["text"], use_threads=False):
            for text in batch.column(0).to_pylist():
                if not isinstance(text, str):
                    raise ValueError("Source text must be a non-null string")
                count += 1
                yield text
    if expected_rows is not None and count != expected_rows:
        raise ValueError(f"Source row count mismatch: expected {expected_rows}, observed {count}")


def prepare_data(config: Path) -> ArtifactRef:
    """H01 source-only preparation; this is not a resolved training config."""
    config = Path(config)
    settings = json.loads(config.read_bytes())
    if not isinstance(settings, dict):
        raise ValueError("Source preparation config must be an object")
    kind = settings.get("kind")
    required = ({"kind", "rows_path", "output"} if kind == "synthetic" else
                {"kind", "repository", "configuration", "revision", "cache", "output"})
    if kind not in ("synthetic", "pinned_source") or set(settings) != required:
        raise ValueError("Invalid source-only preparation config keys/kind")
    for key in required - {"kind"}:
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"Source config {key} must be a nonempty string")
    output = Path(settings["output"])
    split_names = ('synthetic',) if kind == 'synthetic' else ('train', 'validation', 'test')
    generation = [output/'preparation.json', *(output/f'{split}.{suffix}'
        for split in split_names for suffix in ('content.bin', 'rows.bin', 'articles.jsonl'))]
    if any(path.exists() or path.is_symlink() for path in generation):
        raise ValueError('Refusing existing preparation generation; use authenticated reuse or a fresh destination')
    source = None
    splits = {}
    if kind == "synthetic":
        fixture = Path(settings["rows_path"])
        rows = json.loads(fixture.read_bytes())
        if not isinstance(rows, dict) or set(rows) != {"rows"} or not isinstance(rows["rows"], list):
            raise ValueError("Synthetic fixture requires rows list")
        source = {"kind": "synthetic", "path": str(fixture), "sha256": sha256_file(fixture)}
        splits["synthetic"] = write_reconstruction(rows["rows"], output, split="synthetic")
    else:
        ref = fetch_source(settings["repository"], settings["configuration"], settings["revision"], Path(settings["cache"]))
        pins = _pins()
        source = {"kind": kind, "path": str(ref.path), "sha256": ref.sha256}
        for split in ("train", "validation", "test"):
            assets = sorted((a for a in pins["assets"] if a["split"] == split), key=lambda a: a["source_path"])
            paths = [Path(settings["cache"]) / a["name"] for a in assets]
            splits[split] = write_reconstruction(iter_source_rows(paths, expected_rows=pins["split_rows"][split]), output, split=split)
            splits[split]["source_shards"] = [a["source_path"] for a in assets]
            splits[split]["access_policy"] = "protected_non_scoring_audit_only_until_E03" if split == "test" else "non_scoring_provenance_audit"
    report = {"schema_version": 1, "kind": kind, "status": "incomplete", "scientific_gate": None,
              "completed_stages": ["H01 authenticated source and reversible reconstruction"],
              "missing_stages": MISSING_STAGES, "source": source, "splits": splits,
              "config_sha256": sha256_file(config), "config": settings,
              "derived_total_bytes": sum(s["derived_total_bytes"] for s in splits.values())}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preparation.json"
    with path.open('xb') as stream:
        stream.write(canonical_bytes(report) + b"\n")
    return ArtifactRef(path, sha256_file(path))
