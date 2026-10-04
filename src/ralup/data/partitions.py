"""Authenticate H01 references, run S09 audit, and freeze raw-byte partitions."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ..config import ResolvedConfig, canonical_bytes, resolve_config
from ..records import sha256_file
from .articles import ROW_RECORD
from .dedup import CONTRACT, audit_indexed
from .source import IDENTITY, ROOT, PIN_MANIFEST_SHA256, _pins, _verify, prepare_data

PARTITIONS = ('meta-train', 'meta-development', 'target-training')


@dataclass(frozen=True)
class CorpusManifest:
    path: Path
    sha256: str
    payload: dict

    @property
    def partition_counts(self):
        return {k: v['documents'] for k, v in self.payload['partitions'].items()}

    @property
    def partition_bytes(self):
        return {k: v['bytes'] for k, v in self.payload['partitions'].items()}


def _partition_hash64(content):
    return int.from_bytes(hashlib.sha256(b'ralup-split-v1' + content).digest()[:8], 'big')


def partition_article(content: bytes) -> str:
    value = _partition_hash64(content)
    if 5 * value < 2**64:
        return PARTITIONS[0]
    if 4 * value < 2**64:
        return PARTITIONS[1]
    return PARTITIONS[2]


def _authenticate(path, expected_hash, expected_bytes=None):
    path = Path(path)
    if ((expected_bytes is not None and path.stat().st_size != expected_bytes)
            or sha256_file(path) != expected_hash):
        raise ValueError(f'Artifact hash or size mismatch: {path}')


def _source_settings(config):
    settings = json.loads(Path(config).read_bytes())
    if not isinstance(settings, dict):
        raise ValueError('Source-only config must be an object')
    kind = settings.get('kind')
    keys = {'kind', 'rows_path', 'output'} if kind == 'synthetic' else {'kind', 'repository', 'configuration', 'revision', 'cache', 'output'}
    if (kind not in ('synthetic', 'pinned_source') or set(settings) != keys
            or any(not isinstance(v, str) or not v.strip() for v in settings.values())):
        raise ValueError('Invalid source-only config')
    if kind == 'pinned_source' and tuple(settings[k] for k in ('repository', 'configuration', 'revision')) != IDENTITY:
        raise ValueError('Source identity differs from pinned source')
    return settings


def _load_preparation(path, settings, config_hash):
    if settings['kind'] == 'pinned_source':
        accepted = json.loads((ROOT / 'docs/data/h01-evidence.json').read_bytes())['prepared_manifest']
        _authenticate(path, accepted['sha256'], accepted['bytes'])
    report = json.loads(path.read_bytes())
    if report['config'] != settings or report['config_sha256'] != config_hash or report['kind'] != settings['kind']:
        raise ValueError('Preparation/source config mismatch')
    source = report['source']
    _authenticate(source['path'], source['sha256'])
    if settings['kind'] == 'pinned_source':
        pins = _pins()
        source_report = json.loads(Path(source['path']).read_bytes())
        if (tuple(source_report[k] for k in ('repository', 'configuration', 'revision')) != IDENTITY
                or source_report['provenance_manifest_sha256'] != PIN_MANIFEST_SHA256):
            raise ValueError('Source manifest pinned identity mismatch')
        if set(source_report['assets'][i]['name'] for i in range(len(source_report['assets']))) != {a['name'] for a in pins['assets']}:
            raise ValueError('Source asset inventory mismatch')
        for asset in pins['assets']:
            _verify(Path(settings['cache']) / asset['name'], asset)
        if set(report['splits']) != {'train', 'validation', 'test'}:
            raise ValueError('Official split inventory mismatch')
    elif set(report['splits']) != {'synthetic'}:
        raise ValueError('Synthetic split inventory mismatch')
    # ALL derived hashes authenticate before opening any index or content slice.
    for split, item in report['splits'].items():
        if {f['path'] for f in item['files']} != {f'{split}.{suffix}' for suffix in ('content.bin', 'rows.bin', 'articles.jsonl')}:
            raise ValueError('Derived file inventory mismatch')
        if item['source_row_audit_sha256'] != item['reconstruction_audit_sha256']:
            raise ValueError('Reversible source audit mismatch')
        for file in item['files']:
            _authenticate(path.parent / file['path'], file['sha256'], file['bytes'])
    return report


def _indexed_documents(path, report):
    documents = []
    for split in ('train', 'synthetic', 'validation', 'test'):
        if split not in report['splits']:
            continue
        item = report['splits'][split]
        row_file = path.parent / f'{split}.rows.bin'
        content_file = path.parent / f'{split}.content.bin'
        row_stop = content_stop = count = 0
        for line in (path.parent / f'{split}.articles.jsonl').read_bytes().splitlines():
            article = json.loads(line)
            if (article['article_index'] != count or article['source_row_start'] != row_stop
                    or article['content_start'] != content_stop
                    or article['source_row_stop'] <= row_stop or article['content_end'] <= content_stop):
                raise ValueError('Article ordinal/range continuity mismatch')
            documents.append({**article, 'id': f'{split}:{count}', 'split': split,
                              'bytes': article['content_end']-article['content_start'],
                              'protected': split in ('validation', 'test'),
                              'content_path': str(content_file),
                              'row_index': {'path': str(row_file), 'format': '<QQB',
                                            'byte_start': article['source_row_start']*ROW_RECORD.size,
                                            'byte_stop': article['source_row_stop']*ROW_RECORD.size},
                              'source_shards': item.get('source_shards', [])})
            row_stop, content_stop = article['source_row_stop'], article['content_end']
            count += 1
        if (count != item['articles'] or row_stop != item['rows']
                or content_stop != item['derived_content_bytes'] or row_file.stat().st_size != row_stop*ROW_RECORD.size):
            raise ValueError('Article/source counts mismatch')
    return documents


def _read_content(document):
    with Path(document['content_path']).open('rb') as stream:
        stream.seek(document['content_start'])
        content = stream.read(document['bytes'])
    if len(content) != document['bytes'] or hashlib.sha256(content).hexdigest() != document['content_sha256']:
        raise ValueError('Indexed article content hash mismatch')
    return content


def _prepare(config_path, settings, resolved_hash=None):
    path = Path(settings['output']) / 'preparation.json'
    if not path.exists():
        if settings['kind'] == 'pinned_source':
            raise ValueError('Missing accepted H01 preparation; authenticate and accept H01 provenance before H02')
        path = prepare_data(config_path).path
    config_hash = sha256_file(config_path)
    report = _load_preparation(path, settings, config_hash)
    preparation_hash = sha256_file(path)
    documents = _indexed_documents(path, report)
    audit = audit_indexed(documents, _read_content, CONTRACT)
    partitions = {name: {'documents': 0, 'bytes': 0, 'document_ids': [], 'content_hashes': []} for name in PARTITIONS}
    for document in audit['retained_documents']:
        partition = partition_article(_read_content(document))
        document['partition'] = partition
        item = partitions[partition]
        item['documents'] += 1
        item['bytes'] += document['bytes']
        item['document_ids'].append(document['id'])
        item['content_hashes'].append(document['content_sha256'])
    payload = {'schema_version': 1, 'status': 'incomplete', 'scientific_gate': None,
               'completed_stages': report['completed_stages'] + ['H02 duplicate audit and raw-content partitions'],
               'missing_stages': ['H03 byte batches', 'H04-H08 remaining harness'],
               'synthetic': settings['kind'] == 'synthetic',
               'authorization': 'resolved_prepare-data' if resolved_hash else 'source-only_bootstrap_non_scoring',
               'resolved_config_sha256': resolved_hash, 'source_config_sha256': config_hash,
               'preparation': {'path': str(path), 'sha256': preparation_hash}, 'source': report['source'],
               'revision': settings.get('revision'), 'source_files': {k: v['files'] for k, v in report['splits'].items()},
               'partition_algorithm': 'first uint64_be(SHA256(b"ralup-split-v1" || raw_content)); 5*x<2**64, else 4*x<2**64, else target-training',
               'partitions': partitions, 'audit': audit,
               'protected_access': 'non-scoring duplicate audit only; no protected targets or model scoring',
               'license_discrepancy': json.loads(Path(report['source']['path']).read_bytes()).get('license_discrepancy') if settings['kind'] == 'pinned_source' else None}
    # Re-authenticate references before atomic publication to reject stale artifacts.
    if sha256_file(path) != preparation_hash:
        raise ValueError('Preparation changed during audit')
    _load_preparation(path, settings, config_hash)
    destination = path.parent.parent / 'corpus'
    destination.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(payload) + b'\n'
    digest = hashlib.sha256(encoded).hexdigest()
    manifest = destination / f'manifest-{digest}.json'
    fd, name = tempfile.mkstemp(prefix='manifest-', suffix='.part', dir=destination)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(manifest)
    finally:
        Path(name).unlink(missing_ok=True)
    return CorpusManifest(manifest, digest, payload)


def prepare_bootstrap(config_path: Path) -> CorpusManifest:
    """Narrow source-only CLI path; no WP00/full-research authorization implied."""
    return _prepare(Path(config_path), _source_settings(config_path))


def prepare_corpus(config: ResolvedConfig) -> CorpusManifest:
    """Research entry: F01 prepare-data resolution; no future WP02 prerequisite.

    H01 locator is configs/data.json at package project root, whose exact pinned
    revision selects data/processed/wikitext/<revision>/provenance/preparation.json.
    """
    if not isinstance(config, ResolvedConfig):
        raise TypeError('prepare_corpus requires ResolvedConfig')
    payload = config.payload
    if payload.get('purpose') != 'prepare-data':
        raise ValueError('prepare_corpus requires prepare-data purpose')
    # Revalidate the F01 snapshot so directly constructed/forged dataclasses
    # cannot bypass normative schema, value, or prepare-data gate checks.
    plain = json.loads(canonical_bytes(payload))
    gates = plain.pop('evidence_gates')
    plain.pop('purpose')
    checked = resolve_config(plain, {'gates': gates}, purpose='prepare-data')
    if checked.sha256 != config.sha256:
        raise ValueError('Resolved prepare-data snapshot mismatch')
    if payload.get('synthetic') is not False:
        raise ValueError('Research preparation requires non-synthetic pinned source')
    data = payload['data']
    if tuple(data[k] for k in ('source', 'subset', 'revision')) != IDENTITY:
        raise ValueError('Resolved source identity differs from pinned source')
    if dict(data['near_duplicate']) != {k: tuple(v) if isinstance(v, list) else v for k, v in CONTRACT.items()}:
        raise ValueError('Resolved duplicate contract mismatch')
    gate = payload['evidence_gates']['WP00']
    if gate['status'] != 'pass' or not gate['evidence'] or gate['blockers']:
        raise ValueError('Resolved prepare-data requires passing WP00 evidence')
    for ref in [payload['runtime']['dependency_lock'], payload['runtime']['environment'], *gate['evidence']]:
        _authenticate(ref['path'], ref['sha256'])
    source_config = ROOT / 'configs/data.json'
    settings = _source_settings(source_config)
    # Bind the locator to the documented project-root convention.
    if Path.cwd() != ROOT:
        raise ValueError('Pinned preparation convention requires project-root working directory')
    return _prepare(source_config, settings, config.sha256)
