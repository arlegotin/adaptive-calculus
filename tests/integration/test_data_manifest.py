import hashlib
import json
from pathlib import Path

import pytest

from ralup.config import ResolvedConfig
from ralup.data.source import prepare_data
from ralup.records import sha256_file


def preparation(tmp_path):
    rows = tmp_path / 'rows.json'
    rows.write_text(json.dumps({'rows': ['preamble', '= A =', 'Ａ é', '= B =', 'independent bytes']}))
    config = tmp_path / 'source.json'
    config.write_text(json.dumps({'kind': 'synthetic', 'rows_path': str(rows), 'output': str(tmp_path / 'provenance')}))
    return config, prepare_data(config)


def test_manifest_reproducible_with_reversible_source_references(tmp_path):
    from ralup.data.partitions import prepare_bootstrap
    config, ref = preparation(tmp_path)
    result = prepare_bootstrap(config)
    second = prepare_bootstrap(config)
    assert result.sha256 == second.sha256
    assert result.payload['partitions'] == second.payload['partitions']
    assert result.payload['preparation']['sha256'] == ref.sha256
    assert result.payload['status'] == 'incomplete'
    assert result.payload['missing_stages'][0] == 'H03 byte batches'
    assert result.payload['synthetic'] is True
    assert sum(p['documents'] for p in result.payload['partitions'].values()) == 3
    assert sum(p['bytes'] for p in result.payload['partitions'].values()) == sum(s['derived_content_bytes'] for s in json.loads(ref.path.read_bytes())['splits'].values())
    for doc in result.payload['audit']['retained_documents']:
        assert doc['split'] == 'synthetic'
        assert doc['source_row_start'] < doc['source_row_stop']
        assert doc['row_index']['format'] == '<QQB'
        assert 'raw_strings' not in doc
    assert result.payload['audit']['protected_documents'] == []


@pytest.mark.parametrize('suffix', ['content.bin', 'rows.bin', 'articles.jsonl'])
def test_stale_derived_files_rejected_before_index_reading(tmp_path, suffix):
    from ralup.data.partitions import prepare_bootstrap
    config, ref = preparation(tmp_path)
    path = ref.path.parent / ('synthetic.' + suffix)
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match='hash|size'):
        prepare_bootstrap(config)
    assert not list((ref.path.parent.parent / 'corpus').glob('*.json'))


def test_source_reference_corruption_rejected(tmp_path):
    from ralup.data.partitions import prepare_bootstrap
    config, ref = preparation(tmp_path)
    source = json.loads(ref.path.read_bytes())['source']['path']
    Path(source).write_text('corrupt')
    with pytest.raises(ValueError, match='hash|size'):
        prepare_bootstrap(config)


def test_full_research_interface_requires_resolved_preparation_purpose():
    from ralup.data.partitions import prepare_corpus
    with pytest.raises(TypeError, match='ResolvedConfig'):
        prepare_corpus({})
    with pytest.raises(ValueError, match='prepare-data'):
        prepare_corpus(ResolvedConfig({'purpose': 'train'}))


def test_pinned_preparation_rejects_unaccepted_manifest_before_source_access(tmp_path):
    from ralup.data.partitions import ROOT, _load_preparation, _source_settings
    settings = _source_settings(ROOT / 'configs/data.json')
    report = {'kind': 'pinned_source', 'splits': {'train': {'articles': 29445}}}
    changed = tmp_path / 'preparation.json'
    changed.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='hash|size'):
        _load_preparation(changed, settings, sha256_file(ROOT / 'configs/data.json'))


def test_genuine_f01_resolution_requires_authenticated_runtime_inputs(tmp_path):
    from ralup.config import resolve_config
    from ralup.data.partitions import ROOT, prepare_corpus
    template = json.loads((ROOT / 'configs/proxy.json').read_bytes())
    evidence_file = tmp_path / 'environment.json'
    evidence_file.write_text('{}')
    ref = {'path': str(evidence_file), 'sha256': sha256_file(evidence_file)}
    evidence = {'data': {'revision': 'b08601e04326c79dfdd32d625aee71d232d685c3'},
                'runtime': {'environment': ref, 'dependency_lock': ref},
                'gates': {'WP00': {'status': 'pass', 'evidence': [ref], 'blockers': []}}}
    resolved = resolve_config(template, evidence, purpose='prepare-data')
    assert resolved.payload['data']['manifest'] is None  # Future WP02 is not a prerequisite.
    evidence_file.write_text('corrupt')
    with pytest.raises(ValueError, match='hash|size'):
        prepare_corpus(resolved)


def test_forged_resolved_payload_cannot_bypass_fixed_research_contract(tmp_path, monkeypatch):
    from ralup.config import resolve_config, canonical_bytes
    from ralup.data.partitions import ROOT, prepare_corpus
    template = json.loads((ROOT / 'configs/proxy.json').read_bytes())
    evidence_file = tmp_path / 'environment.json'
    evidence_file.write_text('{}')
    ref = {'path': str(evidence_file), 'sha256': sha256_file(evidence_file)}
    resolved = resolve_config(template, {'data': {'revision': 'b08601e04326c79dfdd32d625aee71d232d685c3'},
                                      'runtime': {'environment': ref, 'dependency_lock': ref},
                                      'gates': {'WP00': {'status': 'pass', 'evidence': [ref], 'blockers': []}}}, purpose='prepare-data')
    payload = json.loads(canonical_bytes(resolved.payload))
    payload['data']['split_fractions'] = [0.1, 0.1, 0.8]
    # Isolate slow corpus work; the real F01 contract must reject beforehand.
    monkeypatch.setattr('ralup.data.partitions._prepare', lambda *args: None)
    with pytest.raises(ValueError, match='normative'):
        prepare_corpus(ResolvedConfig(payload))


def test_missing_pinned_h01_preparation_never_reconstructs_or_downloads(tmp_path, monkeypatch):
    from ralup.data import partitions
    settings = partitions._source_settings(partitions.ROOT / 'configs/data.json')
    settings['output'] = str(tmp_path / 'absent')
    # Download/reconstruction is a separate H01 action; stop if it is reached.
    def unexpected(*args):
        pytest.fail('H02 must not rerun pinned H01 reconstruction')
    monkeypatch.setattr(partitions, 'prepare_data', unexpected)
    with pytest.raises(ValueError, match='accepted H01'):
        partitions._prepare(partitions.ROOT / 'configs/data.json', settings)


def test_failed_audit_preserves_previously_published_manifest(tmp_path, monkeypatch):
    from ralup.data import partitions
    config, ref = preparation(tmp_path)
    previous = partitions.prepare_bootstrap(config)
    original_bytes = previous.path.read_bytes()
    def interrupted(*args):
        raise OSError('injected audit interruption')
    monkeypatch.setattr(partitions, 'audit_indexed', interrupted)
    with pytest.raises(OSError, match='interruption'):
        partitions.prepare_bootstrap(config)
    assert previous.path.read_bytes() == original_bytes
    assert sha256_file(previous.path) == previous.sha256
    assert not list(previous.path.parent.glob('*.part'))


def test_source_only_cli_rejects_nonobject_config_cleanly(tmp_path, capsys):
    from ralup.__main__ import main
    path = tmp_path / 'list-config.json'
    path.write_text('[]')
    with pytest.raises(SystemExit) as failure:
        main(['prepare-data', '--config', str(path)])
    assert failure.value.code == 2
    assert 'config must be an object' in capsys.readouterr().err


def test_genuine_resolved_entry_accepts_authenticated_prepare_data_gate(tmp_path, monkeypatch):
    from ralup.config import canonical_bytes, resolve_config
    from ralup.data import partitions
    source_config, _ = preparation(tmp_path)
    fixture_manifest = partitions.prepare_bootstrap(source_config)
    template = json.loads((partitions.ROOT / 'configs/proxy.json').read_bytes())
    environment = tmp_path / 'environment.json'
    environment.write_text('{}')
    ref = {'path': str(environment), 'sha256': sha256_file(environment)}
    resolved = resolve_config(template, {'data': {'revision': 'b08601e04326c79dfdd32d625aee71d232d685c3'},
                                        'runtime': {'environment': ref, 'dependency_lock': ref},
                                        'gates': {'WP00': {'status': 'pass', 'evidence': [ref], 'blockers': []}}}, purpose='prepare-data')
    # The expensive pinned corpus stage is substituted with a real synthetic
    # preparation. F01 schema, gates and runtime authentication remain real.
    monkeypatch.setattr(partitions, '_prepare', lambda *args: fixture_manifest)
    result = partitions.prepare_corpus(resolved)
    assert result.sha256 == sha256_file(result.path)
    assert result.partition_counts == fixture_manifest.partition_counts
    assert result.payload['synthetic'] is True  # Never report this as real corpus evidence.
    assert resolved.payload['training']['microbatch'] is None
