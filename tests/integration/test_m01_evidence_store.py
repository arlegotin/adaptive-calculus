"""Archive proof is byte-authenticated and reconstructs all scientific owners."""
import importlib.util
from pathlib import Path

import pytest
import torch


def test_complete_snapshot_archive_roundtrip_and_tamper_detection(tmp_path):
    path=Path('docs/harness/m01-evidence-store.py')
    assert path.exists(), 'M01 durable snapshot evidence store missing'
    spec=importlib.util.spec_from_file_location('m01_store',path);store=importlib.util.module_from_spec(spec);spec.loader.exec_module(store)
    from test_matrix_optimizer_parity import matrix_fixture
    from test_resume import advance
    from ralup.state import capture_snapshot,_validated,restore_snapshot,exact_cpu_state_hash
    state=matrix_fixture();advance(state,2)
    state.configuration['large_string']='test'*5000
    snapshot=capture_snapshot(state)
    from ralup.state import save_snapshot,load_snapshot,_manifest
    original=tmp_path/'original-h07'
    original_ref=save_snapshot(snapshot,original)
    snapshot=load_snapshot(original_ref)
    ref=store.archive_snapshot(snapshot,tmp_path/'objects')
    original.rename(tmp_path/'isolated-original-not-at-original-path')
    recovered=store.recover_snapshot(ref)
    original_manifest,original_container=_manifest(snapshot)
    actual_manifest,actual_container=_manifest(recovered)
    assert actual_manifest == original_manifest and actual_container == original_container
    reconstructed=save_snapshot(recovered,tmp_path/'materialized-from-archive')
    loaded=load_snapshot(reconstructed)
    assert _manifest(loaded) == _manifest(snapshot)
    assert reconstructed.sha256 == original_ref.sha256
    assert recovered.metadata == snapshot.metadata
    assert recovered.producer_files == snapshot.producer_files
    assert recovered.tensors.keys() == snapshot.tensors.keys()
    assert all(torch.equal(v,recovered.tensors[k]) for k,v in snapshot.tensors.items())
    assert exact_cpu_state_hash(restore_snapshot(recovered,'cpu')) == exact_cpu_state_hash(state)
    payload=_validated(recovered)
    assert payload['configuration']['large_string'] == 'test'*5000
    existing={p.name for p in (tmp_path/'objects').iterdir()}
    assert store.archive_snapshot(snapshot,tmp_path/'objects') == ref
    assert {p.name for p in (tmp_path/'objects').iterdir()} == existing
    ref.path.write_bytes(b'broken')
    with pytest.raises(ValueError,match='checksum'): store.recover_snapshot(ref)
