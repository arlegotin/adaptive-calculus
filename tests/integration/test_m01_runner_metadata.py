"""GPU-free preflight catches actual RunLedger boundaries before a proxy launch."""
import importlib.util
import json
from pathlib import Path

from ralup.config import ResolvedConfig,canonical_bytes
from ralup.records import _validate_metadata


def test_runner_metadata_satisfies_real_runledger_contract():
    spec=importlib.util.spec_from_file_location('m01_proxy',Path('docs/harness/m01-proxy.py'))
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    assert hasattr(runner,'build_metadata'), 'Explicit preflight metadata constructor missing'
    frozen=json.loads(Path('docs/harness/m01-design-v2.json').read_bytes())
    runner.DESIGN=Path('docs/harness/m01-design-v2.json')
    resolved=ResolvedConfig(frozen['resolved_config']);cfg=json.loads(canonical_bytes(resolved.payload))
    metadata=runner.build_metadata(frozen,resolved,cfg,'muon_project_fp32','preflight',32768)
    _validate_metadata(metadata)
    assert set(metadata['source']) == {'commit','dirty_diff_sha256'}
    assert metadata['producer'] == frozen['source']
    assert metadata['endpoint']['purpose'] == 'pilot-bootstrap'
