"""Actual laptop operation gate; unavailable devices retain blocked artifacts."""
import json
import pytest


@pytest.mark.native
def test_mps_operations():
    from ralup.environment import probe_operations
    result = probe_operations("mps")
    report = json.loads(result.evidence[0].path.read_text())
    if result.status == "blocked":
        pytest.skip(f"blocked report: {result.evidence[0].path}: {result.blockers}")
    assert result.status == "pass", result.blockers
    assert {"matmul_backward", "expm1", "rng_restore", "layernorm", "gelu", "attention",
            "reductions", "clipping", "comparison_indexing", "hadamard", "ste_backward",
            "trust_backward", "state_updates", "complete_forward_backward",
            "bucketize", "qr", "eigh", "functional_grad"} <= report["operations"].keys()
    for operation in report["operations"].values():
        assert {"loss", "gradient_norm", "max_error", "gradient_max_error", "seconds"} <= operation.keys()
