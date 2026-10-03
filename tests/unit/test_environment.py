"""Gate tests catch false passes, leaked identifiers, and implicit CPU fallback."""
import hashlib
import json
from types import SimpleNamespace


def test_unavailable_mps_is_blocked(monkeypatch):
    from ralup import environment
    fake = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(
        is_built=lambda: True, is_available=lambda: False)), mps=SimpleNamespace())
    monkeypatch.setattr(environment, "_load_torch", lambda: fake)
    report = environment._probe_document("mps")
    assert report["status"] == "blocked"
    assert "unavailable" in " ".join(report["blockers"]).lower()
    assert report["operations"] == {}


def test_missing_optional_apis_are_named():
    from ralup.environment import _api_inventory
    report = _api_inventory(SimpleNamespace(mps=SimpleNamespace(), linalg=SimpleNamespace()))
    assert report["mps.recommended_max_memory"] is False
    assert report["mps.get_rng_state"] is False
    assert report["linalg.qr"] is False
    assert report["func.grad"] is False


def test_redaction_removes_nested_identifiers_and_raw_command_lines():
    from ralup.environment import redact_hardware
    serial = "C02SECRET123"
    uuid = "ABCD-1234-SECRET"
    result = redact_hardware({"serial_number": serial, "hardware_uuid": uuid,
        "raw": f"Serial Number (system): {serial}\nHardware UUID: {uuid}\nProvisioning UDID: secret-udid\nChip: Apple M3 Max",
        "nested": [{"provisioning_udid": "secret-udid"}]})
    encoded = json.dumps(result)
    assert serial not in encoded and uuid not in encoded and "secret-udid" not in encoded
    assert "Apple M3 Max" in encoded


def test_enabled_fallback_blocks_in_process_probe(monkeypatch):
    from ralup import environment
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    report = environment._probe_document("mps")
    assert report["status"] == "blocked"
    assert "fallback" in " ".join(report["blockers"]).lower()


def test_cpu_probe_is_fresh_disables_fallback_and_hashes_evidence(tmp_path, monkeypatch):
    from ralup.environment import probe_operations
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    result = probe_operations("cpu")
    assert result.status == "pass", result.blockers
    ref, = result.evidence
    assert hashlib.sha256(ref.path.read_bytes()).hexdigest() == ref.sha256
    report = json.loads(ref.path.read_text())
    assert report["fallback"] == "0"
    assert {"matmul_backward", "expm1", "rng_restore", "complete_forward_backward"} <= report["operations"].keys()
    assert report["operations"]["rng_restore"]["max_error"] == 0
    assert all(row["gradient_max_error"] == 0 for row in report["operations"].values() if row["status"] == "pass")


def test_doctor_cli_writes_hashed_discovery(tmp_path, capsys):
    from ralup.__main__ import main
    destination = tmp_path / "environment.json"
    assert main(["doctor", "--output", str(destination)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    report = json.loads(destination.read_text())
    assert "python" in report and "commands" in report and "apis" in report


def test_missing_optional_operation_keeps_raw_failure_without_blocking_core(monkeypatch):
    from ralup import environment
    torch = environment._load_torch()
    monkeypatch.delattr(torch.linalg, "qr")
    report = environment._probe_document("cpu")
    assert report["apis"]["linalg.qr"] is False
    assert report["operations"]["qr"]["status"] == "failed"
    assert "AttributeError" in report["operations"]["qr"]["error"]
    assert report["operations"]["matmul_backward"]["status"] == "pass"
    assert report["status"] == "pass"


def test_nonfinite_core_operation_cannot_pass(monkeypatch, tmp_path):
    from ralup import environment
    torch = environment._load_torch()
    original = environment._operation
    def with_nonfinite(torch_module, name, device):
        if name == "expm1":
            return torch.tensor(float("nan")), torch.ones(1)
        return original(torch_module, name, device)
    monkeypatch.setattr(environment, "_operation", with_nonfinite)
    report = environment._probe_document("cpu")
    assert report["status"] == "negative"
    assert report["operations"]["expm1"]["status"] == "failed"
    assert any("expm1" in reason for reason in report["blockers"])
    ref = environment._write_report(tmp_path / "negative.json", report)
    assert json.loads(ref.path.read_text())["status"] == "negative"
