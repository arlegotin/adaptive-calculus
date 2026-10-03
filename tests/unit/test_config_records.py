"""Contracts for scientific gating and auditable, independent work counts."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ralup.config import UnresolvedConfig, canonical_bytes, resolve_config
from ralup.records import ArtifactRef, GateResult, RunLedger, sha256_file


ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64


def template(name="core"):
    return json.loads((ROOT / "configs" / f"{name}.json").read_text())


def evidence():
    artifact = {"path": "artifacts/measured.json", "sha256": HASH}
    return {
        "data": {"revision": "measured-revision", "manifest": artifact},
        "runtime": {"device": "cpu", "dependency_lock": artifact,
                    "environment": artifact, "tolerances": {"fp32_loss_atol": 0.000001}},
        "quantization": {"rotation": "identity"},
        "training": {"microbatch": 1, "warm_start_steps": 16,
                     "continuation_steps": 8, "b0": artifact,
                     "recipe": "row_fixed_max"},
        "policy": {"controller": artifact, "normalizer": artifact,
                   "meta_sampling": {"early": 0.5, "late": 0.5}},
        "confirmation": {"b_star": artifact, "seeds": [101, 202],
                         "n": 2, "analysis": artifact, "manifest": artifact},
        "gates": {wp: {"status": "pass", "evidence": [artifact], "blockers": []}
                  for wp in ("WP00", "WP01", "WP02", "WP04", "WP06", "WP07")},
    }


def metadata():
    return {
        "run_id": "fixture-001", "created_utc": "2026-10-03T12:00:00Z",
        "local_timezone": "Europe/Amsterdam", "phase": "smoke",
        "source": {"commit": "b" * 40, "dirty_diff_sha256": None},
        "lock_sha256": HASH, "environment_sha256": HASH,
        "resolved_config": {"synthetic": True},
        "method": "synthetic_fixture", "adaptation": "fixture",
        "seeds": {"stream": 17}, "data_sha256": HASH,
        "warm_start_sha256": None, "policy_sha256": None,
        "lineage": {"parent": None, "resume": None},
        "endpoint": {"steps": 2},
    }


def work(**changes):
    result = {"step": 0, "stream": 8, "unique": 8, "additional_loss": 4,
              "backward": 8, "query": 4, "elapsed_seconds": 0.25}
    result.update(changes)
    return result


def test_canonical_hash_is_independent_of_key_order():
    assert canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert canonical_bytes({"word": "é"}) == b'{"word":"\xc3\xa9"}'
    first = resolve_config(template("smoke"), {}, purpose="smoke")
    reversed_keys = dict(reversed(list(template("smoke").items())))
    assert resolve_config(reversed_keys, {}, purpose="smoke").sha256 == first.sha256


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_rejects_nonfinite_nested_numbers(value):
    with pytest.raises(ValueError):
        canonical_bytes({"nested": [{"loss": value}]})


@pytest.mark.parametrize("value", [{1: "not a JSON key"}, {"tuple": (1, 2)}, {"set": {1}}, []])
def test_canonical_rejects_non_json_input(value):
    with pytest.raises((TypeError, ValueError)):
        canonical_bytes(value)


def test_missing_evidence_blocks_training():
    with pytest.raises(UnresolvedConfig) as caught:
        resolve_config(template(), {}, purpose="train")
    assert "data.revision" in str(caught.value)
    assert "WP01" in str(caught.value)


def test_evidence_resolves_without_changing_template_and_returns_immutable_snapshot():
    original = template()
    measured = evidence()
    resolved = resolve_config(original, measured, purpose="train")
    assert original["data"]["revision"] is None
    assert resolved.payload["data"]["revision"] == "measured-revision"
    measured["runtime"]["tolerances"]["fp32_loss_atol"] = 99
    assert resolved.payload["runtime"]["tolerances"]["fp32_loss_atol"] == 0.000001
    with pytest.raises(TypeError):
        resolved.payload["data"]["revision"] = "changed"
    assert resolved.sha256 == hashlib.sha256(canonical_bytes(resolved.payload)).hexdigest()


def test_scientific_constants_cannot_be_overridden_by_evidence():
    measured = evidence()
    measured["model"] = {"width": 1}
    with pytest.raises(ValueError):
        resolve_config(template(), measured, purpose="train")


@pytest.mark.parametrize("gate", ["WP00", "WP01", "WP02", "WP04"])
def test_failed_or_empty_gate_is_not_evidence(gate):
    measured = evidence()
    measured["gates"][gate]["status"] = "blocked"
    measured["gates"][gate]["blockers"] = ["no measured evidence"]
    with pytest.raises(UnresolvedConfig):
        resolve_config(template(), measured, purpose="train")
    measured["gates"][gate].update(status="pass", blockers=[], evidence=[])
    with pytest.raises(ValueError):
        resolve_config(template(), measured, purpose="train")


def test_meta_and_confirmation_have_additional_evidence_gates():
    measured = evidence()
    measured["gates"].pop("WP06")
    assert resolve_config(template(), measured, purpose="train")
    with pytest.raises(UnresolvedConfig):
        resolve_config(template("meta"), measured, purpose="meta")
    measured = evidence()
    measured["gates"].pop("WP07")
    with pytest.raises(UnresolvedConfig):
        resolve_config(template(), measured, purpose="confirm")


def test_confirmation_seed_count_must_match_frozen_n():
    measured = evidence()
    measured["confirmation"]["n"] = 3
    with pytest.raises(ValueError):
        resolve_config(template(), measured, purpose="confirm")


@pytest.mark.parametrize("purpose", ["train", "meta", "confirm"])
def test_synthetic_smoke_cannot_be_used_for_scientific_runs(purpose):
    with pytest.raises(ValueError):
        resolve_config(template("smoke"), evidence(), purpose=purpose)


@pytest.mark.parametrize("changes", [
    {"microbatch": 0}, {"continuation_steps": None}, {"warm_start_steps": True},
    {"recipe": "made_up"},
])
def test_training_resolution_rejects_invalid_measurements(changes):
    measured = evidence()
    measured["training"].update(changes)
    with pytest.raises(ValueError):
        resolve_config(template(), measured, purpose="train")


def test_unknown_purpose_and_unknown_template_fields_are_rejected():
    with pytest.raises(ValueError):
        resolve_config(template(), {}, purpose="typo")
    invalid = template("smoke")
    invalid["typo"] = 1
    with pytest.raises(ValueError):
        resolve_config(invalid, {}, purpose="smoke")


@pytest.mark.parametrize("path,value", [
    ("schema_version", True), ("model.gelu_approximate", "tanh"),
    ("model.layernorm_epsilon", 0.1), ("quantization.levels", [0, 1]),
    ("policy.features", 15), ("model.dropout", 0.1),
    ("data.near_duplicate.jaccard_threshold", 0.7),
])
def test_schema_rejects_changes_to_normative_contract(path, value):
    invalid = template()
    target = invalid
    parts = path.split(".")
    for key in parts[:-1]:
        target = target[key]
    target[parts[-1]] = value
    with pytest.raises(ValueError):
        resolve_config(invalid, evidence(), purpose="train")


def test_early_phases_do_not_require_later_scientific_inputs():
    assert resolve_config(template(), {}, purpose="doctor")
    measured = evidence()
    minimal = {"runtime": {key: measured["runtime"][key]
                           for key in ("dependency_lock", "environment")},
               "gates": {"WP00": measured["gates"]["WP00"]}}
    resolved = resolve_config(template(), minimal, purpose="prepare-data")
    assert resolved.payload["data"]["revision"] is None
    assert resolved.payload["training"]["continuation_steps"] is None
    assert resolved.payload["runtime"]["tolerances"] is None


def test_smoke_has_an_enforced_finite_step_limit():
    invalid = template("smoke")
    invalid["training"]["continuation_steps"] = 17
    with pytest.raises(ValueError):
        resolve_config(invalid, {}, purpose="smoke")


@pytest.mark.parametrize("status", ["negative", "inconclusive"])
def test_negative_and_inconclusive_gates_block_dependent_training(status):
    measured = evidence()
    measured["gates"]["WP01"]["status"] = status
    with pytest.raises(UnresolvedConfig):
        resolve_config(template(), measured, purpose="train")


def test_file_hash_and_artifact_validation(tmp_path):
    asset = tmp_path / "asset"
    asset.write_bytes(b"abc")
    assert sha256_file(asset) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    reference = ArtifactRef(asset, sha256_file(asset))
    assert reference.path == asset
    with pytest.raises(ValueError):
        ArtifactRef(asset, "not-a-hash")
    with pytest.raises(ValueError):
        GateResult("pass", (), ("still blocked",))


def test_run_metadata_is_exclusive_and_preserved_during_append(tmp_path):
    values = metadata()
    ledger = RunLedger(tmp_path, values)
    before = (tmp_path / "run.json").read_bytes()
    values["seeds"]["stream"] = 99
    ledger.record(work())
    first_metrics = (tmp_path / "metrics.jsonl").read_bytes()
    ledger.record(work(step=1, unique=0))
    ledger.record_status("completed", disposition="synthetic fixture only")
    assert (tmp_path / "metrics.jsonl").read_bytes().startswith(first_metrics)
    assert (tmp_path / "run.json").read_bytes() == before
    assert json.loads(before)["seeds"]["stream"] == 17
    with pytest.raises(FileExistsError):
        RunLedger(tmp_path, metadata())
    reopened = RunLedger(tmp_path)
    reopened.record(work(step=2))
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["run_id"] == "fixture-001"
    assert {key: rows[0][key] for key in ("stream", "unique", "additional_loss", "backward", "query")} == {
        "stream": 8, "unique": 8, "additional_loss": 4, "backward": 8, "query": 4,
    }
    assert "total_targets" not in rows[0]
    statuses = [json.loads(line) for line in (tmp_path / "status.jsonl").read_text().splitlines()]
    assert [row["status"] for row in statuses] == ["created", "completed"]


@pytest.mark.parametrize("changes", [
    {"stream": -1}, {"unique": 9}, {"backward": 1.5}, {"query": True},
    {"elapsed_seconds": float("nan")}, {"elapsed_seconds": -1},
    {"run_id": "other-run"}, {"nll_sum": float("inf")},
])
def test_invalid_work_does_not_append(tmp_path, changes):
    ledger = RunLedger(tmp_path, metadata())
    with pytest.raises(ValueError):
        ledger.record(work(**changes))
    assert not (tmp_path / "metrics.jsonl").exists()


def test_incomplete_metadata_does_not_publish_run(tmp_path):
    values = metadata()
    del values["source"]
    with pytest.raises(ValueError):
        RunLedger(tmp_path, values)
    assert not (tmp_path / "run.json").exists()


def test_ledger_rejects_metadata_tampering_and_incomplete_metric_tail(tmp_path):
    ledger = RunLedger(tmp_path, metadata())
    original = (tmp_path / "run.json").read_bytes()
    (tmp_path / "run.json").write_bytes(original + b" ")
    with pytest.raises(ValueError):
        ledger.record(work())
    (tmp_path / "run.json").write_bytes(original)
    (tmp_path / "metrics.jsonl").write_bytes(b'{"step":')
    with pytest.raises(ValueError):
        ledger.record(work())
    assert (tmp_path / "metrics.jsonl").read_bytes() == b'{"step":'


def test_cli_exposes_only_implemented_commands_without_artifacts(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([sys.executable, "-m", "ralup", "--help"], cwd=tmp_path,
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0
    assert "config" in result.stdout
    result = subprocess.run([sys.executable, "-m", "ralup", "config", "--help"], cwd=tmp_path,
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0
    assert "--purpose" in result.stdout
    assert list(tmp_path.iterdir()) == []
