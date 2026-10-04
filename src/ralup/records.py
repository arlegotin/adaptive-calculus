"""Hashed artifact references and durable, append-only single-writer run ledgers."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import _freeze, canonical_bytes, resolve_config


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_PHASE_PURPOSES = {"smoke": ("smoke",), "explore": ("pilot-bootstrap", "train", "meta"), "confirm": ("confirm",)}


def _check_hash(value, name="sha256", *, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    sha256: str

    def __post_init__(self):
        if not isinstance(self.path, (str, Path)) or not str(self.path).strip():
            raise ValueError("Artifact path must be nonempty")
        object.__setattr__(self, "path", Path(self.path))
        _check_hash(self.sha256)


@dataclass(frozen=True)
class GateResult:
    status: str
    evidence: tuple[ArtifactRef, ...]
    blockers: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "blockers", tuple(self.blockers))
        if self.status not in ("pass", "blocked", "negative", "inconclusive"):
            raise ValueError("Unknown gate status")
        if any(not isinstance(ref, ArtifactRef) for ref in self.evidence):
            raise ValueError("Gate evidence must contain ArtifactRef values")
        if any(not isinstance(reason, str) or not reason.strip() for reason in self.blockers):
            raise ValueError("Blockers must be nonempty strings")
        if self.status == "pass" and (self.blockers or not self.evidence):
            raise ValueError("A passing gate requires evidence and no blockers")


_REQUIRED_METADATA = {
    "run_id", "created_utc", "local_timezone", "phase", "source", "lock_sha256",
    "environment_sha256", "resolved_config", "resolved_config_sha256", "method", "adaptation", "seeds", "data_sha256",
    "warm_start_sha256", "policy_sha256", "lineage", "endpoint",
}
_COUNTS = ("stream", "unique", "additional_loss", "backward", "query")
_OPTIONAL_METRICS = {
    "forward_context": None, "nll_sum": None, "nll_count": None,
    "optimizer_seconds": None, "quantizer_seconds": None, "policy_seconds": None,
    "rss_bytes": None, "mps_allocated_bytes": None, "mps_driver_bytes": None,
    "checkpoint_sha256": None, "nonfinite_count": None, "status": "completed_step",
    "matrices": None,
}


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _finite_nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _validate_metadata(metadata):
    missing = _REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError("Missing run metadata: " + ", ".join(sorted(missing)))
    for name in ("run_id", "local_timezone", "phase", "method", "adaptation"):
        if not isinstance(metadata[name], str) or not metadata[name].strip():
            raise ValueError(f"{name} must be nonempty")
    try:
        created = datetime.fromisoformat(metadata["created_utc"].replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("created_utc must be an ISO-8601 UTC timestamp") from error
    if created.tzinfo is None or created.utcoffset().total_seconds() != 0:
        raise ValueError("created_utc must be in UTC")
    try:
        ZoneInfo(metadata["local_timezone"])
    except ZoneInfoNotFoundError as error:
        raise ValueError("local_timezone must be an IANA timezone") from error
    source = metadata["source"]
    if not isinstance(source, dict) or set(source) != {"commit", "dirty_diff_sha256"}:
        raise ValueError("source requires commit and dirty_diff_sha256")
    if not isinstance(source["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source["commit"]):
        raise ValueError("source.commit must be a Git commit digest")
    _check_hash(source["dirty_diff_sha256"], "source.dirty_diff_sha256", nullable=True)
    for name in ("lock_sha256", "environment_sha256", "data_sha256"):
        _check_hash(metadata[name], name, nullable=metadata["phase"] == "smoke")
    for name in ("warm_start_sha256", "policy_sha256"):
        _check_hash(metadata[name], name, nullable=True)
    for name in ("resolved_config", "seeds", "lineage", "endpoint"):
        if not isinstance(metadata[name], dict) or not metadata[name]:
            raise ValueError(f"{name} must be a nonempty object")
    snapshot = metadata["resolved_config"]
    _check_hash(metadata["resolved_config_sha256"], "resolved_config_sha256")
    if hashlib.sha256(canonical_bytes(snapshot)).hexdigest() != metadata["resolved_config_sha256"]:
        raise ValueError("Resolved config payload does not match its SHA-256")
    if snapshot.get("purpose") not in _PHASE_PURPOSES.get(metadata["phase"], ()):
        raise ValueError("Run phase is incompatible with the resolved config purpose")
    if "evidence_gates" not in snapshot:
        raise ValueError("Resolved config must retain its gate provenance")
    template = dict(snapshot)
    purpose = template.pop("purpose")
    gates = template.pop("evidence_gates")
    validated = resolve_config(template, {"gates": gates}, purpose=purpose)
    if validated.sha256 != metadata["resolved_config_sha256"]:
        raise ValueError("Run config is not the complete canonical resolved snapshot")
    for name, seed in metadata["seeds"].items():
        _integer(seed, f"seeds.{name}")
    if set(metadata["lineage"]) != {"parent", "resume"}:
        raise ValueError("lineage requires explicit parent and resume")
    for value in metadata["lineage"].values():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("lineage identifiers must be nonempty strings or null")


class RunLedger:
    """One writer per run directory. Existing metadata can only be reopened unchanged.

    Work counters measure overlapping categories; there is deliberately no total.
    Optional measurements remain null until observed, rather than being invented zeros.
    """

    def __init__(self, path: Path, metadata: dict | None = None):
        self.path = Path(path)
        run_path = self.path / "run.json"
        if metadata is not None:
            copied = json.loads(canonical_bytes(metadata))
            _validate_metadata(copied)
            if "status_history" in copied and copied["status_history"] != "status.jsonl":
                raise ValueError("status_history must be status.jsonl")
            copied["status_history"] = "status.jsonl"
            encoded = canonical_bytes(copied) + b"\n"
            self.path.mkdir(parents=True, exist_ok=True)
            with run_path.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            encoded = run_path.read_bytes()
            copied = json.loads(encoded)
            canonical_bytes(copied)
            _validate_metadata(copied)
            if copied.get("status_history") != "status.jsonl":
                raise ValueError("Run lacks an append-only status history reference")
        self._metadata_bytes = encoded
        self.metadata = _freeze(copied)
        self.run_id = copied["run_id"]
        if metadata is not None:
            self.record_status("created")

    def _append(self, filename: str, record: dict):
        encoded = canonical_bytes(record) + b"\n"
        if (self.path / "run.json").read_bytes() != self._metadata_bytes:
            raise ValueError("Immutable run metadata was modified")
        destination = self.path / filename
        if destination.exists() and destination.stat().st_size:
            with destination.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b"\n":
                    raise ValueError("Incomplete ledger tail; refusing to append")
        with destination.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def record(self, work: dict) -> None:
        row = json.loads(canonical_bytes(work))
        required = {"step", "elapsed_seconds", *_COUNTS}
        missing = required - row.keys()
        if missing:
            raise ValueError("Missing work measurements: " + ", ".join(sorted(missing)))
        if "run_id" in row and row["run_id"] != self.run_id:
            raise ValueError("Work belongs to a different run")
        row["run_id"] = self.run_id
        for name in ("step", *_COUNTS):
            _integer(row[name], name)
        if row["unique"] > row["stream"]:
            raise ValueError("unique cannot exceed stream targets")
        _finite_nonnegative(row["elapsed_seconds"], "elapsed_seconds")
        for name, default in _OPTIONAL_METRICS.items():
            row.setdefault(name, default)
        for name in ("forward_context", "nll_count", "rss_bytes", "mps_allocated_bytes",
                     "mps_driver_bytes", "nonfinite_count"):
            if row[name] is not None:
                _integer(row[name], name)
        for name in ("nll_sum", "optimizer_seconds", "quantizer_seconds", "policy_seconds"):
            if row[name] is not None:
                _finite_nonnegative(row[name], name)
        _check_hash(row["checkpoint_sha256"], "checkpoint_sha256", nullable=True)
        if not isinstance(row["status"], str) or not row["status"].strip():
            raise ValueError("Metric status must be nonempty")
        if row["matrices"] is not None and not isinstance(row["matrices"], dict):
            raise ValueError("Per-matrix telemetry must be an object")
        if "total_targets" in row:
            raise ValueError("Overlapping work categories must not be summed")
        self._append("metrics.jsonl", row)

    def record_status(self, status: str, *, disposition: str | None = None) -> None:
        if status not in ("created", "running", "completed", "failed", "interrupted", "diverged"):
            raise ValueError("Unknown run status")
        if disposition is not None and (not isinstance(disposition, str) or not disposition.strip()):
            raise ValueError("Disposition must be nonempty or null")
        self._append("status.jsonl", {
            "run_id": self.run_id, "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "status": status, "disposition": disposition,
        })


class WorkLedger:
    """Three independent cost views, plus retained completed/failure events.

    Stream advances only on a completed transaction. Pass counts and whole-step
    elapsed time include failed work. Optional RunLedger writes durable events
    before publishing these in-memory totals. H07 will snapshot deterministic
    counts separately from wall-clock observations.
    """
    def __init__(self, run_ledger: RunLedger | None = None):
        self.stream = {'targets':0,'unique':0,'updates':0}
        self.coverage = {}  # Coalesced scored intervals for cursor-free fixtures only.
        self.passes = {'loss_targets':0,'additional_loss':0,'backward':0,'query':0,'forward_context':0}
        self.elapsed = {'seconds':0.}
        self.events = []
        self.run_ledger = run_ledger

    def record(self, work: dict):
        row = json.loads(canonical_bytes(work))
        for key in (*_COUNTS,'loss_targets','forward_context','step'):
            _integer(row[key],key)
        _finite_nonnegative(row['elapsed_seconds'],'elapsed_seconds')
        if self.run_ledger is not None:
            self.run_ledger.record(row)
        self.events.append(row)
        if row['status'] == 'completed_step':
            self.stream['targets'] += row['stream']
            self.stream['unique'] += row['unique']
            self.stream['updates'] += 1
        for key in self.passes:
            self.passes[key] += row[key]
        self.elapsed['seconds'] += row['elapsed_seconds']
