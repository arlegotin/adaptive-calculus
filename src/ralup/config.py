"""Finite canonical JSON and evidence-gated, immutable configuration snapshots."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType


class UnresolvedConfig(ValueError):
    """Required measured inputs or prerequisite gates have not been supplied."""

    def __init__(self, blockers: list[str]):
        self.blockers = tuple(blockers)
        super().__init__("Unresolved configuration: " + "; ".join(blockers))


def _json_value(value, *, frozen=False):
    if isinstance(value, (dict, MappingProxyType)):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_value(item, frozen=frozen) for key, item in value.items()}
    if isinstance(value, list) or (frozen and isinstance(value, tuple)):
        return [_json_value(item, frozen=frozen) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    raise TypeError(f"Not a JSON value: {type(value).__name__}")


def canonical_bytes(value: dict) -> bytes:
    """Encode strict JSON objects, also accepting our immutable payload snapshots."""
    if not isinstance(value, (dict, MappingProxyType)):
        raise TypeError("Canonical JSON root must be an object")
    plain = _json_value(value, frozen=isinstance(value, MappingProxyType))
    return json.dumps(plain, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, init=False)
class ResolvedConfig:
    payload: Mapping
    sha256: str

    def __init__(self, payload: dict):
        encoded = canonical_bytes(payload)
        object.__setattr__(self, "payload", _freeze(json.loads(encoded)))
        object.__setattr__(self, "sha256", hashlib.sha256(encoded).hexdigest())


PURPOSES = ("doctor", "prepare-data", "smoke", "pilot-bootstrap", "calibration", "train", "meta", "confirm")
_BASE_FIELDS = (
    "data.revision", "data.manifest", "runtime.device", "runtime.dependency_lock",
    "runtime.environment", "runtime.tolerances", "quantization.rotation",
    "training.microbatch", "training.warm_start_steps", "training.continuation_steps",
    "training.b0", "training.recipe",
)
_BOOTSTRAP_FIELDS = tuple(field for field in _BASE_FIELDS if field not in
                          ("training.warm_start_steps", "training.continuation_steps", "training.b0"))
_POLICY_FIELDS = ("policy.controller", "policy.normalizer", "policy.meta_sampling")
_CONFIRM_FIELDS = ("confirmation.b_star", "confirmation.seeds", "confirmation.n",
                   "confirmation.analysis", "confirmation.manifest")
_PROBE_FIELDS = (
    "optimizer_probe.name", "optimizer_probe.learning_rate", "optimizer_probe.betas",
    "optimizer_probe.epsilon", "optimizer_probe.decay", "meta.probe_horizon", "meta.probe_pairs",
    "meta.probe_sigma", "meta.outer_learning_rate", "meta.outer_betas", "meta.outer_epsilon",
    "meta.outer_decay",
)
_REQUIRED = {
    "doctor": (), "prepare-data": ("runtime.dependency_lock", "runtime.environment"),
    "smoke": _PROBE_FIELDS, "pilot-bootstrap": _BOOTSTRAP_FIELDS + _PROBE_FIELDS, "train": _BASE_FIELDS + _PROBE_FIELDS,
    "meta": _BASE_FIELDS + _POLICY_FIELDS + _PROBE_FIELDS,
    "confirm": _BASE_FIELDS + _POLICY_FIELDS + _CONFIRM_FIELDS + _PROBE_FIELDS,
}
_REQUIRED['calibration'] = _BOOTSTRAP_FIELDS + _PROBE_FIELDS
_GATES = {
    "doctor": (), "smoke": (), "prepare-data": ("WP00",),
    "pilot-bootstrap": ("WP00", "WP01", "H02", "H04", "H05"),
    "train": ("WP00", "WP01", "WP02", "WP04"),
    "meta": ("WP00", "WP01", "WP02", "WP04", "WP06"),
    "confirm": ("WP00", "WP01", "WP02", "WP04", "WP06", "WP07"),
}
_GATES['calibration'] = (*_GATES['pilot-bootstrap'], 'H07')


def _get(value, dotted):
    for key in dotted.split("."):
        value = value[key]
    return value


def _fill_nulls(template, evidence, prefix=""):
    for key, supplied in evidence.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in template:
            raise ValueError(f"Unknown evidence field: {path}")
        current = template[key]
        if current is None:
            template[key] = supplied
        elif isinstance(current, dict) and isinstance(supplied, dict):
            _fill_nulls(current, supplied, path)
        elif current != supplied or type(current) is not type(supplied):
            raise ValueError(f"Evidence cannot overwrite fixed field: {path}")


def _positive_integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _positive_number(value, name, *, allow_zero=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (value == 0 and not allow_zero)):
        bound = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {bound}")


def _betas(value, name):
    if (not isinstance(value, list) or len(value) != 2
            or any(type(beta) not in (int, float) or not math.isfinite(beta)
                   or not 0 <= beta < 1 for beta in value)):
        raise ValueError(f"{name} must contain exactly two finite numbers in [0, 1)")


def _validate_schema(payload):
    sections = {
        "schema_version", "profile", "synthetic", "data", "model", "quantization",
        "optimizer_probe", "runtime", "training", "policy", "meta", "confirmation",
    }
    if (set(payload) - {'calibration'} != sections or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1):
        raise ValueError("Expected schema_version 1 and exactly the documented sections")
    if 'calibration' in payload:
        settings = payload['calibration']
        keys = {'output', 'seed', 'repeats', 'pilot_target_budget', 'warmstart_target_budget', 'continuation_target_budget', 'doublings'}
        if not isinstance(settings, dict) or set(settings) != keys:
            raise ValueError('Invalid calibration settings')
        if not isinstance(settings['output'], str) or not settings['output'].strip():
            raise ValueError('Calibration output must be a nonempty path')
        for field in keys - {'output'}:
            _positive_integer(settings[field], 'calibration.' + field, minimum=0 if field == 'seed' else 2 if field == 'repeats' else 1)
        if settings['doublings'] > 16:
            raise ValueError('Calibration duration jobs must be bounded by 16 doublings')
    if payload["profile"] not in ("smoke", "proxy", "core", "meta"):
        raise ValueError("Unknown configuration profile")
    if type(payload["synthetic"]) is not bool:
        raise ValueError("synthetic must be explicit boolean")
    section_keys = {
        "data": {"source", "subset", "revision", "manifest", "vocabulary_size",
                 "boundary_token", "context_length", "target_block_bytes", "logical_batch_bytes",
                 "split_domain", "split_fractions", "near_duplicate"},
        "model": {"layers", "width", "heads", "parameters", "quantized_parameters", "mlp_ratio",
                  "dropout", "linear_bias", "pre_layernorm", "layernorm_epsilon", "gelu_approximate",
                  "tied_embeddings", "initialization_std", "residual_std_rule"},
        "quantization": {"format", "simulation", "levels", "scope", "other_dtype",
                         "moment_coordinates", "rotation", "assistance_tau_one"},
        "optimizer_probe": {"name", "learning_rate", "betas", "epsilon", "decay"},
        "runtime": {"device", "dependency_lock", "environment", "tolerances"},
        "training": {"microbatch", "warm_start_steps", "continuation_steps", "b0", "recipe"},
        "policy": {"controller", "normalizer", "meta_sampling", "features", "parameters",
                   "initial_bias", "alpha_range", "h_range", "tau_range", "loss_ema_decay",
                   "normalizer_std_floor", "standardized_clip"},
        "meta": {"probe_horizon", "probe_pairs", "probe_sigma", "outer_learning_rate",
                 "outer_betas", "outer_epsilon", "outer_decay"},
        "confirmation": {"b_star", "seeds", "n", "analysis", "manifest"},
    }
    for section, keys in section_keys.items():
        if not isinstance(payload[section], dict) or set(payload[section]) != keys:
            raise ValueError(f"Invalid fields in {section}")
    normative = {
        "model.mlp_ratio": 4, "model.dropout": 0, "model.linear_bias": False,
        "model.pre_layernorm": True, "model.layernorm_epsilon": 1e-5,
        "model.gelu_approximate": "none", "model.tied_embeddings": False,
        "model.initialization_std": 0.02, "model.residual_std_rule": "0.02/sqrt(2*layers)",
        "quantization.levels": [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
        "data.split_domain": "ralup-split-v1", "data.split_fractions": [0.20, 0.05, 0.75],
        "data.near_duplicate": {"normalization": ["NFKC", "lowercase", "whitespace"],
                                "shingle_tokens": 13, "minhash_functions": 128, "bands": 16,
                                "rows_per_band": 8, "jaccard_threshold": 0.8},
        "policy.features": 14, "policy.parameters": 45, "policy.initial_bias": [0, 0, 1],
        "policy.alpha_range": [0, 1], "policy.h_range": [0.25, 4],
        "policy.tau_range": [0.125, 1], "policy.loss_ema_decay": 0.95,
        "policy.normalizer_std_floor": 1e-6, "policy.standardized_clip": [-5, 5],
    }
    for name, expected in normative.items():
        actual = _get(payload, name)
        if actual != expected or (type(expected) is bool and type(actual) is not bool):
            raise ValueError(f"{name} differs from the normative scientific contract")
    model = payload["model"]
    for key in ("layers", "width", "heads"):
        _positive_integer(model[key], f"model.{key}")
    layers, width = model["layers"], model["width"]
    shape = {"smoke": (1, 8, 2), "proxy": (2, 128, 4),
             "core": (4, 256, 4), "meta": (2, 128, 4)}[payload["profile"]]
    if (layers, width, model["heads"]) != shape:
        raise ValueError("Model shape disagrees with the declared profile")
    count = 12 * layers * width**2 + 4 * layers * width + 2 * width + 2 * 257 * width + 256 * width
    if width % model["heads"] or model["parameters"] != count:
        raise ValueError("Model dimensions or parameter count disagree with S11")
    if model["quantized_parameters"] != 12 * layers * width**2:
        raise ValueError("Internal projection count disagrees with S11")
    q = payload["quantization"]
    if (q["format"], q["simulation"], q["scope"], q["other_dtype"],
        q["moment_coordinates"], q["assistance_tau_one"]) != (
            "E2M1", "W4A32", "internal_projections", "float32", "W", "bypass"):
        raise ValueError("Quantization contract differs from S13/S15/S18")
    if q["rotation"] is not None and q["rotation"] not in ("identity", "hadamard"):
        raise ValueError("Unsupported quantization.rotation")
    data = payload["data"]
    if (data["vocabulary_size"], data["boundary_token"], data["context_length"],
        data["target_block_bytes"], data["logical_batch_bytes"]) != (257, 256, 256, 128, 8192):
        raise ValueError("Byte scoring contract differs from S10")
    if payload["synthetic"] != (payload["profile"] == "smoke"):
        raise ValueError("Only the smoke profile may be synthetic")
    if payload["synthetic"] and data["source"] != "synthetic-bytes-v1":
        raise ValueError("Smoke requires explicitly synthetic data")
    if not payload["synthetic"] and (data["source"], data["subset"]) != (
            "Salesforce/wikitext", "wikitext-103-raw-v1"):
        raise ValueError("Scientific data must follow S08")


def _validate_values(payload):
    from .records import ArtifactRef

    optimizer = payload["optimizer_probe"]
    if optimizer["name"] is not None and optimizer["name"] != "AdamW":
        raise ValueError("optimizer_probe.name must be AdamW")
    for section, fields in (
            ("optimizer_probe", ("learning_rate", "epsilon", "decay")),
            ("meta", ("probe_sigma", "outer_learning_rate", "outer_epsilon", "outer_decay"))):
        for field in fields:
            value = payload[section][field]
            if value is not None:
                _positive_number(value, f"{section}.{field}", allow_zero=field.endswith("decay"))
    for section, field in (("optimizer_probe", "betas"), ("meta", "outer_betas")):
        value = payload[section][field]
        if value is not None:
            _betas(value, f"{section}.{field}")
    for field in ("probe_horizon", "probe_pairs"):
        value = payload["meta"][field]
        if value is not None:
            _positive_integer(value, f"meta.{field}")
    if payload["meta"]["outer_decay"] is not None and payload["meta"]["outer_decay"] != 0:
        raise ValueError("meta.outer_decay must be zero: S21 outer Adam has no decay")

    artifact_fields = (
        "data.manifest", "runtime.dependency_lock", "runtime.environment", "training.b0",
        "policy.controller", "policy.normalizer", "confirmation.b_star",
        "confirmation.analysis", "confirmation.manifest",
    )
    for name in artifact_fields:
        value = _get(payload, name)
        if value is not None:
            if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
                raise ValueError(f"{name} must be a path/SHA-256 artifact reference")
            ArtifactRef(**value)
    revision = payload["data"]["revision"]
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("data.revision must be a nonempty exact revision")
    training = payload["training"]
    for name in ("microbatch", "warm_start_steps", "continuation_steps"):
        if training[name] is not None:
            _positive_integer(training[name], f"training.{name}",
                              minimum=0 if name == "warm_start_steps" else 1)
    if training["recipe"] is not None and training["recipe"] not in (
            "row_fixed_max", "row_dynamic_max", "quest_fp4_row_reference"):
        raise ValueError("Unsupported training.recipe")
    runtime = payload["runtime"]
    if runtime["device"] is not None and runtime["device"] not in ("cpu", "mps"):
        raise ValueError("runtime.device must be cpu or mps")
    tolerances = runtime["tolerances"]
    if tolerances is not None:
        if not isinstance(tolerances, dict) or not tolerances:
            raise ValueError("Measured tolerances must be a nonempty object")
        for key, value in tolerances.items():
            if type(value) not in (int, float) or value < 0:
                raise ValueError(f"Tolerance {key} must be nonnegative finite numeric")
    sampling = payload["policy"]["meta_sampling"]
    if sampling is not None:
        if (not isinstance(sampling, dict) or not sampling
                or any(type(v) not in (int, float) or v < 0 for v in sampling.values())
                or not math.isclose(sum(sampling.values()), 1, abs_tol=1e-12)):
            raise ValueError("meta_sampling must contain nonnegative proportions summing to one")
    confirm = payload["confirmation"]
    if confirm["n"] is not None:
        _positive_integer(confirm["n"], "confirmation.n")
    if confirm["seeds"] is not None:
        seeds = confirm["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("confirmation.seeds must be a nonempty unique integer list")
        for seed in seeds:
            _positive_integer(seed, "confirmation seed", minimum=0)
        if len(set(seeds)) != len(seeds) or (confirm["n"] is not None and len(seeds) != confirm["n"]):
            raise ValueError("Confirmation seeds must be unique and match frozen n")


def resolve_config(template: dict, evidence: dict, *, purpose: str) -> ResolvedConfig:
    """Resolve null inputs using measured gate provenance; never modify a template."""
    from .records import ArtifactRef, GateResult

    if purpose not in PURPOSES:
        raise ValueError(f"Unknown purpose: {purpose}")
    payload = json.loads(canonical_bytes(template))
    measured = json.loads(canonical_bytes(evidence))
    _validate_schema(payload)
    if purpose == "smoke" and not payload["synthetic"]:
        raise ValueError("Smoke purpose requires an explicitly synthetic smoke profile")
    if payload["synthetic"] and purpose not in ("smoke", "doctor", "calibration"):
        raise ValueError("Synthetic smoke cannot be used for scientific runs")
    gates = measured.pop("gates", {})
    if not isinstance(gates, dict):
        raise ValueError("evidence.gates must be an object")
    _fill_nulls(payload, measured)
    _validate_schema(payload)
    _validate_values(payload)
    if purpose == "smoke":
        training = payload["training"]
        if not (type(training["continuation_steps"]) is int
                and 1 <= training["continuation_steps"] <= 16
                and training["warm_start_steps"] == 0 and training["microbatch"] == 1):
            raise ValueError("Smoke requires microbatch one, no warm start, and 1..16 finite steps")
    if purpose == 'calibration':
        if 'calibration' not in payload:
            raise ValueError('Calibration requires explicit finite pilot settings')
        if payload['synthetic'] and max(payload['calibration']['pilot_target_budget'], payload['calibration']['warmstart_target_budget']) > 16*8192:
            raise ValueError('Synthetic calibration exceeds bounded engineering horizon')
        if not payload['synthetic'] and payload['calibration']['pilot_target_budget'] < 8192*payload['calibration']['repeats']:
            raise ValueError('Real sustained calibration requires full 8192-byte logical steps')
    required = (_PROBE_FIELDS + ('runtime.device', 'training.recipe', 'training.microbatch')
                if purpose == 'calibration' and payload['synthetic'] else _REQUIRED[purpose])
    required_gates = () if purpose == 'calibration' and payload['synthetic'] else _GATES[purpose]
    blockers = [f"missing {field}" for field in required if _get(payload, field) is None]
    for name in required_gates:
        record = gates.get(name)
        if record is None:
            blockers.append(f"missing {name} gate")
            continue
        if not isinstance(record, dict) or set(record) != {"status", "evidence", "blockers"}:
            raise ValueError(f"Invalid gate record: {name}")
        if not isinstance(record["evidence"], list) or not isinstance(record["blockers"], list):
            raise ValueError(f"Gate evidence/blockers must be lists: {name}")
        gate = GateResult(record["status"], tuple(ArtifactRef(**item) for item in record["evidence"]),
                          tuple(record["blockers"]))
        if gate.status != "pass" or not gate.evidence:
            blockers.append(f"{name} is {gate.status} or has no hashed evidence")
    if blockers:
        raise UnresolvedConfig(blockers)
    if purpose == "pilot-bootstrap" or (purpose == 'calibration' and not payload['synthetic']):
        # Bootstrap acceptance is engineering scope, never a fabricated full
        # WP02/WP04/WP06 gate. Rehash all actual prerequisite artifacts at entry.
        from .records import sha256_file
        references = [_get(payload,name) for name in
                      ("data.manifest","runtime.dependency_lock","runtime.environment")]
        references += [item for name in required_gates for item in gates[name]["evidence"]]
        for item in references:
            ref = ArtifactRef(**item)
            if sha256_file(ref.path) != ref.sha256:
                raise ValueError(f"Bootstrap artifact hash mismatch: {ref.path}")
    payload["purpose"] = purpose
    payload["evidence_gates"] = {name: gates[name] for name in required_gates}
    return ResolvedConfig(payload)
