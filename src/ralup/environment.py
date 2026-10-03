"""Read-only discovery and isolated FP32 backend smoke evidence (not WP01)."""
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from .config import canonical_bytes
from .records import ArtifactRef, GateResult, sha256_file


_APIS = ("backends.mps.is_built", "backends.mps.is_available", "mps.synchronize",
         "mps.current_allocated_memory", "mps.driver_allocated_memory",
         "mps.recommended_max_memory", "mps.get_rng_state", "mps.set_rng_state",
         "linalg.qr", "linalg.eigh", "bucketize", "func.grad")
_OPTIONAL = {"bucketize", "qr", "eigh", "functional_grad"}
# Conservative predeclared smoke bounds; measured errors remain in the report.
# Scientific oracle/assistance/identity tolerances require later F03-F05 evidence.
_SMOKE_ATOL = 1e-4
_SMOKE_RTOL = 1e-4


def _load_torch():
    return importlib.import_module("torch")


def _api_inventory(torch):
    result = {}
    for name in _APIS:
        value = torch
        for part in name.split("."):
            value = getattr(value, part, None)
        result[name] = callable(value)
    return result


def redact_hardware(value):
    """Redact structured and system_profiler textual machine identifiers."""
    sensitive = re.compile(r"serial|uuid|udid|provisioning", re.IGNORECASE)
    if isinstance(value, dict):
        return {key: "[REDACTED]" if sensitive.search(key) else redact_hardware(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_hardware(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(?im)^([^\n]*(?:serial|uuid|udid|provisioning)[^\n:]*:)\s*[^\n]*",
                      r"\1 [REDACTED]", value)
    return value


def _command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "returncode": None, "error": str(error)}


def discover_environment() -> dict:
    commands = {
        "architecture": ["uname", "-m"], "macos": ["sw_vers"],
        "memory": ["sysctl", "-n", "hw.memsize"],
        "rosetta": ["sysctl", "-n", "sysctl.proc_translated"],
        "hardware": ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"],
        "storage": ["df", "-h", "."], "pressure": ["vm_stat"],
        "power": ["pmset", "-g", "batt"],
    }
    report = {"schema_version": 1, "recorded_utc": datetime.now(timezone.utc).isoformat(),
              "python": {"version": platform.python_version(), "machine": platform.machine(),
                         "executable": sys.executable, "isolated": sys.prefix != sys.base_prefix},
              "system": platform.system(), "commands": {key: _command(argv) for key, argv in commands.items()},
              "fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
              "packages": {}, "apis": {}, "mps": {}}
    for name in ("torch", "numpy", "scipy", "pyarrow", "safetensors"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        torch = _load_torch()
        report["apis"] = _api_inventory(torch)
        report["mps"] = {"built": torch.backends.mps.is_built(),
                         "available": torch.backends.mps.is_available()}
        report["mps"]["memory"] = _memory(torch, report["mps"]["available"])
    except (ImportError, AttributeError, RuntimeError) as error:
        report["torch_error"] = str(error)
    translated = report["commands"]["rosetta"].get("stdout", "").strip()
    report["architecture"] = {"process": platform.machine(),
                              "rosetta_translated": {"0": False, "1": True}.get(translated)}
    memory = report["commands"]["memory"].get("stdout", "").strip()
    report["physical_memory_bytes"] = int(memory) if memory.isdecimal() else None
    report["checkpoint_container"] = {"tensor": "safetensors", "metadata": "validated JSON",
        "executable": False, "resume_verified": False,
        "constraints": "CPU dense contiguous copies; explicit alias metadata; verify hashes/shapes/dtypes before transfer"}
    return redact_hardware(report)


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(redact_hardware(report)) + b"\n")
    return ArtifactRef(path.resolve(), sha256_file(path))


def write_environment_report(path: Path) -> ArtifactRef:
    return _write_report(path, discover_environment())


def _memory(torch, available):
    report = {}
    for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
        api = getattr(getattr(torch, "mps", None), name, None)
        if not callable(api):
            report[name] = {"status": "missing"}
        elif not available:
            report[name] = {"status": "unavailable"}
        else:
            try:
                report[name] = {"status": "pass", "bytes": api()}
            except Exception as error:
                report[name] = {"status": "failed", "error": str(error)}
    return report


def _operation(torch, name, device):
    """Small operator fixtures; generic surrogates do not implement format oracles."""
    F = torch.nn.functional
    generator = torch.Generator(device="cpu").manual_seed(173)
    x = (torch.randn((4, 8), generator=generator, dtype=torch.float32) * .2).to(device).requires_grad_()
    if name in ("matmul_backward", "complete_forward_backward", "state_updates"):
        w = (torch.randn((8, 8), generator=generator) * .2).to(device).requires_grad_()
    else:
        w = None
    if name == "matmul_backward":
        y = x @ w
    elif name == "layernorm":
        y = F.layer_norm(x, (8,), eps=1e-5)
    elif name == "gelu":
        y = F.gelu(x, approximate="none")
    elif name == "attention":
        q = x.reshape(1, 1, 4, 8)
        y = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    elif name == "reductions":
        y = x.mean(0) + x.sum(0) + torch.linalg.vector_norm(x, dim=0)
    elif name == "clipping":
        loss = x.square().sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x], .1)
        return loss.detach().cpu(), x.grad.detach().cpu()
    elif name == "expm1":
        y = torch.expm1(x)
    elif name == "comparison_indexing":
        y = x[x > 0]
    elif name == "hadamard":
        y = x
        for stride in (1, 2, 4):
            parts = y.reshape(4, -1, 2, stride)
            a, b = parts[:, :, 0, :], parts[:, :, 1, :]
            y = torch.stack((a + b, a - b), dim=2).reshape(4, 8)
        y = y / (8 ** .5)
    elif name in ("ste_backward", "trust_backward"):
        quantized = x.detach().round()  # Generic operator gate, not E2M1 semantics.
        mask = torch.ones_like(x) if name == "ste_backward" else ((x.detach() - quantized).abs() <= .15).float()
        y = quantized + (x - x.detach()) * mask
        # A nonzero upstream derivative exercises the surrogate despite zero values.
        loss = y.sum()
        loss.backward()
        return loss.detach().cpu(), x.grad.detach().cpu()
    elif name == "state_updates":
        g = torch.autograd.grad((x @ w).square().mean(), w)[0]
        m = .9 * torch.zeros_like(w) + .1 * g
        v = .99 * torch.zeros_like(w) + .01 * g.square()
        y = w - .001 * (m / .1) / ((v / .01).sqrt() + 1e-8)
    elif name == "rng_restore":
        if device == "mps":
            initial = torch.mps.get_rng_state()
            try:
                torch.mps.manual_seed(173)
                state = torch.mps.get_rng_state()
                first = torch.rand(16, device=device)
                torch.mps.set_rng_state(state)
                second = torch.rand(16, device=device)
            finally:
                torch.mps.set_rng_state(initial)
        else:
            state = generator.get_state()
            first = torch.rand(16, generator=generator)
            generator.set_state(state)
            second = torch.rand(16, generator=generator)
        # Compare restored draws, rather than assuming identical CPU/MPS algorithms.
        return (first - second).detach().cpu(), torch.zeros(1)
    elif name == "complete_forward_backward":
        y = F.gelu(F.layer_norm(x @ w, (8,), eps=1e-5), approximate="none")
        q = y.reshape(1, 1, 4, 8)
        logits = F.scaled_dot_product_attention(q, q, q, is_causal=True).reshape(4, 8)
        loss = F.cross_entropy(logits, torch.arange(4, device=device))
        loss.backward()
        return loss.detach().cpu(), torch.cat((x.grad.flatten(), w.grad.flatten())).detach().cpu()
    elif name == "bucketize":
        y = torch.bucketize(x, torch.tensor([-.2, 0, .2], device=device))
        return y.float().cpu(), torch.zeros(1)
    elif name == "qr":
        _, r = torch.linalg.qr(x)
        y = r.diagonal().abs()
    elif name == "eigh":
        y, _ = torch.linalg.eigh(x.T @ x + torch.eye(8, device=device))
    elif name == "functional_grad":
        y = torch.func.grad(lambda z: F.gelu(z, approximate="none").square().sum())(x)
    else:
        raise ValueError(name)
    loss = y.square().mean()
    loss.backward()
    gradient = w.grad.flatten() if name == "state_updates" else (x.grad.flatten() if w is None else torch.cat((x.grad.flatten(), w.grad.flatten())))
    return loss.detach().cpu(), gradient.detach().cpu()


def _probe_document(device):
    report = {"schema_version": 1, "device": device, "dtype": "float32", "status": "blocked",
              "recorded_utc": datetime.now(timezone.utc).isoformat(), "process_id": os.getpid(),
              "fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
              "blockers": [], "operations": {}, "apis": {},
              "acceptance": {"kind": "operator smoke only", "atol": _SMOKE_ATOL, "rtol": _SMOKE_RTOL,
                             "scientific_tolerances_frozen": False},
              "surrogate_scope": "Generic round STE/trust operator smoke; E2M1 oracle and MetaQuant parity remain later gates",
              "cpu_subroutines": "CPU seeded fixture creation/reference and detached result comparison; reference is timed separately, device fixture timing includes host/device transfers; no CPU execution substitutes for device operators"}
    if device not in ("cpu", "mps"):
        report["blockers"].append("Unsupported device")
        return report
    if device == "mps" and report["fallback"] == "1":
        report["blockers"].append("Silent CPU fallback is enabled")
        return report
    try:
        torch = _load_torch()
    except ImportError as error:
        report["blockers"].append(str(error))
        return report
    report["apis"] = _api_inventory(torch)
    if device == "mps" and not torch.backends.mps.is_available():
        report["blockers"].append("MPS unavailable in this process; inspect OS/sandbox/device access")
        return report
    report["torch_version"] = str(torch.__version__)
    report["memory_before"] = _memory(torch, device == "mps")
    operations = ("matmul_backward", "layernorm", "gelu", "attention", "reductions", "clipping",
                  "expm1", "comparison_indexing", "hadamard", "ste_backward", "trust_backward",
                  "state_updates", "rng_restore", "complete_forward_backward", "bucketize", "qr", "eigh", "functional_grad")
    for name in operations:
        row = {"status": "failed", "optional": name in _OPTIONAL, "loss": None, "gradient_norm": None,
               "max_error": None, "gradient_max_error": None, "seconds": None}
        start = time.perf_counter()
        try:
            cpu_start = time.perf_counter()
            expected, expected_grad = _operation(torch, name, "cpu")
            row["cpu_reference_seconds"] = time.perf_counter() - cpu_start
            if device == "mps":
                torch.mps.synchronize()
            device_start = time.perf_counter()
            actual, actual_grad = _operation(torch, name, device)
            if device == "mps":
                torch.mps.synchronize()
            row["seconds"] = time.perf_counter() - device_start
            repeat, repeat_grad = _operation(torch, name, device)
            row.update(loss=float(actual.float().mean()), gradient_norm=float(actual_grad.norm()),
                       max_error=float((actual - expected).abs().max()),
                       gradient_max_error=float((actual_grad - expected_grad).abs().max()),
                       duplicate_max_error=float((actual - repeat).abs().max()),
                       duplicate_gradient_max_error=float((actual_grad - repeat_grad).abs().max()))
            pairs = ((actual, expected), (actual_grad, expected_grad), (actual, repeat), (actual_grad, repeat_grad))
            if not all(torch.isfinite(a).all() and torch.isfinite(b).all() and
                       torch.allclose(a, b, atol=_SMOKE_ATOL, rtol=_SMOKE_RTOL) for a, b in pairs):
                raise ValueError("CPU/device or duplicate mismatch exceeds predeclared smoke bounds")
            row["status"] = "pass"
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            invalid = {key: str(value) for key, value in row.items()
                       if isinstance(value, float) and not math.isfinite(value)}
            if invalid:
                row["nonfinite_measurements"] = invalid
                for key in invalid:
                    row[key] = None
            if name not in _OPTIONAL:
                report["blockers"].append(f"{name}: {row['error']}")
        row["total_seconds"] = time.perf_counter() - start
        report["operations"][name] = row
    report["memory_after"] = _memory(torch, device == "mps")
    if device == "mps":
        for name, value in report["memory_after"].items():
            if value["status"] != "pass":
                report["blockers"].append(f"MPS working-set API {name}: {value['status']}")
    report["status"] = "negative" if report["blockers"] else "pass"
    return report


def probe_operations(device: str) -> GateResult:
    """Launch fresh interpreter so fallback is disabled before torch import."""
    if device not in ("cpu", "mps"):
        raise ValueError("device must be cpu or mps")
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0")
    package_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    argv = [sys.executable, "-m", "ralup.environment", "--worker", device]
    try:
        result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=180)
        report = json.loads(result.stdout) if result.returncode == 0 else {
            "status": "blocked", "blockers": ["Probe subprocess failed"], "operations": {}}
        report["process"] = {"argv": argv, "returncode": result.returncode, "stderr": result.stderr}
        if result.returncode != 0:
            report["process"]["stdout"] = result.stdout
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        report = {"status": "blocked", "blockers": [str(error)], "operations": {}}
    path = Path("artifacts/environment") / f"operations-{device}-{time.time_ns()}.json"
    evidence = _write_report(path, report)
    return GateResult(report["status"], (evidence,), tuple(report["blockers"]))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(canonical_bytes(_probe_document(sys.argv[2])).decode())
    else:
        raise SystemExit("Internal probe worker: --worker cpu|mps")
