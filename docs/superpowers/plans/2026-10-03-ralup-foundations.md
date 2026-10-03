# RALUP Numerical Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the executable environment/numerical gates before neural training.

**Architecture:** A small CLI calls pure configuration, oracle, quantizer and action modules. CPU float64 defines expected behavior; device implementations must satisfy the same contract with separately frozen tolerances.

**Tech Stack:** Python, NumPy, PyTorch, SciPy, pytest, standard-library records and hashing; versions resolved on the actual machine.

**Spec:** [S07, S13–S18, S27–S32, S36](../specs/2026-10-03-ralup-design.md). [Master plan/type vocabulary](2026-10-03-ralup.md).

## Global Constraints

- Quantize internal projections only; all other parameters, arithmetic, gradients, shadows and optimizer state remain FP32.
- Moments remain in original W coordinates.
- At tau exactly one bypass assistance.
- Official validation is development; official test stays protected until the locked manifest and all planned final checkpoints exist.
- Runtime versions and tolerances come from measured gates; no invented compatible release or passing MPS result.

## Review Focus

- Malformed/nonfinite JSON cannot become a silently hashed run configuration — F01.
- Rosetta, unavailable MPS, implicit CPU fallback and leaking hardware serials must be detected — F02.
- Negative midpoint ties and signed zero require encoding-aware rounding — F03.
- Zero groups, scale refresh and microbatch graph reuse must not change quantizer semantics — F04.
- Tiny FP32 nudges and endpoints must fail closed while preserving shadow accumulation — F05.

---

## F01 — Validated package configuration and artifact records

**Files:** Create `pyproject.toml`, `.gitignore`, `src/ralup/{__init__,__main__,config,records}.py`, `tests/unit/test_config_records.py`, `configs/{smoke,proxy,core,meta}.json`, `decisions.md`, `progress.json`; modify `README.md` with real commands only.

**Interfaces:** Produce `resolve_config(template: dict, evidence: dict, *, purpose: str) -> ResolvedConfig`, `canonical_bytes(value: dict) -> bytes`, `sha256_file(path: Path) -> str`, `ArtifactRef(path, sha256)`, `GateResult(status, evidence, blockers)`, and `RunLedger.record(work: dict) -> None`. `ResolvedConfig` exposes `.payload` and `.sha256`. Ledger work has stream, unique, additional_loss, backward, query target counts plus elapsed seconds; categories can overlap and must not be blindly summed.

- [ ] Write `test_config_records.py` with reordered-key hash equality, nonfinite rejection, missing evidence blocking, and immutable run metadata. Minimal assertions:

  ```python
  assert canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
  with pytest.raises(ValueError): canonical_bytes({"loss": float("nan")})
  with pytest.raises(UnresolvedConfig): resolve_config(core_template, {}, purpose="train")
  ```

- [ ] Run `python -m pytest tests/unit/test_config_records.py -q`; establish failure from the missing package/API, then specific validation failures before completing validation.
- [ ] Implement standard-library canonicalization/config schemas and append-only ledger, phase-specific evidence validation, and a CLI parser exposing help/version. Templates carry spec constants and deliberate unresolved fields. Fold package build/test configuration and ignore rules into this deliverable. Create an explicit synthetic smoke config with tiny finite step budget; reject its use for confirm. Record the spec's engineering resolutions in decisions.
- [ ] Run the focused test and `python -m ralup --help`; expect exit zero, valid subcommand help and no network/experiment side effects. JSON-parse templates and progress.
- [ ] Commit these explicit source/config/test/docs paths as `feat: add validated research configuration and records`; record F01 in progress.

## F02 — Hardware and supported-operation evidence

**Files:** Create `src/ralup/environment.py`, `tests/unit/test_environment.py`, `tests/integration/test_mps_operations.py`, `licenses/ASSETS.md`, `vendor_manifest.json`; modify CLI; produce `requirements.lock` and `artifacts/environment/` during execution.

**Interfaces:** Consume F01 records. Produce `discover_environment() -> dict`, `probe_operations(device: str) -> GateResult`, `write_environment_report(path: Path) -> ArtifactRef`. Expose `python -m ralup doctor --output PATH` and separate `--probe mps` mode. Importing the package must work on CPU-only hosts.

- [ ] Add tests for unavailable MPS producing a blocked MPS gate, missing optional APIs being recorded, serial redaction, and explicit fallback detection. Device tests probe all S27 operations and save loss/gradient/max-error/timing fields; unavailable MPS skips the test with a **blocked** report.

  ```python
  assert unavailable_mps_report.status == "blocked"
  assert serial not in json.dumps(redacted_hardware_report)
  assert {"matmul_backward", "expm1", "rng_restore"} <= report["operations"].keys()
  ```

- [ ] Run `python -m pytest tests/unit/test_environment.py -q`; verify failures exercise blocked/redaction branches.
- [ ] Implement read-only discovery commands from S27, architecture/Rosetta and API detection, isolated stable dependency resolution and wheel/license provenance. Capture failed environments too. Probe with silent fallback disabled; report QR/eigh/bucketize/functional-gradient support individually. Do not turn deliberate CPU subroutines into unreported success. Choose a safe non-executable tensor container for H07 after evaluating installed support; record its dependency/hash if external.
- [ ] Run unit tests and `python -m ralup doctor --output artifacts/environment/environment.json`; run device tests on the laptop. Pass requires supported core FP32 operations and documented tolerances/working-set APIs; optional unsupported operations remain named for later adapters. No network-bound package install is presumed authorized outside a later implementation request.
- [ ] Commit environment code/tests, successful lock and license records as `feat: record hardware and numerical backend capabilities`; retain reports as hashed local artifacts.

## F03 — Exact E2M1 oracle and portable tensor quantizer

**Files:** Create `src/ralup/quantization/{__init__,formats,oracle}.py`, `tests/unit/test_e2m1.py`, `tests/integration/test_e2m1_mps.py`; extend CLI numerical dispatch.

**Interfaces:** Produce `e2m1_oracle(x: np.ndarray, scale: np.ndarray) -> np.ndarray`, `quantize_e2m1(x: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]` returning values and uint8 code indices, and `directional_neighbor(codes: Tensor, direction: Tensor) -> tuple[Tensor, Tensor]` returning neighbor indices/feasibility. Tables expose unique values, actual nibble encodings and parity independently. Scales must be finite and positive.

- [ ] Write exhaustive 15-level/14-tie tests, signed zero, broadcasts, nearest representable values either side of every tie, finite saturation, NaN/Inf and nonpositive-scale rejection. Verify `.25→0`, `.75→1`, `1.25→1`, `1.75→2`, and their negative mirrors.

  ```python
  np.testing.assert_array_equal(e2m1_oracle(np.array([.25, .75, 1.25, 1.75]), np.array(1.)), [0., 1., 1., 2.])
  assert not np.signbit(e2m1_oracle(np.array([-0.]), np.array(1.)))[0]
  assert directional_neighbor(endpoint_codes, outward_directions)[1].sum() == 0
  ```

- [ ] Run `python -m pytest tests/unit/test_e2m1.py -q`; demonstrate a naive sorted-index tie rule fails a negative midpoint case.
- [ ] Implement all-distance float64 oracle and device thresholds/verified bucketize without `[P,15]` storage. Explicitly separate ordinal neighbor indices from exported encoding nibbles. Round in scaled coordinates consistently; document FP32 versus exact-scale behavior through tests.
- [ ] Run CPU and MPS test files; expect exhaustive fixed-point/tie agreement in supported exact cases, bounded numerical errors elsewhere, and no CPU fallback in production quantization. Persist oracle/device discrepancies and tolerance choices.
- [ ] Commit as `feat: add encoding-aware E2M1 oracle and device quantizer`.

## F04 — Scales, rotations, detached contexts and surrogate graphs

**Files:** Create `src/ralup/quantization/{scales,rotation,surrogates,context}.py`, `tests/unit/test_quantization_context.py`, `tests/unit/test_gaussian_fit.py`, `tests/integration/test_surrogate_microbatches.py`.

**Interfaces:** Consume F03. Produce `hadamard(x: Tensor) -> Tensor`, `fit_gaussian_e2m1() -> dict`, `compute_scales(u: Tensor, recipe: str, previous: Tensor | None) -> Tensor`, `prepare_context(w: Tensor, recipe: dict, step: int, rng: object) -> QuantizationContext`, and `quantized_view(w: Tensor, context: QuantizationContext) -> Tensor`. Context includes S07/S15 fields; tensors are detached. Storage/original transforms take an explicit rotation ID.

- [ ] Test Hadamard involution/gradient, zero rows, fixed versus dynamic scales, deterministic Gaussian integration versus independently integrated squared error, STE identity and trust mask. Backward through two separately constructed microbatch views using one cached context must succeed without retained graphs; context tensors require no grad.

  ```python
  torch.testing.assert_close(hadamard(hadamard(x)), x)
  assert all(not t.requires_grad for t in context.numeric_tensors())
  assert fit_a["scale_constant"] == fit_b["scale_constant"]
  torch.testing.assert_close(ste_grad, upstream_gradient)
  ```

- [ ] Run the three task test files; observe failures for graph reuse and wrong inverse normalization.
- [ ] Implement orthonormal butterfly, S14 analytic moments/bracketed log-scale minimization, positive sentinel for zero groups, explicit fixed/dynamic/group recipes, and fresh STE/trust graph construction over cached numerical values. Context creation does not mutate optimizer history. Reserve stochastic realization injection for M05; one supplied code realization can be reused across microbatches.
- [ ] Run tests on CPU/MPS and `python -m ralup check-numerics --suite quantizer`; expect fitted constant/objective/residual artifact, all involution/backward/cache checks passing, and grouping labels in reports.
- [ ] Commit as `feat: add quantization contexts scales and surrogate rules`.

## F05 — Safe geometry, assistance and discriminating tiny objectives

**Files:** Create `src/ralup/policy/{__init__,actions}.py`, `src/ralup/diagnostics/{__init__,quadratics}.py`, `tests/unit/test_actions.py`, `tests/unit/test_quadratic_diagnostics.py`; extend numerical CLI.

**Interfaces:** Produce `multiplicative_proposal(u0: Tensor, da: Tensor) -> Tensor`, `assist_boundaries(u0: Tensor, d: Tensor, tau: float, context: QuantizationContext) -> tuple[Tensor, dict]`, `compose_action(u0: Tensor, da: Tensor, alpha: float, h: float, tau: float, context: QuantizationContext) -> tuple[Tensor, dict]`, and `run_quadratic_diagnostics(seed: int, horizons: list[int]) -> dict`.

- [ ] Write sign/zero/exponent-cap, identity, decay-reference separation, neighbor/amplification, saturation/margin rejection and dynamic-scale tests. Include 100,000 deterministic random coordinates at tau .125,.25,.5,.75,1 and all boundary ties. Scalar scale-one case must reproduce helpful and harmful forced crossings.

  ```python
  assert torch.equal(assist_boundaries(u, d, 1., ctx)[0], u + d)
  assert float(assist_boundaries(tensor([1.]), tensor([.04]), .125, ctx)[0]) == pytest.approx(1.2500390625)
  assert report["target_1_1_after"] > report["target_1_1_before"]
  assert report["target_1_8_after"] < report["target_1_8_before"]
  ```

- [ ] Run the two task test files; show naive negative-weight multiplication and unconditional boundary nudging fail.
- [ ] Implement S17–S18 exactly, including no post-exponential/post-assistance normalization and explicit accepted/rejected masks. Define float32 tolerance from F02/F03 evidence and report excess instead of hiding it. Add q/exponential, finite-q quadratic, FP32-nextafter, persistent-shadow, stochastic-expectation and scale-homogeneity diagnostics. Coupled fixtures enumerate one immediate {-1,0,+1} action per coordinate then K common baseline steps for dimensions 2/4/8 and condition numbers 1/10/100; record oracle privilege/cost.
- [ ] Run focused tests and `python -m ralup check-numerics --output artifacts/checks/results.json`; expect all invariants, both favorable/unfavorable crossing controls, seeded coupled trajectories, environment and source hashes. A small FP32 rejected nudge is valid evidence, not a test failure.
- [ ] Commit as `feat: add bounded update geometry and mechanism diagnostics`; close WP01 only after all invariants pass.

## Stage exit

F01–F05 deliver WP00–01 evidence and a working numerical CLI. If a numerical invariant fails, later neural work is blocked. If no faithful MPS path exists, retain CPU diagnostics and report the scope limit. No core runtime forecast or neural improvement is inferred from this stage.
