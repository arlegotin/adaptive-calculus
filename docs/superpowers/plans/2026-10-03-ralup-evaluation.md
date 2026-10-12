# RALUP Controlled Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn validated implementations into a fair, locked, reproducible assessment with honest uncertainty and cost.

**Architecture:** Development search and power planning produce immutable artifacts consumed by a manifest-gated confirmation runner. One scorer/analysis pipeline preserves every arm and failure; reporting derives from raw records and decoded packed models.

**Tech Stack:** Common RALUP engine/artifact store, SciPy t/noncentral-t/chi-square distributions, NumPy, standard plotting tools for exportable figures, pytest and a safe packed tensor container.

**Spec:** [S01–S06, S22, S26, S33–S39](../specs/2026-10-03-ralup-design.md). [Master plan](2026-10-03-ralup.md). Requires F/H/M artifacts appropriate to the stated claim.

## Global Constraints

- Official validation is development; official test stays protected until the locked manifest and all planned final checkpoints exist.
- Two-sided 97.5% paired intervals for each of two co-primary comparisons; both upper bounds must be below zero for H1+H2.
- A fresh complete target-training seed is one independent unit.
- Finite-budget divergence is a substantive failure, never silently replaced.
- No percentage improvement or laptop runtime is presumed.
- Every additional query/gradient/Hessian pass and processed target is recorded; include the common warm start.

## Review Focus

- An edge optimum or slower-converging baseline must not lose by premature search termination — E01.
- Zero/noisy variance and heavy tails must not produce a spurious affordable confirmation budget — E02.
- Tampered artifacts, missing arms and post-freeze edits must deny protected scoring — E03.
- Divergences, repeated scoring and repeated timing observations must not inflate run-level evidence — E04.
- A seemingly small export or successful report must decode correctly and reproduce from raw evidence — E06/E07.

---

## E01 — Fair finalist tournament and independently selected B*

**Files:** Extend `src/ralup/search.py`; create `src/ralup/analysis/{__init__,cost}.py`, `tests/unit/test_finalist_selection.py`, `tests/unit/test_cost_ledgers.py`; extend CLI.

**Interfaces:** Consume M07 `TrialSpec`, M11 artifacts and registry parity. Produce `allocate_search(families: list[dict], allowance_seconds: float, evidence: list[dict]) -> list[TrialSpec]`, `select_finalists(results: list[dict], eligibility: dict) -> ArtifactRef`, and `cost_quality_frontier(runs: list[dict], ledger: str) -> dict`. B* selection record includes all alternatives, full-budget fresh development seeds, source/fidelity status, mean BPB and tie cost.

- [ ] Create fixture curves where a slow method crosses later, a method wins at an LR boundary, and a family is blocked rather than defeated. Assert equal initial synchronized allowance, documented justified extensions, no candidate-only generic improvement, B* allowed to differ from B0 and exact ties resolved by end-to-end cost.

  ```python
  assert selection["B_star"] == strongest_full_budget_development_method
  assert blocked_method not in selection["defeated_methods"]
  assert not selection["uses_protected_scores"]
  assert extra_pass_method.total_targets > extra_pass_method.stream_targets
  ```

- [ ] Run tests; detect fixed trial-count fairness, free-query accounting and requiring B*==B0 as invalid.
- [ ] Complete S26 tournament over all locally competitive required families. Bracket LR/betas/decay/schedules and extend edge/uncertain/unsettled cases. Retune candidate/LR/static and share generic improvements. Include data/compute-augmented ordinary training and full-budget fresh development replication. Produce best BPB versus cumulative seconds/targets, matched stream/total/wall-time curves and claim scope narrowed for unresolved competitors. Never infer winning against an unrun method.
- [ ] Run tests and `python -m ralup select-finalists --config PATH`; expect immutable B*/candidate/LR configs, selection rationale, all failed/losing trials and complete ledger curves. No stable development advantage may close the research with a supported negative/inconclusive development report.
- [ ] Commit as `feat: add fair finalist selection and full cost accounting`.

## E02 — Fresh paired pilot and precision/cost design

**Files:** Create `src/ralup/analysis/power.py`, `tests/unit/test_power_design.py`; extend CLI; write pilot artifacts separately from confirmation.

**Interfaces:** Produce `paired_sd_upper(deltas: np.ndarray, coverage: float) -> float`, `confirmation_n(sd_upper: float, delta_design: float, power: float = .95, interval_confidence: float = .975) -> int`, and `plan_confirmation(pilot_runs: list[ArtifactRef], costs: dict, precision_choices: list[float], assumptions: dict) -> ArtifactRef`.

- [ ] Test SD upper bounds against direct chi-square formulas and minimum integer n against noncentral-t power. Both contrasts use .025 total interval tail probability and per-comparison power .95; choose max n. Check n increases with SD/tighter detectable difference. Degenerate all-zero/few-point variance returns an inadequate-pilot flag instead of claiming near-zero sample need.

  ```python
  assert selected_n == max(n_vs_baseline, n_vs_learned_lr)
  assert computed_power(selected_n) >= .95
  assert selected_n == 2 or computed_power(selected_n - 1) < .95
  assert confirmation_n(sd, effect/2) >= confirmation_n(sd, effect)
  ```

- [ ] Run tests; ensure using z(.975) instead of z(.9875) fails the normal approximation fixture.
- [ ] Implement S33 with four fresh paired development runs as initial variance probe, configurable recorded upper-bound coverage and stability diagnostics, noncentral-t finalization, detectable-effect/CI-width/cost curves and heavy-tail review. Pilot seeds are disjoint from later confirmation seeds. Record resource envelope, target precision and decision rationale before locking n; no arbitrary success percentage. Inadequate/unaffordable precision is a reportable gate outcome, not a reason to inspect protected scores or add seeds adaptively.
- [ ] Run tests and `python -m ralup plan-confirmation --pilot PATH --resources PATH`; expect forecast curves and a fixed design or explicit insufficient-pilot/resource disposition. Bootstrap/sign/rank sensitivities and failure protocol are specified before freeze.
- [ ] Commit as `feat: add pilot-based confirmation power and cost design`.

## E03 — Frozen manifest, artifact identities and one-way test gateway

**Files:** Extend `src/ralup/{config,evaluate,records}.py`; create `tests/unit/test_confirm_manifest.py`, `tests/integration/test_protected_access.py`; extend CLI.

**Interfaces:** Produce `ConfirmationManifest`, `freeze_confirmation(inputs: dict) -> ArtifactRef`, `validate_confirmation(reference: ArtifactRef, inventory: list[dict]) -> ConfirmationManifest`, and `open_protected_targets(*, phase: str, manifest_ref: ArtifactRef, inventory: list[dict]) -> Iterable[TargetBlock]`. Data H01/H02 audit reads stay a separate non-scoring interface. Scoring gateway accepts complete terminal inventory, not a caller-supplied boolean claiming readiness.

- [ ] Assert phase=confirm and exact hashes required; altered data/policy/normalizer/model/config/analysis code, incomplete arm inventory, duplicate/missing seeds and changed endpoints fail. Fixture divergence status may be terminal but does not fabricate a final checkpoint. Testing code must never import protected scores during training/search.

  ```python
  with pytest.raises(ProtectedAccessError): open_protected_targets(phase="explore", manifest_ref=ref, inventory=complete)
  with pytest.raises(IntegrityError): validate_confirmation(tampered_ref, complete)
  with pytest.raises(IncompleteConfirmation): validate_confirmation(ref, missing_one_arm)
  assert validated_manifest.sha256 == original_manifest.sha256
  ```

- [ ] Run tests; demonstrate a stale analysis hash or last missing arm denies access before any target payload read.
- [ ] Implement all S34 fields, canonical hashing and artifact validation, immutable seed/arm budgets, no protected training evaluation and append-only audit events. Enable one locked scorer only after every planned arm has valid terminal status. Record amendments after freeze; protected inspection marks changed claims exploratory. Clarify procedural rather than OS-security enforcement in README/help.
- [ ] Run tests and `python -m ralup plan-confirmation --freeze --design PATH`; expect hashed `artifacts/confirm_manifest.json` plus all resolvable dependencies, or refusal listing missing evidence. No real test scoring during this verification; use synthetic protected fixtures.
- [ ] Commit as `feat: gate protected scoring on frozen complete confirmation`.

## E04 — Paired confirmation execution and simultaneous inference

**Files:** Create `src/ralup/runner.py`, `src/ralup/analysis/confirm.py`, `tests/integration/test_confirmation_runner.py`, `tests/unit/test_paired_inference.py`; extend CLI.

**Interfaces:** Produce `PairedResults`, `run_confirmation(manifest: ConfirmationManifest) -> list[ArtifactRef]`, `score_confirmation(manifest: ConfirmationManifest, inventory: list[dict]) -> ArtifactRef`, and `analyze_confirmation(results: PairedResults, manifest: ConfirmationManifest) -> dict`. Runner uses named streams and randomized frozen arm order; one warm start per paired seed is shared bit for bit.

- [ ] Use synthetic paired fixtures to check both exact t(.9875) intervals, delta sign, ratio/interval transform, one-versus-two passing contrasts, constant differences, missing pairs and nonfinite outcomes. Simulate interruption/resume, substantive divergence, implementation invalidation and duplicate scoring. Timing repetitions cannot change statistical n.

  ```python
  assert result["n"] == len(independent_seed_ids)
  assert result["H1_H2_supported"] == (upper_B < 0 and upper_LR < 0)
  assert result["ratio_vs_B"] == pytest.approx(2 ** mean_delta_B)
  assert divergent_result["all_run_bpb_claim_established"] is False
  assert second_score_request.result_hash == first_score_request.result_hash
  ```

- [ ] Run task tests; prove dropping divergent seeds or pooling all chunk losses fails independent-unit/failure rules.
- [ ] Implement S01–S06/S34 fixed endpoint execution, complete costs/terminal inventory, delayed scoring and append-only results. Analyze both t intervals, marginal/per-seed BPB and signs, failure probability/successful quality plus ordinal sensitivity. Predeclared bootstrap/sign/rank summaries remain sensitivities. Invalidate every bug-affected arm without deleting originals; do not replace divergent seeds or continue until significance. Account for actual shared warm-start cost in project ledger and include an attributable warm start in each deployment cost comparison without double-counting project work.
- [ ] Run tests, then `python -m ralup confirm --manifest artifacts/confirm_manifest.json` followed by `python -m ralup analyze-confirmation --manifest artifacts/confirm_manifest.json`. Expect all-run status, both intervals, quality/cost curves and scoped conclusion; synthetic integration tests must exercise success, null, inconclusive and divergence paths before real protected use.
- [ ] Commit as `feat: add locked paired confirmation and honest failure inference`; mark WP08 with evidence, not a presumed positive finding.

## E05 — Attribution contrasts and explicit extension gates

**Files:** Create `src/ralup/analysis/attribution.py`, `configs/ablations/{actions,information,scales,noise,budgets}.json`, `tests/unit/test_ablation_contracts.py`; extend registry/CLI.

**Interfaces:** Produce `make_ablation_trials(frozen_parent: ArtifactRef, contrast: str, phase: str, retune: bool) -> list[TrialSpec]`, `analyze_attribution(runs: list[ArtifactRef], declared_family: dict) -> dict`, and `check_extension_eligibility(evidence: dict, extension: str) -> GateResult`.

- [ ] Assert shared seed/data/model/budget/scorer across paired contrasts, retraining and retuning status explicit, history shuffle restricted to predeclared matched-shape permutation, no-history models still charged retained feature scans, privileged norm matching labeled, and post-test selection remains exploratory. Unsupported FP8/MXFP4/W4A4/new-corpus configurations must require their new contract rather than inherit an FP4 label.

  ```python
  assert candidate_trial.stream_hash == ablation_trial.stream_hash
  assert history_removed_trial.feature_cost_retained
  assert norm_matched_trial.is_privileged_diagnostic
  assert selected_after_test.claim_status == "exploratory"
  assert undeclared_fp8_gate.status == "blocked"
  ```

- [ ] Run task tests; verify inference-only scrambling cannot be reported as the retrained information ablation.
- [ ] Implement S37 contrasts: learned-LR wider/shared range, static/single actions, retrained alpha-zero/history/cell removals, fixed/dynamic/learned scales and rotation, random matched-rate assistance/noise/damping, data/compute-augmented ordinary baseline, and per-article/entropy/length regressions. Add magnitude-preconditioned additive control as the first-order normalized z from S17, with independently tuned LR and an explicit variant name. Paired exploratory comparisons inform priority; mechanism confirmation uses a declared family/multiplicity rule and fresh design. Gate larger shape, independent meta-training clusters, from-scratch and carry-state branches on evidence/budgets; exact FP8/MXFP4/W4A4/new-corpus contracts are separate future specifications.
- [ ] Run tests and `python -m ralup run --phase explore --suite attribution --config PATH`; expect contrasts labeled intervention versus retuned competition, favorable/unfavorable outcomes and charged costs. No requirement to exhaust the full cross-product or extend blindly after realistic failure.
- [ ] Commit as `feat: add mechanism attribution and evidence-triggered extensions`.

## E06 — Packed E2M1 export and decoded-forward parity

**Files:** Create `src/ralup/export.py`, `tests/unit/test_packed_export.py`, `tests/integration/test_export_forward.py`; extend CLI.

**Interfaces:** Produce `pack_e2m1_codes(codes: Tensor) -> bytes`, `unpack_e2m1_codes(payload: bytes, count: int) -> Tensor`, `export_model(model, context: dict, destination: Path) -> ArtifactRef`, and `load_export(reference: ArtifactRef) -> object` exposing the common inference forward. Metadata includes shapes, exact scales, rotation, encoding order, high-precision auxiliaries, padding count and checksums.

- [ ] Test every code nibble including positive-zero canonicalization, odd element count padding, corrupt checksum, missing scales and rotated matrices. Decode final deterministic quantized contexts and compare logits/BPB with H06 reference. Verify analytical payload ledger separately from actual container bytes.

  ```python
  assert torch.equal(unpack_e2m1_codes(pack_e2m1_codes(codes), len(codes)), codes)
  assert len(pack_e2m1_codes(codes)) == (len(codes) + 1) // 2
  assert core_payload_bytes == 1572864 + 806912 + 36864 == 2416640
  torch.testing.assert_close(decoded_logits, final_quantized_logits)
  ```

- [ ] Run task tests; detect float32 fake-quantized storage being mislabeled as packed FP4 and zero-encoding mismatch.
- [ ] Implement S35 nibbles/scales/auxiliaries with a safe versioned container, metadata/checksums, deterministic scale finalization and portable decoder. Report quantized fraction and actual file bytes; compressed storage never implies native execution speed. Persist source checkpoint/context hash and parity evidence in export metadata.
- [ ] Run tests and `python -m ralup export --checkpoint PATH --output PATH`; expect exact codes, loss/logit parity within the frozen numerical contract and measured container size. Re-run final artifact parity after any representation/grouping changes.
- [ ] Commit as `feat: add packed model export with decoded-forward verification`.

## E07 — Reproduction, complete report and research disposition

**Files:** Create `src/ralup/report.py`, `reproduce.sh`, `tests/integration/test_report_reproduction.py`; modify README/CLI; generate report/figures from artifact records.

**Interfaces:** Produce `reproduce(manifest: ArtifactRef, environment: dict) -> GateResult`, `build_report(artifacts: list[ArtifactRef], destination: Path) -> ArtifactRef`, and `research_disposition(evidence: dict) -> dict`. Report accepts exploratory-only terminal evidence when no confirmation was justified; it must not fabricate a manifest/test result in that case.

- [ ] Use synthetic raw ledgers to regenerate known tables deterministically; omit a failed trial, alter source hashes or remove a required cost and require an incomplete-evidence finding. Assert null/inconclusive/blocked/positive-quality-but-slower reports use distinct conclusions. Meta-recipe claims without independent meta-run clusters are rejected.

  ```python
  assert tables_from_raw_hash == tables_from_second_build_hash
  assert negative_fixture_report.decision == "local_non_advantage"
  assert slower_positive_fixture_report.efficiency_improvement is False
  assert no_confirmation_report.claim_status == "development_only"
  assert break_even_reuses is None if candidate_cost >= baseline_cost else break_even_reuses > 0
  ```

- [ ] Run report integration tests; show dropping failures or using stream-only compute for efficiency fails report completeness.
- [ ] Implement S39 tables/plots from raw records, every required report section, useful per-seed details, source deviations and unresolved R41/QuEST/fidelity limitations. Label inherited Linux prototype claims as supplied historical reports, and new CPU/M3 measurements by provenance. Generate cost/quality and precision curves with standard plotting tools as standalone artifacts. Calculate break-even only at comparable quality with positive saving; separately include method-specific tuning. Write a shell entry point with explicit immutable inputs and no hidden manual steps or untrusted pickle loads.
- [ ] Run tests, recreate the successful locked environment separately, replay selected numerical fixtures and one paired trajectory, test interruption/resume, and regenerate outputs from raw data. Use `python -m ralup reproduce --manifest artifacts/confirm_manifest.json` for confirmed studies; for earlier dispositions use `python -m ralup reproduce --evidence PATH`. Run `python -m ralup report --evidence PATH`. Expect a runnable handoff or explicit reopened gate for unexplained discrepancies.
- [ ] Commit as `feat: add reproducible research report and outcome ledger`; update README/current gate, decisions and progress. Do not publish/push or start expensive extensions merely because report generation completed.

## Stage exit and completion definition

A complete research handoff includes runnable source, successful dependency lock, rights/provenance records, numerical/fidelity checks, complete raw development/failure/cost logs, frozen analysis if confirmation was performed, packed-export parity when a final model exists, and a reproducible scope-limited decision. Independent full meta-training runs are necessary before claiming the training recipe is reliable. Lack of novelty, no advantage, conditional utility, inadequate sensitivity and local resource blocks are all valid final decisions when supported by the evidence; missing checks are never relabeled as passing.

## Execution status —12 October2026

The drafting checklist above is retained as history. The subsequently authorized execution completed31 predecessor engineering tasks; E07 implements the report/reproduction handoff and remains subject to its fresh task review, followed by root's one whole-branch review. Current scientific WP04 remains inconclusive/B0 null; WP06 NEEDS_REVISION; final C/LR/B*, budgets, n and confirmation are absent. See [current progress](../../../progress.json), [E07 contract](../../harness/e07-contract.md) and [immutable evidence index](../../harness/e07-evidence.json). No confirmation or scientific advantage is inferred from engineering completion.
