# RALUP Baselines and Learned Policies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish strong local comparators, a correct shared controller, and an informative future-loss meta-training procedure.

**Architecture:** All methods implement H05's common hooks and extend complete snapshots explicitly. Original-format fixtures precede named workload adaptations. B0 is chosen before final policy learning; final B* is chosen independently in E01.

**Tech Stack:** Locked PyTorch/NumPy/SciPy stack, permitted pinned reference implementations, common H06/H07 engine and records.

**Spec:** [S16–S26, S28–S32, S36–S37](../specs/2026-10-03-ralup-design.md). [Master plan](2026-10-03-ralup.md). Requires F/H deliverables; M07 specifically depends on M01/M02.

## Global Constraints

- Moments remain in original W coordinates.
- The deployed policy never receives query or protected outcomes.
- At tau exactly one bypass assistance.
- One affine map from 14 scalars plus intercept to three outputs: exactly 45 learned parameters, no matrix ID.
- Learned LR sees all 14 features/history/meta-data and rollout opportunities; network `14→3→1`, tanh hidden, 49 trainable parameters.
- A licensing/operation/parity block is not a defeated baseline.

## Review Focus

- Rectangular shapes and auxiliary groups must not silently change optimizer normalization/order — M01/M02.
- Accumulation, zero signals and moving scales must not manufacture transition feedback or routing probability — M03–M06.
- Missing/invalid history and SOAP basis transforms must yield the declared features and coverage — M08.
- An initialized policy must preserve the actual backbone trajectory, including decay and Parameter identity — M09.
- Fast-weight cancellation or partial ES cloning must not erase meta-gradients or spoil matched randomness — M10/M11.

---

## M01 — Muon and Muon+ with explicit geometry variants

**Files:** Create `src/ralup/optim/{muon,muon_plus}.py`, `tests/unit/test_muon_variants.py`, `tests/integration/test_matrix_optimizer_parity.py`; update registry/vendor manifest.

**Interfaces:** Implement H05 `BaseUpdateRule` and expose `orthogonalize_five_step(matrix: Tensor) -> Tensor`, `MuonUpdateRule(config: dict)`, `MuonPlusUpdateRule(config: dict)`. Moment access returns original-coordinate state; auxiliary parameters use separate configured AdamW groups.

- [ ] Write fixed-gradient fixtures for square/tall/wide/zero matrices and all four Muon+ normalization orders. Pin a CPU float64 literal polynomial fixture independently from the production function. Compare normalization-on/off under identical momentum, separately from the Nesterov variant.

  ```python
  torch.testing.assert_close(actual_polynomial, literal_five_step_reference)
  torch.testing.assert_close(muon_displacement, -lr * math.sqrt(max(1, rows/cols)) * polar)
  torch.testing.assert_close(muon_plus_displacement, -lr * math.sqrt(rows/cols) * normalized_polar)
  ```

- [ ] Run task tests; verify swapped rectangular multipliers and shared momentum conventions fail.
- [ ] Implement S23 exact coefficients, norm stabilizer, momentum conventions, serial normalization epsilon/order, registry names and original-coordinate proposals. Pin primary source revisions and rights; document numerical dtype and all departures. Keep initialization/decay/auxiliary grouping explicit.
- [ ] Run CPU/MPS fixtures and `python -m ralup verify-baselines --suite muon`; expect tolerance-compliant polynomial and one-step output with variant metadata. Add short quantized proxy trajectories without selecting winners yet.
- [ ] Commit as `feat: add declared Muon and Muon+ optimizer variants`.

## M02 — SOAP fidelity and sign-restricted Madam

**Files:** Create `src/ralup/optim/{soap,madam}.py`, `tests/unit/test_soap_adapter.py`, `tests/unit/test_madam_adapter.py`, `tests/integration/test_optimizer_state_cost.py`; update source/license ledger.

**Interfaces:** Implement `SoapUpdateRule(config: dict)` and `MadamUpdateRule(config: dict)` using H05. SOAP additionally exposes `first_moment_in_original_coordinates(state: dict) -> Tensor` and `state_inventory(state: dict) -> dict` with factor/basis/transfer bytes and timing.

- [ ] Use a pinned permitted SOAP reference to test first-step initialization, eig/QR refresh, basis-change moments, bias correction and decay order on a fixed sequence; test Madam negative/zero weights, fixed initial bound, 0/0, clipping and no sign changes. Fixtures assert first moments map back to W correctly.

  ```python
  torch.testing.assert_close(adapter_soap_weights, reference_soap_weights)
  assert torch.equal(madam_weights[initial_weights == 0], torch.zeros_like(madam_weights[initial_weights == 0]))
  assert bool((madam_weights.abs() <= initial_bound).all())
  assert inventory["tensor_bytes"] == sum(t.numel()*t.element_size() for t in actual_state_tensors)
  ```

- [ ] Run task files; demonstrate skipped SOAP initialization and hidden Madam additive escape fail.
- [ ] Adapt the pinned MIT SOAP source without altering preconditioning; explicitly time CPU factorization/transfers when MPS support is missing. Implement Madam equations cleanly or comply with its verified license; no indiscriminate vendoring. Store complete factors/bases in H07 snapshots and source fixture status in registry. Screen SOAP refresh 1/4/10 later, not by hard-coding a winner.
- [ ] Run tests and `verify-baselines --suite soap-madam`; expect exact reference mechanics or an explicit fidelity block. A SOAP operation block may leave Muon available as the required strong matrix optimizer, but SOAP is not reported defeated.
- [ ] Commit as `feat: add faithful SOAP adapter and declared Madam implementation`.

## M03 — EWGS original fixture and FP4 backward adaptation

**Files:** Create `src/ralup/baselines/ewgs.py`, `tests/unit/test_ewgs.py`, `tests/integration/test_ewgs_hvp.py`.

**Interfaces:** Produce `ewgs_backward(upstream: Tensor, normalized_shadow: Tensor, normalized_code: Tensor, delta: float, clipping_mask: Tensor) -> Tensor`, `estimate_delta(loss_fn: Callable, quantized_leaf: Tensor, probes: Tensor) -> dict`, and `EwgsAdapter(config: dict)` using H05 hooks.

- [ ] Assert the S24 original uniform formula and FP4 dimensionless residual, positive/negative upstream signs and clipping. For a known quadratic Hessian, Rademacher trace expectation/exhaustive small probes equals its trace; verify derivatives use a detached quantized leaf.

  ```python
  torch.testing.assert_close(actual, g * (1 + delta * torch.sign(g) * (xn-xq)))
  assert estimated_trace == pytest.approx(torch.trace(hessian).item())
  assert adaptive_delta >= 0
  ```

- [ ] Run both test files; show differentiating through hard rounding yields the wrong HVP and fails.
- [ ] Implement fixed delta first, then the exact S24 population-std/trace adaptive estimator with explicit normalization/probe/cadence config. Capture extra Hessian/backward bytes and time. Register `ewgs_uniform_fixture` and `ewgs_fp4` separately; adaptive execution is conditional on pilot relevance and operation support.
- [ ] Run tests and `verify-baselines --suite ewgs`; expect original-rule parity, analytic-Hessian agreement and explicit adaptive-pass cost. Unsupported higher-order MPS behavior is a declared gate, with CPU fixture evidence kept distinct.
- [ ] Commit as `feat: add EWGS fixtures and dimensionless FP4 adapter`.

## M04 — TRS release mechanics and logical-step feedback

**Files:** Create `src/ralup/baselines/trs.py`, `tests/unit/test_trs.py`, `tests/integration/test_trs_accumulation.py`.

**Interfaces:** Produce `TrsState(r: float, K: float, previous_codes: Tensor, observations: int)`, `trs_effective_lr(state: TrsState, observed_rate: float, target_rate: float, alpha: float, mu: float) -> tuple[float, TrsState]`, and `TrsBackboneAdapter(base: BaseUpdateRule, config: dict)`.

- [ ] Pin released code fixture outputs for initialization and subsequent rates, including coupled decay in the fixture. Verify first LR alpha/first-rate ignored, multiplicative feedback, one history update per logical batch and no validation mutation.

  ```python
  assert first_lr == alpha
  assert second_K == pytest.approx(mu * initial_K + (1-mu) * observed_rate)
  assert lr_microbatch_one == lr_microbatch_four
  assert state_hash_before_eval == state_hash_after_eval
  ```

- [ ] Run task tests; demonstrate additive feedback or counting every microbatch fails fixture values.
- [ ] Implement S24 release fixture and separately named backbone adapter with declared decoupled decay. Transition measurement uses normalized quantized values/scales per contract and advances at the logical boundary. Register target-rate schedules separately from ordinary LR schedules; include full scalar/code state in snapshots.
- [ ] Run tests and `verify-baselines --suite trs`; expect source fixture parity plus accumulation invariance. Record the code/paper difference and scale contract in vendor/method metadata.
- [ ] Commit as `feat: add TRS release fixture and logical-step adapter`.

## M05 — QAR endpoint routing and stochastic rounding controls

**Files:** Create `src/ralup/baselines/{qar,rounding}.py`, `tests/unit/test_endpoint_routing.py`, `tests/integration/test_stochastic_context.py`.

**Interfaces:** Produce `routing_probabilities(signal: Tensor, gaps: Tensor, feasible: Tensor, lr: float, gamma: float, group_axes: tuple) -> Tensor`, `stochastic_round(values: Tensor, scales: Tensor, uniforms: Tensor) -> tuple[Tensor, Tensor]`, and `QarAdjacentAdapter(config: dict)`. Explicit uniforms permit reproducible coupling without accidental cross-method RNG consumption.

- [ ] Test gamma zero including zeros, all-zero groups, missing neighbors, saturated probabilities and unequal directional gaps. Test stochastic expectation analytically and by enumerated uniforms, exact levels/endpoints, one realization across microbatches and unchanged RNG under deterministic evaluation.

  ```python
  assert bool((routing_probabilities(zero_signal, gaps, feasible, .1, 0., axes) == 0).all())
  assert probability_at_1_04_between_1_and_1_5 == pytest.approx(.08)
  assert torch.equal(codes_first_microbatch, codes_last_microbatch)
  assert max_code_step <= 1
  ```

- [ ] Run task tests; show persistent shadow state in endpoint QAR or per-microbatch redrawing violates the declared contract.
- [ ] Implement S24 uniform QAR fixture then `qar_fp4_adjacent` with endpoint-only state and persistent optimizer moments. Add separate shadow-forward stochastic and endpoint-only rounding controls. Use cached contexts from F04; primary final evaluation always deterministic. Charge probability computation/extra state and log clipping.
- [ ] Run tests and `verify-baselines --suite routing-rounding`; expect explicit method/state labels, probabilities and seed-replay parity, with no claim that uniform-grid theory applies to FP4.
- [ ] Commit as `feat: add quantized endpoint routing and rounding controls`.

## M06 — Learned scales, damping and masking explanations

**Files:** Create `src/ralup/baselines/{learned_scale,noise}.py`, `tests/unit/test_learned_scale.py`, `tests/unit/test_noise_controls.py`.

**Interfaces:** Produce `learned_scale_view(w: Tensor, log_scale: Tensor, contract: dict) -> Tensor`, `damping_penalty(w: Tensor, quantized: Tensor, coefficient: float) -> Tensor`, and `mask_displacement(displacement: Tensor, moment: Tensor, gradient: Tensor, state: dict, uniform: float, config: dict) -> tuple[Tensor, dict]`.

- [ ] Test uniform LSQ fixture and FP4 scale derivative inside/outside range, log-scale chain rule and `1/sqrt(N*Cmax)` factor. Do not compare an STE derivative with the true discontinuous finite difference. Test noise controls' moments always update; simple masking doubles accepted moves while Magma multiplies alignment without doubling; mask probability stays .5.

  ```python
  torch.testing.assert_close(log_scale_grad, declared_scale_surrogate_grad * scale)
  assert rejected_step_moment_count == accepted_step_moment_count
  torch.testing.assert_close(simple_accepted, 2 * base_displacement)
  torch.testing.assert_close(magma_accepted, alignment * base_displacement)
  ```

- [ ] Run both task files; verify using scale-gradient finite differences as the target, or changing Magma mask probability with alignment, is caught.
- [ ] Implement S24 adapters with explicit gradient paths, independently configurable scale LR, clipping/range/grouping and all state in snapshots. Register damping as project adaptation. Pin source fixtures where applicable and share scale recipe/info/export precision with matching arms.
- [ ] Run tests and `verify-baselines --suite scale-noise`; expect analytic surrogate parity, deterministic masks/state replay and additional-pass/state accounting.
- [ ] Commit as `feat: add learned-scale damping and masking controls`.

## M07 — Ordinary-backbone headroom tournament and B0 decision

**Files:** Create initial `src/ralup/search.py`, `tests/unit/test_headroom_selection.py`; modify calibration/CLI; create development run/config manifests in artifacts.

**Interfaces:** Produce `TrialSpec(config_hash, family, seed, budget, source_revision)`, `run_search_trial(trial: TrialSpec) -> ArtifactRef`, `screen_headroom(trials: list[ArtifactRef]) -> dict`, and `select_backbone(development_results: list[dict], rationale: dict) -> ArtifactRef`. B0 artifact contains exact method/quantizer/grouping/schedule/budget/normalization configuration and selection evidence.

- [ ] Use synthetic curve records to assert unsettled/edge/overlapping contenders advance, a quantizer eliminating headroom is retained, and meta/target warm starts are partition-separated. Selection cannot read a test score or choose a winner solely from a short unstable proxy curve.

  ```python
  assert edge_optimum_trial in screen["expand"]
  assert unsettled_competitor in screen["advance"]
  assert "test_bpb" not in selection_inputs
  assert selection["reference_gap_is_upper_bound"] is False
  ```

- [ ] Run `test_headroom_selection.py`; demonstrate fixed-trial-count/shortest-budget selection fails advancement rules.
- [ ] Implement S12/S26 risk-ordered ordinary optimizer/quantizer screening using H08 calibration. Include FP32 AdamW, STE/clipped STE, QuEST row/tensor grouping, rotation/trust sensitivities and available strong matrix methods. Require AdamW + strong matrix parity before real headroom. Calibrate warm-start/continuation budget doublings and bracket actual quantized LR. Store source/grouping ambiguity and amend development settings explicitly.
- [ ] Run tests, `python -m ralup run --phase explore --suite headroom --config configs/proxy.json`, then competitive core continuations. Expect interpretable learning/quantization curves, measured costs and B0 rationale, or a supported no-headroom/unsuitable-proxy/resource-limited disposition. No policy-favorable weakening of the strongest recipe.
- [ ] Commit search machinery as `feat: add development headroom and backbone selection`; freeze B0 only from measured evidence and mark WP04 accordingly.

## M08 — Fourteen causal features, history and frozen normalization

**Files:** Create `src/ralup/policy/{features,history,normalizer}.py`, `tests/unit/test_policy_features.py`, `tests/integration/test_history_lifecycle.py`.

**Interfaces:** Produce `FeatureVector(values: Tensor)` of shape `[14]`, `MatrixHistory`, `FrozenNormalizer(mean: Tensor, std: Tensor, source_manifest: str)`, `collect_features(u0, da, gradient, first_moment_w, history, context, loss_ema, progress) -> FeatureVector`, `commit_history(history, before, after, context) -> MatrixHistory`, `resolve_realized_history(history, next_context) -> MatrixHistory`, and `fit_normalizer(trajectories: list[ArtifactRef]) -> FrozenNormalizer`.

- [ ] Hand-compute all 14 entries on small matrices; include empty eligible/nonzero populations, zero denominators, saturation, shape ratios, missing history, scale incompatibility, reversal and refreshed/cached disagreement. Assert EMA updated once, full-schedule progress retained in short episodes, no controller-output circularity and meta-train-only normalization provenance.

  ```python
  assert features.values.shape == (14,)
  assert first_observation_feature_12 == 0
  assert feature_13 == full_schedule_progress
  assert reversal_rate == 0 and compatible_coverage == 0  # changed scales
  with pytest.raises(ProvenanceError): fit_normalizer(target_or_protected_trajectories)
  ```

- [ ] Run task files; show repeated microbatch history commit and unrotated SOAP moment misuse fail.
- [ ] Implement every S19 formula/denominator exactly and log empty/invalid coverage separately. Use original-coordinate first moments from adapters or explicitly charged common EMA. At next-step prepare resolve realized history before features; preserve pending state in snapshots. Fit/freeze normalization on meta-train reference trajectories with std floor1e-6 and clipping[-5,5]. Inventory byte histories, scan overhead and detailed-trace cadence.
- [ ] Run tests and `python -m ralup test-policy --suite features --backbone artifacts/B0.json`; expect hand-fixture parity, no evaluation mutation, full history resume and measured feature overhead/clipping rates.
- [ ] Commit as `feat: add causal cell history features and frozen normalization`.

## M09 — Exact-initialized candidate and matched LR/static controls

**Files:** Create `src/ralup/policy/controllers.py`, `tests/unit/test_controllers.py`, `tests/integration/test_policy_identity.py`; extend adapter/CLI/snapshot schemas.

**Interfaces:** Produce `PolicyAction(alpha: float, h: float, tau: float)`, `Affine45.forward(features: FeatureVector) -> PolicyAction`, `LearnedLR49.forward(features: FeatureVector) -> float`, `Static3.forward(features: FeatureVector) -> PolicyAction`, `PolicyAdapter(base: BaseUpdateRule, controller, normalizer: FrozenNormalizer)`, and `ControllerArtifact` containing family, parameters, normalization, B0, data and training provenance hashes.

- [ ] Assert exact capacities/bounds/initial actions; a complete initialized-policy trajectory equals bypass B0 including auxiliary updates, decay, state/cursor and Parameter identities. Changed actions obey F05 bounds. LR control sees identical features/history/rollout opportunities and can reach multiplier32.

  ```python
  assert count_parameters(Affine45()) == 45
  assert count_parameters(LearnedLR49()) == 49
  assert Affine45()(features) == PolicyAction(0., 1., 1.)
  assert exact_cpu_state_hash(identity_policy_run) == exact_cpu_state_hash(base_run)
  assert lr_controller.max_multiplier == 32
  ```

- [ ] Run task files; prove a policy that reorders base arithmetic or amplifies decay fails identity/decay tests.
- [ ] Implement S17–S20, delegating identity actions to H05's authoritative commit. Use F05 for nonidentity storage actions and map back to W without moving moments. Add future-loss-trained constants, restricted actions and norm-matched attribution mode (flagged privileged). Freeze controllers/normalizers at deployment; equality of source information does not imply equal output range. Charge feature costs equally in controls. Serialize every controller family without executable pickles.
- [ ] Run tests and `python -m ralup test-policy --backbone artifacts/B0.json`; expect exact CPU identity, measured rotated/MPS contract, no Parameter replacement and complete controller-state resume. Record WP05 evidence.
- [ ] Commit as `feat: add RALUP and information-matched policy controls`.

## M10 — MetaQuant fast weights, factor gradients and deployment variants

**Files:** Create `src/ralup/baselines/metaquant.py`, `tests/unit/test_metaquant_fast_weights.py`, `tests/integration/test_metaquant_adapter.py`; extend snapshots/registry.

**Interfaces:** Produce `FactorizedMetaNetwork(widths: tuple = (1,100,100,1))`, `make_fast_weights(u: Tensor, correction: Tensor, previous_direction: Tensor, lr: float) -> Tensor`, and `MetaQuantLMAdapter(base: BaseUpdateRule, config: dict)` with gprev/rprev/theta/outer-state schema. Functional model evaluation uses H04 common forward/scorer and explicit parameter mapping; no hidden module mutation.

- [ ] Compare factor-product forward and each factor's gradient with unfused linear layers. Assert fast numerical value equals `u-lr*rprev` but theta derivative follows correction. Verify first selected-matrix step only primes, next step uses previous gradients, cW is computed before theta update, and rotations map signals both ways.

  ```python
  torch.testing.assert_close(fast_u, u - lr * previous_direction)
  torch.testing.assert_close(actual_theta_gradient, independently_derived_theta_gradient)
  assert priming_selected_matrix_updates == 0
  torch.testing.assert_close(committed_signal, pre_theta_update_cw)
  ```

- [ ] Run tests; demonstrate simplifying detached cancellation to `u-lr*rprev` erases the intended meta-gradient.
- [ ] Implement S25 original uniform learning-mechanism fixture and named FP4 adaptation, preserving factor parameters and their gradients. Define physical-unit calibration, cached-range normalization, auxiliary fast/committed updates and decay explicitly. Initialize bias-free maps with pinned framework/meta seed; meta Adam starts .001 for tuning. Support frozen-theta and separately labeled online-target-training variants; snapshot every history/state field and charge priming/extra loss passes. Verify source rights before reuse.
- [ ] Run tests and `verify-baselines --suite metaquant`; expect original mechanism parity and separate value/derivative checks. MPS functional-gradient limitations remain explicit. Failure prevents a superiority claim over MetaQuant while allowing other diagnostics.
- [ ] Commit as `feat: add validated MetaQuant learning mechanism and LM adaptation`.

## M11 — Full-state antithetic ES, horizon calibration and frozen artifacts

**Files:** Create `src/ralup/meta/{__init__,tasks,es,selection}.py`, `tests/unit/test_es.py`, `tests/integration/test_meta_rollouts.py`; extend CLI/progress.

**Interfaces:** Produce `MetaTask(snapshot: TrainingSnapshot, inner_batches: list[LogicalBatch], query_blocks: list[TargetBlock], rng_bundle: dict, provenance: dict)`, `RolloutResult(objective, initial_query, final_query, work, final_snapshot_ref)`, `rollout(task: MetaTask, controller, horizon: int) -> RolloutResult`, `estimate_es(phi: Tensor, perturbations: Tensor, plus: Tensor, minus: Tensor, sigma: float) -> Tensor`, `sample_meta_task(pool, rng, recipe) -> MetaTask`, and `select_controller(meta_dev: list[dict], official_dev: list[dict]) -> ControllerArtifact`.

- [ ] For a quadratic outer objective and deterministic complete ±basis perturbations scaled to unit covariance, ES matches its analytic gradient; verify division by2mσ and no sign swap. Clones must isolate moments/scales/history/cursor/RNG; plus/minus share all inputs. Assert inner/query document disjointness, identical query endpoint lists and no protected/validation gradient path.

  ```python
  torch.testing.assert_close(estimated_gradient, analytic_quadratic_gradient)
  assert plus_task.initial_state_hash == minus_task.initial_state_hash
  assert inner_document_hashes.isdisjoint(query_document_hashes)
  assert source_snapshot_hash_after_rollouts == source_snapshot_hash_before_rollouts
  assert rollout_work["optimizer_steps"] == 2 * m * K
  ```

- [ ] Run tests; prove weight-only restoration or different plus/minus batch lists fails replay/objective tests.
- [ ] Implement sequential actual rollouts using H06/H07, normalized S21 objective and outer Adam without decay. Include baseline/policy-origin early/mid/late snapshots with recorded mixture, refresh independent of antithetic candidate. Count initial/final query, reset and snapshot costs, including valid initial-query reuse. Start probes K16,m4,σ.05,LR.01; calibrate duplicate noise, perturbation/query signal and K/2K/4K rank stability before setting final budgets. Train candidate/LR/static with equivalent recorded opportunities. Freeze meta-train normalizer and retain global progress. Select on meta-development then full official-development; support the predeclared core-meta branch after diagnosed failed proxy transfer. No autograd graph across inner steps.
- [ ] Run task tests and `python -m ralup meta-train --config configs/meta.json` after required fields resolve. Expect objective/noise/horizon/cost/snapshot artifacts and a selected frozen controller, or a flat/noisy/short-horizon-only/failed-transfer disposition requiring documented revision. A synthetic ES pass is not evidence of Transformer improvement.
- [ ] Commit as `feat: add full-state rollout meta-training and calibrated selection`; update WP06 gates and exact next action.

## Stage exit

Before E01, B0 and candidate/LR artifacts have development provenance; all claim-relevant comparators have explicit parity/adaptation status. Flat objectives, lost long-horizon gains, simpler controls matching the candidate, and no headroom are retained outcomes. Final B* and protected confirmation remain unselected at this stage. Methods blocked by rights or operations stay visible in the eventual scope.
