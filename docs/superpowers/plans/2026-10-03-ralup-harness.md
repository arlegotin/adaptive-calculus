# RALUP Shared Research Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a byte-exact data/model/training/scoring system with complete resumability and measured hardware calibration.

**Architecture:** Immutable source and block manifests feed one logical-batch engine. Methods exchange proposals through a common adapter; evaluation is read-only and snapshots restore the entire computation, not only weights.

**Tech Stack:** F02-locked Python/PyTorch/NumPy/SciPy stack, pinned Parquet reader/downloader, safe tensor storage and pytest.

**Spec:** [S05–S12, S15–S16, S27–S32](../specs/2026-10-03-ralup-design.md). [Master plan](2026-10-03-ralup.md). Requires F01–F05.

## Global Constraints

- Vocabulary is 257: raw bytes `0..255` and input-only boundary token `256`.
- Context length 256; target blocks contain at most 128 bytes.
- Logical batch initially contains exactly 8,192 scored bytes.
- Moments remain in original W coordinates.
- Official validation is development; official test stays protected until the locked manifest and all planned final checkpoints exist.
- Backpropagate summed NLL across microbatches, divide gradients once by total selected targets, then clip once.

## Review Focus

- Non-ASCII rows, empty records and nested headings must not change source bytes or discard preamble — H01.
- A near-duplicate chain through protected text must exclude the entire connected training component — H02.
- A split target block must retain original context/positions across batch and epoch boundaries — H03.
- Evaluation or physical accumulation must not advance quantizer history/RNG or change the gradient — H04/H06.
- A crash during publication or an incomplete update must restore one coherent state/cursor/cost history — H07/H08.

---

## H01 — Pinned source assets and reversible article construction

**Files:** Create `src/ralup/data/{__init__,source,articles}.py`, `configs/data.json`, `tests/unit/test_articles.py`, `tests/fixtures/wikitext_rows.json`; modify CLI/license ledger.

**Interfaces:** Produce `Article(content: bytes, source_rows: list[dict], content_sha256: str)`, `fetch_source(repository: str, configuration: str, revision: str, destination: Path) -> ArtifactRef`, `is_article_heading(text: str) -> bool`, and `reconstruct_articles(rows: Iterable[str]) -> list[Article]`. Downloads verify repository revision and per-asset hash; reconstruction retains source row bytes independently of appended separators.

- [ ] Write fixtures with preamble, empty row, CRLF, existing LF, Unicode, duplicate headings and nested headings. Assert exact reconstruction and reversible row offsets.

  ```python
  assert is_article_heading(" = Title = ")
  assert not is_article_heading("= = Section = =")
  assert not is_article_heading("== Section ==")
  assert reconstruct_articles(["é", ""])[0].content == "é\n\n".encode("utf-8")
  ```

- [ ] Run `python -m pytest tests/unit/test_articles.py -q`; verify failures for Unicode normalization, double-appended LF and dropped preamble.
- [ ] Implement S08–S09 with exact UTF-8 preservation, deterministic source ordering, non-executable asset reads and source/content hashes. Resolve a real immutable dataset revision only during execution; record license/attribution texts and download/preprocessed bytes. Add `prepare-data --config PATH` as a pipeline command, initially reporting later missing stages rather than inventing partitions.
- [ ] Run tests and prepare the local synthetic row fixture; require byte-identical reconstruction and row audit manifest. For real download verify each hash, configuration and source split before advancing; no model is scored.
- [ ] Commit as `feat: add pinned corpus ingestion and reversible articles`; preserve large source assets locally outside Git.

## H02 — Deterministic deduplication, partitions and manifests

**Files:** Create `src/ralup/data/{dedup,partitions}.py`, `tests/unit/test_dedup_partitions.py`, `tests/integration/test_data_manifest.py`; complete prepare-data CLI.

**Interfaces:** Consume `Article`. Produce `CorpusManifest`, `audit_duplicates(train: list[Article], protected: list[Article], config: dict) -> dict`, `partition_article(content: bytes) -> str`, and `prepare_corpus(config: ResolvedConfig) -> CorpusManifest`. Manifest exposes `.sha256`, partition counts/bytes, source row references, canonical/excluded hashes, reasons and near-duplicate candidates/exact similarities.

- [ ] Build deterministic exact-duplicate and shingle fixtures, including protected→train-A→train-B chains, short documents, NFKC-only overlaps and partition hash threshold boundaries. Fixture-injected hash values test all three exact rational ranges; repeated seeds must reproduce candidates/partitions.

  ```python
  assert retained_training_hashes.isdisjoint(protected_hashes)
  assert {hash_a, hash_b} <= excluded_component_hashes
  assert partition_counts == second_run_partition_counts
  assert raw_unicode_bytes == source_article.content
  ```

- [ ] Run the task tests; verify a pairwise-only exclusion implementation fails the connected-component case.
- [ ] Implement full S09 order: exact canonicalization/overlap, seeded 128-MinHash 16×8 candidate generation, exact Jaccard at .8, connected-component exclusion/canonical choice, then raw-content partition hashing. Define hash functions using recorded stable seed bytes and a documented 64-bit hash/permutation algorithm; no Python hash. Record LSH miss limitation and audit-only normalization. Hashing protected documents here permits overlap inspection, not model scoring.
- [ ] Run tests and `python -m ralup prepare-data --config configs/data.json`; expect revision/hash/count/byte/exclusion manifests, no unresolved source fields and zero exact retained-training/protected overlap. Pause real modeling if partition sizes require a dated allocation amendment.
- [ ] Commit as `feat: add reproducible corpus deduplication and partitions`; record WP02 data evidence.

## H03 — Exact byte tiling, logical batches and resumable cursor

**Files:** Create `src/ralup/data/{blocks,batches,cursor}.py`, `tests/unit/test_byte_blocks.py`, `tests/unit/test_batch_cursor.py`.

**Interfaces:** Produce `TargetBlock(article_hash, target_start, target_end, inputs, labels, positions, score_mask)`, `tile_article(article: Article) -> list[TargetBlock]`, `BatchCursor(state: dict)`, `LogicalBatch(block_slices: list[TargetBlock], target_count: int)`, and `next_logical_batch(cursor: BatchCursor, target_budget: int) -> LogicalBatch`. Cursor stores epoch/permutation position and carried score interval; no physical microbatch field affects selection.

- [ ] Test article lengths 1,127,128,129,255,256,257; every `(article_hash,byte_offset)` appears exactly once per epoch, target IDs exclude 256 and masks exclude context/padding. Use byte budget 131 to split a block; compare contexts/positions before/after carry. Test final partial run budget and epoch rollover.

  ```python
  assert sorted(scored_offsets) == list(range(len(article.content)))
  assert first_block.inputs[0] == 256
  assert first_batch.target_count == 131
  assert resumed_selected_offsets == uninterrupted_selected_offsets
  assert packing_one_selected_offsets == packing_four_selected_offsets
  ```

- [ ] Run both tests; verify a retiled carried block fails the fixed-context/position assertion.
- [ ] Implement S10 shift/masking and seeded block permutation without replacement. Split only score masks, preserving original sequence inputs and positions. Store safe cursor metadata and named data RNG state. Track unique offsets independently of cumulative trained offsets.
- [ ] Run focused tests and `python -m ralup test-harness --suite data`; expect complete once-only coverage, deterministic cursor restoration, equal logical schedule across physical packing, and correct final short batch denominator.
- [ ] Commit as `feat: add exact-byte batching and resumable stream cursors`.

## H04 — Reference Transformer and quantized layers

**Files:** Create `src/ralup/models/{__init__,transformer,linear}.py`, `tests/unit/test_transformer.py`, `tests/integration/test_model_parity.py`.

**Interfaces:** Consume F04 contexts/H03 batches. Produce `ModelConfig(layers: int, width: int, heads: int)`, `ByteTransformer(config: ModelConfig)`, `forward(inputs: Tensor, positions: Tensor, padding_mask: Tensor | None) -> Tensor`, `quantized_matrix_manifest() -> list[dict]`, and `prepare_quantized_views(step: int, rng: object) -> dict[str, QuantizationContext]`. Forward returns `[batch,time,257]` logits; quantized layers retain original W Parameter identity.

- [ ] Test all S11 parameter counts and matrix groups, future-token causal invariance, initialization distributions/gains/biases, untied embedding/head, LayerNorm epsilon and exact GELU. Test short/one-byte padded rows for finite loss/gradients, with no all-masked attention softmax contaminating scored positions; reject an empty logical optimizer batch. On small deterministic weights, Hadamard full-precision output equals unrotated output; quantized forward equals direct oracle arithmetic.

  ```python
  assert [parameter_count(c) for c in configs] == [493056, 3347456, 19281920]
  assert core_quantized_parameter_count == 3145728
  torch.testing.assert_close(logits_before[:, :cut], logits_after[:, :cut])
  assert model.embedding.weight is not model.output.weight
  ```

- [ ] Run unit tests; prove a missing causal mask and fused-QKV manifest violate fixtures.
- [ ] Implement the S11 reference architecture and explicit stable matrix names. Integrate F04 cached numeric contexts with fresh surrogate graphs. Quantize projections only. Keep optimized attention disabled until independently tested against reference masks, logits and gradients.
- [ ] Run CPU/MPS loss/gradient parity and `python -m ralup test-harness --suite model`; expect exact counts and frozen numeric-tolerance compliance. Skip unsupported device execution visibly rather than altering architecture.
- [ ] Commit as `feat: add specified byte Transformer and quantized projections`.

## H05 — Uncommitted optimizer proposals and common method adapter

**Files:** Create `src/ralup/optim/{__init__,base,adamw}.py`, `src/ralup/baselines/{__init__,registry}.py`, `tests/unit/test_adamw_adapter.py`, `tests/unit/test_method_registry.py`.

**Interfaces:** Produce `SchedulePoint(lr, weight_decay, progress)`, `Proposal(reference_after_decay, gradient_displacement, new_optimizer_state)`, `BaseUpdateRule.propose(w, grad, state, schedule) -> Proposal`, `BaseUpdateRule.commit(w, proposal) -> None`, and `MethodAdapter`. Common adapter hooks are `prepare_step(training_state, logical_batch) -> None`, `backward_microbatch(training_state, batch_slice) -> dict`, and `finish_step(training_state, schedule) -> dict`; they expose work counts, never privately select data. Default adapter delegates ordinary backward/proposal/commit; exceptional methods override declared hooks. Registry requires S22 fields and reports claim eligibility.

- [ ] Compare fixed gradient sequences against pinned `torch.optim.AdamW` with foreach/fused disabled: first/late steps, nonzero decay, zeros, beta choices and epsilon. Verify proposal construction does not mutate W or input state; commit preserves object identity and exact upstream ordering. Reject unregistered/missing-adaptation methods.

  ```python
  assert torch.equal(adapter_weight, torch_weight)
  assert original_parameter is parameter_after_commit
  assert torch.equal(weight_before, weight_after_propose)
  assert state_before_hash == state_after_propose_hash
  ```

- [ ] Run task tests; demonstrate a changed decay order and duplicate moment update fail.
- [ ] Implement S16 using the same primitive ordering as the selected upstream scalar path. `commit` is the authoritative bypass path for M09 identity actions. Keep no-decay auxiliary groups explicit and include schedule/method registry metadata. Every later adapter extends snapshot schemas and declares extra passes.
- [ ] Run tests and `python -m ralup verify-baselines --suite adamw`; expect bitwise identity for the configured CPU path and declared MPS comparison results. A factorized mathematically equivalent calculation that changes rounding must not silently satisfy exact identity.
- [ ] Commit as `feat: add base update protocol and exact AdamW adapter`.

## H06 — One logical-step engine and common byte scorer

**Files:** Create `src/ralup/{train,evaluate}.py`, `tests/integration/test_logical_step.py`, `tests/integration/test_scoring.py`; extend CLI/records.

**Interfaces:** Produce `TrainingState(model, method, optimizer_state, quantizer_state, histories, schedule, cursor, rng_streams, ledger)`, `StepResult(nll_sum, target_count, work, telemetry)`, `ScoreResult(nll_sum, target_count, bpb, per_article)`, `train_step(state: TrainingState, batch: LogicalBatch) -> StepResult`, and `score_blocks(model, blocks, *, evaluation_context: dict) -> ScoreResult`. Only the future E03 protected gateway can supply test blocks to a scoring command.

- [ ] Assert one logical batch with unequal score counts gives the same gradient/update under microbatch sizes 1 and 3, clips once, updates moments once, and does not double-normalize. Duplicate model scorings are identical; a controlled degraded byte predictor must score worse. Evaluation leaves training state/history/RNG unchanged.

  ```python
  torch.testing.assert_close(accumulated_gradient, full_batch_gradient)
  assert score.nll_sum / (math.log(2) * score.target_count) == score.bpb
  assert score.target_count == sum(len(a.content) for a in fixture_articles)
  assert training_state_hash_before == training_state_hash_after_evaluation
  ```

- [ ] Run both integration files; test wrong unweighted mean accumulation, repeated transition history and in-place Parameter replacement as detectable regressions.
- [ ] Implement S15 transaction ordering with H05 adapters and H03 cursor, sum-based backward, optional global clipping, explicit auxiliary update, and three ledgers. Expose `run --phase explore --config PATH` and `test-harness`. Guard production unresolved values; synthetic fixture runs remain labeled. Scorer aggregates sums in a declared stable order/precision and uses independent evaluation RNG/context. Method failure records nonfinite state and stops before committing a corrupt checkpoint.
- [ ] Run integration tests and `python -m ralup test-harness`; expect coverage/model/accumulation/scoring reports and duplicate score equality. Run a short synthetic CPU trajectory and, after F02, its MPS counterpart; no held-out test data.
- [ ] Commit as `feat: add shared logical training and byte scoring engine`.

## H07 — Complete snapshots and interruption-safe publication

**Files:** Create `src/ralup/state.py`, `tests/unit/test_checkpoint_integrity.py`, `tests/integration/test_resume.py`; extend state/CLI.

**Interfaces:** Produce `TrainingSnapshot`, `capture_snapshot(state: TrainingState) -> TrainingSnapshot`, `restore_snapshot(snapshot: TrainingSnapshot, device: str) -> TrainingState`, `save_snapshot(snapshot, destination: Path) -> ArtifactRef`, and `load_snapshot(reference: ArtifactRef) -> TrainingSnapshot`. Capture deeply copies all tensor/RNG/state payloads; it does not alias the running state. Serializer permits only registered schema types, plain metadata and tensor arrays, never arbitrary pickle code.

- [ ] Compare uninterrupted N steps with A + save/load + N−A using a partially consumed target block and nontrivial moments/scales/history/EMA/scheduler. Inject corruption, truncation, unknown schema, mismatch hashes, interrupted fsync/rename, and partial-step capture. Restore costs retain past compute but record new process elapsed time distinctly.

  ```python
  assert exact_cpu_state_hash(uninterrupted) == exact_cpu_state_hash(resumed)
  assert uninterrupted.cursor.state == resumed.cursor.state
  with pytest.raises(IntegrityError): load_snapshot(corrupted_reference)
  with pytest.raises(IncompleteStep): capture_snapshot(mid_step_state)
  ```

- [ ] Run both task files; prove restoring only W fails trajectory equivalence.
- [ ] Implement S30 in the safe format chosen by F02, with payload checksums and atomic published directory/manifest on one filesystem; recover the last valid snapshot after failed publication. Retain two valid generations plus declared milestones. Named initialization/data/quantization/meta/timing streams are independent and serializable. Register new method state schemas rather than silently omitting unknown fields.
- [ ] Run tests on CPU and MPS; expect exact deterministic CPU trajectory, MPS differences bounded by predeclared duplicate-run variation, and no usable partial/corrupt checkpoint. Produce a resume report with full state-field inventory.
- [ ] Commit as `feat: add complete atomic training snapshots and replay`.

## H08 — Measured calibration and separate warm-start machinery

**Files:** Create `src/ralup/calibrate.py`, `tests/unit/test_calibration_accounting.py`, `tests/integration/test_warmstart_isolation.py`; extend CLI/progress.

**Interfaces:** Consume H06/H07. Produce `calibrate(config: ResolvedConfig, microbatches: list[int]) -> ArtifactRef`, `train_warmstart(config: ResolvedConfig, partition: str, seed: int) -> ArtifactRef`, and `forecast_cost(measurements: dict, design: dict) -> dict`. Expose `calibrate --config PATH` and `run --phase explore --suite warmstart --config PATH`.

- [ ] Test disposable warmup/restore, synchronized full-step timings, overhead/reset/query/I/O accounting, no double-counted RSS+Metal, and same logical byte stream under smaller OOM-retry packing. Warm starts for meta episodes may consume only meta-train; target warm starts only target-training.

  ```python
  assert meta_warmstart.source_partition == "meta_train"
  assert target_warmstart.source_partition == "target_train"
  assert post_warmup_initial_hash == original_initial_hash
  assert oom_retry_target_offsets == no_retry_target_offsets
  ```

- [ ] Run task tests; verify wrong partition ancestry and uncharged extra queries are detected.
- [ ] Implement S12/S29 hardware caps, microbatch search from one, empirical memory model, sustained randomized-arm timing and query/reset/checkpoint transfer measurements. Record resident and driver peaks separately plus inventory of live tensors. Build duration-doubling jobs and immutable warm-start artifacts; do not select B0 yet. Source snapshot cost estimates come from actual states, including SOAP when later present.
- [ ] Run focused tests, a bounded synthetic calibration, then real proxy calibration under the hardware cap. Expect a measured capacity/time interval and separate partition ancestry. Continue scientific warm-start duration selection in M07; H08 completion does not invent final duration or convergence.
- [ ] Commit as `feat: add measured capacity and warm-start calibration`; record completed WP02 checks and calibration infrastructure.

## Stage exit

`test-harness` and complete resume must pass before comparisons. H08 produces measured calibration and warm-start tools; M07 resolves scientific durations/backbone after strong ordinary optimizer ports. Protected test scoring remains unavailable. No successful neural comparison is required to finish the harness implementation.
