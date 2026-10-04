# H04 implementation report

Status: DONE. Branch feat/ralup-execution, base eff1085; all work confined to
/Volumes/git/legotin/adaptive-calculus/.worktrees/ralup-execution.

## Implementation

Added the specified S11 FP32 byte decoder and bias-free quantized projections.
Exact total counts: 493,056 / 3,347,456 / 19,281,920; internal counts: 393,216 /
3,145,728 / 18,874,368. Explicit stable ordered names preserve separate
Q/K/V/attention-output/up/down matrices. Embedding/head remain untied; positions
length 256 and vocabulary 257. LayerNorm epsilon is 1e-5, GELU is exact, no
dropout/linear biases, matrix initialization obeys .02 and residual gain.

Reference causal/key-padding attention is the sole path. Padded queries use a
safe diagonal, preserving real-query masks while preventing all-masked softmax
NaNs from reaching gradients. The bounded one-byte/short-block fixture collates
H03 TargetBlock/LogicalBatch fields and original masks/counts. Empty forward
shapes reject; per coordinator decision actual empty logical optimizer batch
rejection belongs H06, before contexts/moments/cursor mutation.

Added configure_quantization with scalar recipe and separate caller-owned runtime
handoffs, once-per-step atomic numeric context preparation, explicit cache
release, read-only cache mapping and optional complete forward context override.
Repeated current-step preparation returns the same detached numeric contexts.
Fixed scales/initial zero flags are explicitly supplied by the caller. Fresh
surrogate graphs are reconstructed every forward; original W Parameter identity
and original-coordinate gradients persist. No moments/history/RNG advances,
production per-step CPU copies or higher-precision arithmetic were introduced.
Pure alternate forward contexts support later H06 read-only final evaluation and
functional parameter callers without mutating training cache. Full precision is
default and explicit empty override mapping also selects it.

Added bounded model checker, test-harness model CLI and numeric contract/evidence
documentation. The model scope is H04 engineering only; H05-H08, complete WP02 and
scientific gates remain pending. No training, corpus tensor materialization,
protected target access or neural quality/capacity claim was performed.

## TDD and controls

RED initial command: `.venv/bin/python -m pytest tests/unit/test_transformer.py -q`
produced the expected absent ByteTransformer/ModelConfig import failure, recorded
in artifacts/checks/h04-red-missing-model.txt. A subsequent test-setup error used
a nonexistent oracle alias; corrected to existing e2m1_oracle before GREEN.
Initial feature GREEN: 13 passed in 1.20s.

RED checker command: `.venv/bin/python -m pytest
 tests/unit/test_transformer.py::test_model_harness_records_counts_and_staged_remaining_work
 tests/integration/test_model_parity.py -q` produced 3 expected failures: CLI
rejected suite=model and model checker module was absent. Artifact:
artifacts/checks/h04-red-checker.txt. GREEN sandbox focused: 19 passed, 1 skipped
in 3.08s; sandbox skip explicitly means actual MPS unavailable.

Mandatory negative controls temporarily changed and restored production source:
- Missing causal-mask `.tril()` failed future-token invariance with 771/771
  differing logits and max absolute difference .03231491148471832.
- Fused-QKV manifest preserved total/internal counts but failed all three exact
  separate-matrix manifest fixtures (3 failures); correct source restored.
Artifacts h04-red-missing-causal-mask.txt and h04-red-fused-qkv-manifest.txt.

Self-review found dependence on an external global default dtype. Added a RED
regression with global float64 default: asserted FP32 parameters, observed
float64, 1 failure (h04-red-default-dtype.txt). Explicit constructor FP32 fixed
it; GREEN 1 passed in .49s. This observed regression justified one final refresh
of focused/full tests and checker evidence after the earlier passing run.

## Final verification and actual device evidence

Pinned .venv, PYTORCH_ENABLE_MPS_FALLBACK=0 before interpreter startup:
- `python -m pytest tests/unit/test_transformer.py tests/integration/test_model_parity.py -q`:
  21 passed in 6.00s on actual MPS via authorized escalation.
- `python -m pytest -q`: 390 passed in 17.49s, pristine output, actual MPS enabled.
- `python -m ralup test-harness --suite model --output artifacts/checks/h04-cpu.json`: pass.
- Same CLI `--device mps --output artifacts/checks/h04-mps.json`: pass under
  authorized escalation. Separate sandbox mps CLI report is blocked, not a pass.

Frozen before output measurements: loss/logits/gradient atol=rtol=1e-4 from F02
conservative FP32 smoke bounds; rotation atol=8e-6, rtol=2e-6 from F04; code
mismatches zero and duplicate loss/logits/gradients/code variability zero from
prior F02/F03-F05 measured allowances. No bounds were loosened after results.
F02 prior complete-backward gradient error was 3.2782554626464844e-7.

Four cases: full-precision proxy; small identity STE; small full-Hadamard STE;
small full-Hadamard QuEST trust. Each has five scored bytes plus short/one-byte
padding and an entirely padded physical row. Final actual-MPS max absolute
errors: loss 1.9073486328125e-6, gradients 2.2649765014648438e-6, logits
4.76837158203125e-7; rotation 0, codes mismatches 0, all duplicate errors 0.
CPU repeated checks had zero error. CPU/MPS source hash dictionaries match and
current source hashes were verified. Every artifact hash, including full/focused
test logs, mutation logs and separate sandbox report, is committed in
`docs/numerics/h04-evidence.json`; raw artifacts remain local ignored files.

## Files and self-review

Task-owned files: src/ralup/models/{__init__,linear,transformer,checks}.py;
src/ralup/__main__.py; tests/unit/test_transformer.py;
tests/integration/test_model_parity.py; docs/numerics/{model.md,h04-evidence.json};
this report. Coordinator retains SDD ledger ownership.

Read all new production source, CLI diff, test coverage and numerical evidence;
checked architecture formula/counts, masks, scale handoff/cache lifecycle,
Parameter identity, evaluation override and atomic context publication. No
optimizer/trainer implementation or optimized attention was added. No known
correctness concern remains in the H04 scope. Context tensor immutability is a
caller convention inherited from F04. Callers must clear/reprepare contexts
when parameter device/storage changes and explicitly pass previous scales for
fixed recipes. Scientific model/optimizer/trainer trajectory evidence and empty
logical optimizer rejection are future gates, not represented by this report.
