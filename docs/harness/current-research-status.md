# Current research assessment — 12 October 2026

RALUP has **no demonstrated improvement yet**. The measured early signal is unfavorable for the two provisional policies tested: they improved short rollout endpoints but harmed the longer endpoint. This is development evidence from one proxy META task, not a controlled conclusion that representation-aware learning is ineffective.

| Observation | Measured result | Interpretation |
|---|---:|---|
| Two provisional policies at the remaining 496-step endpoint | +0.006914 and +0.013553 bits/byte relative to the initialized policy | Both longer-horizon outcomes are worse; they are not independent meta-training replications |
| Shorter 32/64-step endpoints | Both provisional policies improved | Short-horizon learning can select an action that loses later; WP06 requires revision |
| Same-weight full-precision versus quantized diagnostic | Quantized loss +0.022868 bits/byte | Quantization has a measurable effect in this diagnostic; the difference is a reference gap, not attainable policy headroom or an upper bound |

Ordinary backpropagation still supplies gradients. E2M1 FP4 is simulated in the forward representation with FP32 shadows, arithmetic and optimizer state. These runs establish no native low-bit speed or memory benefit.

The proposed controller has not completed matched, properly tuned comparisons against a final learned-LR controller and independently selected strongest competitor. A robust B0, final C/LR/B*, final scientific budgets, fresh paired development pilot, justified confirmation n and locked manifest remain absent. Official test scoring has not occurred. The current disposition is **exploratory/inconclusive; WP04 inconclusive; WP06 NEEDS_REVISION**.

The next scientifically justified work is longer-horizon/objective calibration followed by matched candidate/LR/static training and development selection. It needs newly admitted current-source state pools and a measured resource budget. Completing software verification does not itself authorize or replace those research prerequisites.

The implementation completed all 32 original engineering task gates. The single broad review and its six-finding consolidated fix are closed. A separate narrow correction repaired malformed synthetic replication-cost fixtures without changing product code. The fresh CPU integration at `f2394fd` completed **1,287 passed, 76 skipped and 2 deselected in 144.37 seconds**. Native availability was deliberately masked; this result qualifies the recorded CPU scope. Historical native/resource failures retain their original scope. Exact evidence and acceptance are linked in [final integration evidence](final-integration-evidence.json).

Current verification and late evidence preservation are administrative work. They do not train or select a policy, justify a confirmation sample, score the protected test, or change the research outcome. Complete project cost remains unknown.

The exact late handoff preserves 514 plan artifacts and all 92 chronological rulings. Only recoverable task scratch was removed after authentication; the branch, worktree, raw experimental records and failed attempts remain intact. Integration is awaiting the user’s choice.
