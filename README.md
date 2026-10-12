# RALUP

Can an update policy learn when a quantized weight needs to cross a rounding boundary, and improve the final model? RALUP tests that question on a byte Transformer. A small controller sees tensor history and the FP32 shadow weights' positions inside quantization cells, then adjusts an ordinary optimizer's proposal: blend additive and multiplicative moves, scale the step, or help a stalled move cross a nearby boundary. Future-loss rollouts train the controller through evolution strategies. Backprop still supplies the model's gradients.

There's no demonstrated advantage yet. Two provisional policies, each produced by a one-step ES proposal from the **same proxy META task**, improved the 32/64-step endpoints and worsened the remaining **496-step endpoint by +0.006914 / +0.013553 bits per byte**, relative to the initialized policy. Those are diagnostics, not independent replications or confirmation. Short rollouts picked moves that lost later. [Current findings](docs/harness/current-research-status.md) retain the evidence and the unresolved comparisons.

The forward weights use simulated E2M1 FP4; shadows, arithmetic and optimizer state remain FP32. This establishes no native low-bit speed or memory benefit. A robust baseline, final candidate and matched learned-LR comparison, scientific budgets and a justified confirmation sample remain unresolved. The official test hasn't been scored.

The useful thing you can run today is the engineering system: explicit quantization contracts, bounded actions, optimizer adapters, complete-state snapshots, cost accounting and guarded scoring. The portable CPU suite uses small authored byte fixtures and needs no private corpus or retained experimental outputs. Passing it checks those contracts. It doesn't settle the research question.

Start with the standard-library CLI:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
ralup --version
ralup config configs/smoke.json --purpose smoke
```

For numerical development, follow the platform-specific dependency installation in [public development](docs/public-development.md), then run:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest --cpu-only -ra
ralup check-numerics --suite numerics --device cpu --output artifacts/public-development/numerics.json
```

Read the [scientific design](docs/superpowers/specs/2026-10-03-ralup-design.md) for the proposed experiment, [publication notes](docs/PUBLICATION.md) for verification and asset scope, and the [archived engineering handoff](docs/harness/engineering-handoff.md) for the detailed local history. Historical hashes and local paths keep their original meaning.

Original code is AGPL-3.0-only; original writing is CC BY 4.0. [Licensing](LICENSING.md) keeps the SOAP MIT exception and external rights explicit. Please credit Artem Legotin, cite [RALUP](CITATION.cff), and record the commit you used.
