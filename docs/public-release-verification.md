# Public-release verification

The public preparation is ready for local integration as exploratory research software. RALUP has no demonstrated optimizer advantage. No repository visibility change, push or package publication was performed.

## Fresh installation and CPU checks

A separate clean clone of commit `67b7241e3e6b707b372abb1121508111b3c2c036` was created without hardlinks or ignored research files. No private corpus, retained environment or experiment artifacts were copied into it. A fresh isolated Python3.14.6 environment installed stable PyPI wheels and `pip install -e '.[test]'`; `pip check` passed. This was not the archived numerical environment. The subsequent source-path spelling correction changes only the public plan, not executable source, dependencies, README, licensing or test behavior.

| Check | Observed result |
|---|---|
| Portable CPU suite, `PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest --cpu-only -ra` | **1286 passed,86 skipped in135.03s**, exit0 |
| Explicit exclusions |78 native cases and8 opt-in historical-evidence cases; no scientific confirmation |
| Numerical and quantizer CPU commands |Both pass, no blockers |
| Data/model harness, `test-harness --suite all --device cpu` |Pass, no blockers |
| Version and smoke configuration |`ralup0.1.0`; smoke JSON resolves |
| Separate lightweight base environment, `pip install -e .` |Version/config work with torch,numpy,scipy,pyarrow,safetensors and pytest all absent |
| Build from clean clone |Wheel built successfully; all6 packaged notice files match source bytes; SOAP code/MIT notice unchanged |
| Public local links and citation |35 exact-case links pass; CFF parsed and required fields checked |
| Historical preservation |Original16453-byte README preserved as exact archive suffix; executable source, historical lock,vendor manifest,asset rights and all existing JSON records unchanged |

The fresh numerical environment used torch2.14.1,numpy2.5.4,scipy1.18.1,pyarrow26.0.0,safetensors0.8.0 and pytest9.1.1. The development extras express compatibility bounds; every permitted version combination was not tested. This verification ran on macOS arm64. The Linux/Python3.11–3.14 GitHub workflow is configured and its action references exist; the workflow itself has not run. Native execution and optional historical reproduction were not part of this check.

The numerical entry points used their documented CPU flags and separate output paths. Doctor prepared a fresh real environment report in the clone before pytest; its bytes were unchanged after tests. No operation/scientific/native qualification was inferred from discovery. Tests in a shared checkout run sequentially, or need a persistent doctor report before overlapping sessions.

## Documentation, licenses and scope

README contains334 words excluding commands and puts the question, mechanism and unfavorable early diagnostic first. Write Like Me source and its required references match repository commit `b7a28e64fed749406a2fd8406d5ebd3eedd584c0`. The original code/prose license convention matches ExactKV,RBC,FamilyTiles and DAV. Exact license-copy hashes and third-party scope are in [LICENSING.md](../LICENSING.md) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Independent task reviews accepted portability and documentation. The source-path spelling nit was corrected. This report is engineering verification, not a new experiment. The one-task short/long horizon diagnostic, missing scientific finalists/budgets/confirmation and protected official test retain the scope in [current research status](harness/current-research-status.md).

Raw verification logs and installer provenance remain in the local ignored `artifacts/public-release-v1/` directory. They are not part of a published dataset or a promise of complete historical experiment replay. Current public setup commands are in [public development](public-development.md); [publication notes](PUBLICATION.md) describe the retained asset boundary.

## Local verification record hashes

- `clean-pytest.log`: `25fa1de9fa318fa0dd8a80e62e61e31be5a2c611c6f38149986b1d5f3affcf15`
- `install-dependencies.json`: `ec97aead312bf4c5bfee28fcaf65baaed6b2c6924db035e760ed4b38763a8517`
- `install-project.json`: `ba11d4688a5bce3109e35afc80504f31cb66b3b0e075014927ed1ac010019c96`
- `base-install.json`: `9d8d0d8578bb67634be957795a7c4fadca9229a46bf75e5b1063965e1e454804`
- `release-checks.json`: `d5594d74ebb3438a8fe5330bd55acb8691e095e878b8fcd613bde4016fdf38c9`
- `clean-numerics.json`: `862a9f155f5bf304a38491c0e79face2854b0d3f0bf4fc871615d9110b27222f`
- `clean-quantizer.json`: `a2edcdb0c6a49a766f262d71ee3200b274aa938b1d616df99a7cf0c9e887caf3`
- `clean-harness.json`: `350107acb48b9610d8281b903b53a3786650db59f7845b8676f0d6af7caf631c`

## Local integration

Final whole-branch review approved the public preparation without outstanding findings. The reviewed commits were fast-forwarded into local `main`. A separate check of that checkout at `e6b6eea` passed **1,286 tests with86 explicit skips in116.16seconds**, exit0, using the same fresh numerical environment and the explicit CPU-only mode. This check covers the main source path and its pre-existing ignored environment records, which differ from the clean clone's newly prepared report. Every pre-existing environment JSON hash remained unchanged. The Git volume retained more than1GiB free. Native execution, protected-test scoring and GitHub publication were not performed.

Merged-main log SHA-256: `2c64d11773ffc7adeadde486c3d37a5acda8561f36b14d6c1e14ed435d8420f7`. Raw log and preservation check: local `artifacts/public-release-v1/main-pytest.log` and `main-verification.json`.
