# Public development

The portable suite exercises engineering contracts with small, independently authored byte fixtures. It needs no private corpus or retained experiment outputs and provides no scientific confirmation. Run commands from the repository root. Python 3.11–3.14 are the CPU CI targets; newer Python versions are not yet CI targets.

## Install

A standard-library configuration CLI needs only an editable install:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
ralup --version
ralup config configs/smoke.json --purpose smoke
```

For numerical development on Linux, install PyTorch from its CPU wheel source first. This avoids selecting the large CUDA dependencies distributed with ordinary Linux wheels. Then install the project's numerical and test extras from PyPI:

```sh
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m pip check
```

On macOS, use `python -m pip install -e '.[test]'`; the normal PyTorch macOS wheels support CPU and MPS. `.[numerics]` installs numerical dependencies without pytest. Base installation leaves those optional dependencies out.

The extras bound PyTorch, NumPy, SciPy, PyArrow and safetensors to compatible API families. The code uses `torch.func`, default-device inspection, NumPy generators, SciPy distributions/solvers, Arrow Parquet readers and safetensors byte serialization. Release metadata establishes the lower release families: [PyTorch 2.6](https://pypi.org/project/torch/2.6.0/), [NumPy 2.2](https://pypi.org/project/numpy/2.2.0/), [SciPy 1.15](https://pypi.org/project/scipy/1.15.0/), [PyArrow 19](https://pypi.org/project/pyarrow/19.0.0/) and [safetensors 0.5.2](https://pypi.org/project/safetensors/0.5.2/). These are compatibility bounds, not a claim that every permitted combination was tested. CPU wheel selection follows [PyTorch's installation instructions](https://pytorch.org/get-started/previous-versions/).

## Test and check numerics

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest --cpu-only -ra
ralup check-numerics --suite numerics --device cpu --output artifacts/public-development/numerics.json
ralup check-numerics --suite quantizer --device cpu --output artifacts/public-development/quantizer.json
ralup test-harness --suite all --device cpu --output artifacts/public-development/harness.json
```

`--cpu-only` visibly skips native cases and disables native RNG inspection in the test process and its replay children. `RALUP_CPU_TESTS=1` is an equivalent test-only switch. The header and `-ra` summary identify the exclusions; this is CPU coverage, not native validation.

Default pytest prepares a fresh, real `discover_environment()` report at `artifacts/environment/environment.json` when absent, labels it as engineering-test preparation, and removes only its own unchanged report after the session. An existing report is never overwritten or deleted. Source files, lock bytes, environment bytes and actual numerical runtime still authenticate normally, including fresh-interpreter replays. Unprepared production use continues to fail closed. This discovery record is not an operation gate or archived proof. Run pytest sessions sequentially within one checkout, or prepare a persistent report before parallel sessions so an owning session cannot remove a borrowed report. To retain that deliberately prepared local report, run this first (CI does so explicitly):

```sh
ralup doctor --output artifacts/environment/environment.json
```

## Optional local evidence and native execution

Historical tests have the `local_evidence` marker and require an explicit `--local-evidence` opt-in. They include frozen M01 metadata, retained E01/E02 refusals and preserved META normalizer/data checks. They retain original path/hash authentication and may require the original local evidence and corpus layout. Portable synthetic authority and denial regressions remain in the default suite.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest --cpu-only --local-evidence -m local_evidence -ra
```

For native coverage, use a supported Apple Silicon macOS environment, omit the CPU switch and start with fallback disabled. Availability and numerical gates remain enforced; unavailable MPS cases skip or retain blocked reports. No long native training is part of this setup.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest -m native -ra
PYTORCH_ENABLE_MPS_FALLBACK=0 ralup check-numerics --suite quantizer --device mps --output artifacts/public-development/native-quantizer.json
```

`requirements.lock` remains the historical pip `--require-hashes` input for one macOS arm64 CPython 3.14 environment. Its exact platform-specific wheel hashes are research identity bytes, not a portable cross-platform installation recipe. It is not JSON; separate archived wheel metadata reports are JSON. Use the extras above for public development and consult the historical environment instructions when reproducing that specific archived environment.
