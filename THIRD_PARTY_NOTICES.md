# Third-party notices and scope

Project licenses cover original material only. This inventory supplements the unchanged [asset rights record](licenses/ASSETS.md) and [vendor manifest](vendor_manifest.json); their historical hashes, scope statements and unresolved rights remain authoritative for the assets they identify. No dependency wheel, corpus or external model is bundled in the RALUP wheel.

## Copied SOAP reference

The untouched pinned SOAP CPU oracle is [soap_reference.py](src/ralup/baselines/references/soap_reference.py), from `a1e553530fde97d0e6b307d7c82ac6d38b072340`, SHA-256 `9e9021335ad02371584622fa256e73fe75f3eb74446a8d8a216e956534148f6b`. Copyright (c) 2024 Nikhil Vyas. It is licensed under MIT. Its original AdamW/GaLore provenance comments remain intact. The full [MIT license](licenses/methods/soap/LICENSE), SHA-256 `c77e3df866b9eed32fa87976d0d9f1912d8a6fb697ce522cecdc0d5dee309188`, remains verbatim in the packaged [reference initializer](src/ralup/baselines/references/__init__.py) and is included as a wheel license file. The project adapter changes ownership/transport while preserving the declared source mechanics; see [M02 methods](docs/harness/m02-methods.md).

## Independently implemented equations

Muon and Muon+ equations were independently implemented; no inspected repository code was copied. Muon's inspected MIT notice remains in [licenses/methods/muon/LICENSE](licenses/methods/muon/LICENSE). Muon+'s arXiv distribution grant does not establish an author software reuse license; source-reference eligibility remains blocked. See [M01 methods](docs/harness/m01-methods.md).

The Madam comparator independently implements equations; it copies no author software governed by the inspected CC BY-NC-SA 4.0 notice. Author-source copying remains blocked and release reproduction unverified. Independent mathematical implementation, bounded equation fidelity, actual operations and future scientific eligibility are separate assessments; see [M02 methods](docs/harness/m02-methods.md).

The QuEST-related quantization recipes are project adaptations of paper mathematics, with a project Gaussian refit, full-row Hadamard rotation and declared physical trust threshold. No inspected upstream code or restricted kernel is copied; these recipes do not establish upstream release reproduction. See [quantization provenance](docs/numerics/quantization-context.md).

## Installed dependencies

Numerical, build and test packages keep their own licenses. The version-scoped inventory in [licenses/ASSETS.md](licenses/ASSETS.md) links to retained license and bundled-notice texts under `licenses/packages/`; [vendor_manifest.json](vendor_manifest.json) records exact wheel paths and hashes. Those records describe the historical CPython 3.14 macOS arm64 environment. Public extras can resolve different versions and require their own installed distribution notices. RALUP builds do not bundle those dependency wheels or imply that the project license replaces their terms.

## Dataset and research references

The pinned WikiText card lists CC BY-SA 3.0/GFDL in YAML while its prose links CC BY-SA 4.0; the GFDL version is unspecified. The retrieved GFDL 1.3 text is reference-only, never a selected dataset license version. Article-level author histories are absent. Existing records authorize local research, not corpus redistribution, and establish no legal clearance. The repository and RALUP wheel redistribute no corpus. See [source preparation](docs/data/source-preparation.md) and [licenses/ASSETS.md](licenses/ASSETS.md) for exact notices, attribution and limitations. Cited papers and separately marked reference material retain their own terms.
