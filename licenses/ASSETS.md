# Third-party assets

Runtime, build, and test wheels are installed locally. F02 copied no third-party code into the application; subsequent method-specific reuse is declared below. This manifest is platform-specific. `requirements.lock` authenticates the exact successful wheel archives against official PyPI release metadata; installed bytes are separately checked against distribution RECORD hashes. Archive bytes were not downloaded a second time.

Wheel license and bundled third-party notice texts are retained below under `licenses/packages/`; their SHA-256 values and original wheel paths are in `vendor_manifest.json`. Redistribution of application wheels is deferred; any later distribution must retain all these notices and reassess bundled obligations. No external data, model, source fixtures, or weights are included by F02.

| Package | Version | Metadata license | Notices |
| --- | --- | --- | --- |
| filelock | 4.0.9 | MIT | 1 |
| fsspec | 2026.9.0 | BSD-3-Clause | 1 |
| iniconfig | 2.3.0 | MIT | 1 |
| Jinja2 | 3.1.6 | Unspecified; see retained notices | 1 |
| MarkupSafe | 3.0.4 | BSD-3-Clause | 1 |
| mpmath | 1.3.0 | BSD | 1 |
| networkx | 3.7 | BSD-3-Clause | 1 |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | 19 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | 3 |
| pip | 26.1.2 | MIT | 42 |
| pluggy | 1.6.0 | MIT | 1 |
| pyarrow | 25.0.1 | Apache-2.0 | 2 |
| Pygments | 2.21.0 | BSD-2-Clause | 2 |
| pytest | 9.1.1 | MIT | 1 |
| safetensors | 0.8.0 | Unspecified; see retained notices | 1 |
| scipy | 1.18.1 | Full license in retained notices | 5 |
| setuptools | 84.0.0 | MIT | 17 |
| sympy | 1.14.0 | BSD | 3 |
| torch | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | 104 |
| typing_extensions | 4.16.0 | PSF-2.0 | 1 |

Safe checkpoint choice: safetensors 0.8.0 (Apache-2.0), dense contiguous CPU tensors and schema-validated non-executable JSON. Safetensors and PyArrow are imported successfully in the isolated environment; H07 owns snapshot round-trip, alias preservation and full trajectory/resume verification.

WikiText source assets are pinned to `Salesforce/wikitext`, `wikitext-103-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Exact card bytes, immutable URLs, source hashes, original verified downloads and repository HEAD/tree evidence are retained under `vendor/source/wikitext/`. Official retrieved license notices and their byte hashes are retained under `licenses/sources/wikitext/`. See `docs/data/source-preparation.md` and `docs/data/h01-evidence.json` for authentication and measured storage.

The supplied card YAML lists CC BY-SA 3.0 and GFDL with no GFDL version; its prose links CC BY-SA 4.0. This discrepancy remains unresolved. The retrieved GFDL 1.3 text is **reference_only_dataset_does_not_name_version**, never a selected dataset license version. Intended use is local research; no corpus redistribution is authorized. Dataset paper attribution and the complete supplied card BibTeX are retained, while article-level author histories are absent from the repository. These retained notices do not establish legal clearance.

M01 adds independently implemented Muon and Muon+ equations. The inspected Muon repository is pinned to `f98f1cacc0263b04290753e32be8d498c1efc806`; its exact MIT notice is retained under `licenses/methods/muon/LICENSE` (SHA-256 `8c8d17398ecce3dd3fcb1bb77b5d268a427f8250f364ac655c9b38df5b693f6b`). No repository code was copied. Muon+ is pinned to arXiv:2602.21545v3, primary HTML SHA-256 `10536d1f1921f61f8f427c196990ca1855aef3a414be28a4be594bfacd0a2679`. Its arXiv distribution grant does not establish an author software reuse license; raw license retrieval remains unresolved, no hash or software grant is invented. The clean equation implementation is a declared project adaptation and source-reference eligibility remains blocked, separate from numerical/operations checks. Neither method has a source release reproduction or scientific winner claim.

WP00's original and v2 local data research acceptance cover only their immutable, hash-addressed historical metadata snapshots in `docs/data/wp00-objects/`. These M01 metadata additions are outside those acceptances; historical proof remains unchanged. Full permanent numerical choices and scope are recorded in `docs/harness/m01-methods.md` and `vendor_manifest.json`.

M02 reuses the permitted pinned SOAP MIT source `a1e553530fde97d0e6b307d7c82ac6d38b072340`. The untouched 19,180-byte oracle is `src/ralup/baselines/references/soap_reference.py`, SHA-256 `9e9021335ad02371584622fa256e73fe75f3eb74446a8d8a216e956534148f6b`. Its AdamW/GaLore provenance comments remain intact. The full Nikhil Vyas MIT notice is `licenses/methods/soap/LICENSE`, SHA-256 `c77e3df866b9eed32fa87976d0d9f1912d8a6fb697ce522cecdc0d5dee309188`, and is included verbatim in the packaged reference module's initializer. The adapter changes ownership/transport while preserving the declared source mechanics.

M02 Madam independently implements the mathematical equations, with no restricted author software copied. The inspected release is `c2d23592c5e18d6c9e6fcb2a5213ac288a6153cc`, with CC BY-NC-SA 4.0 notice SHA-256 `1349a4b6148492b44f629e64eed676612e234fe9a839e4f3b277c1482c8849f1`. Author-source copying remains blocked; the independent named mathematical local comparator follows a separate research route, without implying a software grant or legal assurance. Source-copy/release, bounded equation fidelity, actual operations and future scientific eligibility remain separate. See `vendor_manifest.json` and `docs/harness/m02-methods.md`. Historical WP00 proofs and source audit remain unchanged.
