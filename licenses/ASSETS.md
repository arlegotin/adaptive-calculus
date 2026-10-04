# Third-party assets

Runtime, build, and test wheels are installed locally. No third-party code was modified or copied into the application. This manifest is platform-specific. `requirements.lock` authenticates the exact successful wheel archives against official PyPI release metadata; installed bytes are separately checked against distribution RECORD hashes. Archive bytes were not downloaded a second time.

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
