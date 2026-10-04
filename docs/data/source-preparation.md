# Pinned source preparation

Run from the repository root using the pinned local environment:

```sh
.venv/bin/python -m ralup prepare-data --config configs/data.json
```

H01 authenticates the repository, configuration, immutable revision and each asset's SHA-256/size before any Parquet row is parsed. Existing assets are fully rehashed and reused. Absent assets are streamed from immutable non-executable Parquet URLs after HEAD revision/hash/size authentication, fsynced and atomically accepted only after full byte authentication. Corrupt cache assets fail rather than being replaced. No remote dataset code is loaded. The fixed permanent provenance manifest also authenticates source-card, legal-notice and original download/tree/HEAD evidence bytes.

The H01 `prepare_data` API writes local provenance under ignored `data/processed/`. The CLI now consumes accepted H01 provenance and completes H02 duplicate auditing/partitions, reporting `status=incomplete` with exit 1 because H03-H08 remain missing; see [dedup-partitions.md](dedup-partitions.md). It creates no training targets or scoring interface, and claims no scientific gate or WP02 pass. `configs/data.json` is a narrow source-only settings file, not a resolved research training configuration. The existing full configuration `purpose=prepare-data` boundary requires environment/dependency-lock evidence and WP00; H02 `prepare_corpus(ResolvedConfig)` uses that boundary for full research preparation settings; the source-only CLI remains an explicit bootstrap.

For a synthetic run, use a source-only settings object with `kind=synthetic`, `rows_path=tests/fixtures/wikitext_rows.json` and `output=data/processed/synthetic-h01`. Synthetic artifacts carry their own kind and fixture hash and cannot serve as evidence that official source assets were prepared. Paths in source-only settings are resolved relative to the invocation directory.

Rows stream in deterministic source split/shard order. Both train shards feed a single article iterator so a shard boundary cannot introduce an artificial article boundary. Whitespace is inspected only to recognize headings; source text is never stripped or normalized. Each row contributes its exact UTF-8 bytes followed by LF only if it lacks LF; empty rows contribute LF. Initial non-heading rows form a separately identified preamble document. Duplicate heading titles remain separate articles. No deduplication or partition assignment occurs in H01.

The public `Article(content, source_rows, content_sha256)` interface gives per-row byte offsets, row ordinals, exact raw-byte hashes and appended-LF flags. `reconstruct_articles(rows)` returns a list for small fixtures. Real preparation uses one article at a time and batches of at most 4096 source rows; it does not collect 1.8 million Python strings.

Each split has three local files:

- `*.content.bin`: reconstructed content bytes stored once in article order.
- `*.rows.bin`: fixed 17-byte `<QQB` records, giving absolute content offset, original UTF-8 byte length and appended-LF flag. Row ordinal is implicit. Reading the original-length slice recovers every raw row independently of the derived separator, including empty and CRLF/LF rows.
- `*.articles.jsonl`: article content offsets, source-row half-open ranges, preamble flag and content SHA-256. Identifiers are split plus ordinal, not heading titles.

The `preparation.json` manifest records actual raw UTF-8, derived content/index and downloaded/cache bytes, file hashes and source-row audit hashes. Raw-row audit format is SHA-256 over the ordered concatenation of each original row's uint64 little-endian byte length followed by its original UTF-8 bytes. Preparation independently reverses all indexed rows from disk and requires the resulting audit digest to equal the source digest. Card generated-byte estimates are retained as publisher metadata, never reported as measurements. `docs/data/h01-evidence.json` retains measured evidence without corpus contents; large assets and derived representations remain ignored.

Raw official test-source parsing and hashing are authorized exclusively for this non-scoring provenance/dedup audit, required before H02 partitions. Test targets and all model scoring remain closed until E03. Test artifacts carry `protected_non_scoring_audit_only_until_E03`; that label does not grant scoring access. The historical original download record predates this parsing authorization and correctly records its earlier no-parsing scope.

The source card is preserved at its exact pinned URL and hash with Salesforce Research attribution, the Merity/Xiong/Bradbury/Socher (2016) paper citation, DOI `10.48550/arXiv.1609.07843`, and original BibTeX. The card YAML's CC BY-SA 3.0/GFDL labels conflict with its CC BY-SA 4.0 prose link; this remains unresolved. Official CC BY-SA 3.0/4.0 and GFDL 1.3 reference bytes retrieved on 4 October 2026 are retained and authenticated. GFDL 1.3 remains `reference_only_dataset_does_not_name_version`. Local research is the intended use; no corpus redistribution is authorized and no legal clearance is claimed. Article-level Wikipedia attribution histories are not supplied by this source repository.
