# H04 reference byte Transformer

`ModelConfig(layers, width, heads)` contains only architecture values. The model
uses the S11 separate bias-free projections, pre-LayerNorm (`eps=1e-5`), exact
GELU, learned 256 absolute positions, 257 byte/boundary inputs, untied head,
final LayerNorm and specified initialization. Parameters are explicitly FP32.
The reference attention implementation is the only attention path.

`forward(inputs, positions, padding_mask=None)` accepts matching int64
`[batch,time]` tensors and returns `[batch,time,257]` FP32 logits. True padding
mask entries denote padding. Real queries combine causal and key-padding masks.
Padded queries use a safe diagonal attention row; no all-masked softmax enters
backward, including an entirely padded physical row. Loss consumers select only
original H03 score-mask entries. Bounded tests collate original TargetBlock tuple
fields without changing positions/context or selected-byte counts.

The ordered `quantized_matrix_manifest()` names every original W Parameter as
`blocks.{i}.{q,k,v,attention_output,up,down}.weight`. Internal projections alone
are quantized: proxy/core/escalation internal counts are 393,216 / 3,145,728 /
18,874,368 and total counts are 493,056 / 3,347,456 / 19,281,920.

## Runtime context boundary

The default forward is full precision. Call
`configure_quantization({'name': recipe, 'rotation': rotation, 'backward': rule},
runtime=state_by_matrix)` and then `prepare_quantized_views(step, rng)` before a
quantized forward. Runtime entries may contain `previous_scales`,
`previous_zero_groups`, `warm_start` and externally supplied ordinal `codes`.
These tensor handoffs are caller-owned state, outside canonical ModelConfig and
scientific configuration JSON. Codes are numerical ordinals, not export nibbles.
F04 validates recipe/numeric contents at preparation.

Preparation atomically publishes detached numeric contexts for all matrices and
returns the same dictionary when called again for the current step. The caller
must pass frozen scales/initial zero flags for fixed-scale recipes on later
preparations. No automatic scale-history or optimizer-state update occurs. Each
forward rebuilds a fresh surrogate graph and computes `(X@R)@Q(W@R).T`; gradients
flow to original W, whose Parameter identity never changes. There is no retained
microbatch graph, optimizer moment rotation, training RNG draw or production
CPU tensor transfer.

`quantization_contexts` exposes the current cache through a read-only mapping;
its detached tensors, like F04 contexts, must be treated as immutable.
`clear_quantized_views()` releases model ownership of the numeric cache. Clear
and reprepare after changing devices or parameter storage; externally retained
contexts remain caller-owned. The latest cache persists until another step is
prepared, configuration is replaced, or it is cleared. Runtime handoff tensors
remain referenced until configuration is replaced/model disposal.

`forward(..., contexts=override)` consumes a complete alternate context mapping
without touching training cache/RNG/history. `contexts={}` selects full precision.
Evaluation can prepare pure contexts for actual final weights; future functional
fast parameters can supply contexts matching their values and cached scales. H04
does not implement either optimizer transactions or MetaQuant.

## Frozen engineering parity bounds

Loss, logits and each named gradient use atol=rtol=1e-4, inherited from F02's
conservative FP32 operation bounds (prior complete-backward measured gradient
error 3.28e-7). Rotation uses F04's atol=8e-6 / rtol=2e-6. Codes require zero
mismatches; duplicate loss/logits/gradient/code variability must be zero, following
F02/F03-F05 measured duplicates. Bounds were selected before H04 output inspection
and were unchanged after measurements. These bounds cover the declared synthetic
fixtures; they are not neural quality or scientific gate evidence.

`python -m ralup test-harness --suite model --device cpu|mps --output PATH` checks
all exact counts and four loss/gradient fixtures: full-precision proxy, small
identity STE, small full-Hadamard STE and small QuEST trust. All fixtures include
short/one-byte padding and an empty physical row. Actual MPS requires
`PYTORCH_ENABLE_MPS_FALLBACK=0` before interpreter startup; a sandbox blocked
artifact is reported separately from target-device success. Source and evidence
hashes are in [h04-evidence.json](h04-evidence.json).

H04 rejects empty forward tensor shapes. H06 must reject an empty logical
optimizer batch before preparing contexts or updating moments/cursor state;
there is no H04 optimizer entry point. H06 also owns summed NLL accumulation,
one logical target-count division, one clipping operation and transactions.
H05-H08, complete WP02 and scientific gates remain pending. These fixtures make
no model capacity/quality or hardware-throughput claim.
