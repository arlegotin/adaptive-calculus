# E06 packed export contract

Version `ralup.packed_e2m1.v1`: eight-byte `RALUPE2`/version1 magic, uint64
little-endian canonical-JSON length, canonical finite JSON, contiguous raw
blobs. Header <=4MiB, container <=128MiB, trusted S11 transformer geometry
<=6 layers/512 width/20M parameters. No artifact-selected constructors,
imports, hooks, pickle, compression, or alternate attention implementation.

Ordinals [0,14] map to hardware nibbles
`15,14,13,12,11,10,9,0,1,2,3,4,5,6,7`. Matrix storage is row-major, first
element in the low nibble. All15 numerical levels include full±6. Zero
exports as nibble0; nibble8 decodes positive zero. An odd final element has
unused high nibble0, explicitly counted. Each matrix is packed independently,
so odd-width geometry includes its per-matrix padding in the analytical ledger.
Scales and auxiliary tensors retain exact little-endian FP32 bytes.

Export authenticates H07 checkpoint bytes, producer archive, exact current
whole source, model/configuration/runtime, and registered owner state before
publishing. It does not migrate old full states. Final read-only H06
evaluation_contexts recompute dynamic/QuEST scales from final weights, retain
frozen fixed warm-start scales, and use authoritative physical learned scales.
Learned state validation requires exact exp(log) on its declared live producing
backend, complete physical/runtime/quantizer ownership, and full±6. CPU
artifact validation never substitutes CPU exponentials for an MPS producer.
Cross-backend learned full-state restoration remains rejected. Portable
decoding reconstructs from stored physical scales using the validated FP32
integer helper; it never regenerates exponentials.

Each payload descriptor binds shape, dtype, offset, length and SHA256. The
container ArtifactRef authenticates all metadata. Final context identity binds
exact code/scale descriptors, recipe, rotation, grouping, step, scale metadata,
evaluation context and scale owner. Source checkpoint identity, archived
producer identity, source configuration hash and measured parity are retained.
Analytical payload/fraction and actual container bytes are distinct. Core row
payload is1,572,864+806,912+36,864=2,416,640B, fraction3,145,728/3,347,456.

Prospectively frozen E06 numerical contract: zero code/scale/storage-value
mismatches; exact same-backend FP32 logits, summed H06 loss and BPB (atol=rtol=0).
For two new actual MPS compositions only (dynamic full Hadamard and learned
identity), the portable CPU comparison inherits H04 logits atol=rtol=1e-4;
same-backend decoded forward remains exact. Native processes require actual
MPS availability, fallback0 before imports, normal escalated execution and
current runtime/source receipts. This finite export composition check is not
the accepted Cartesian neural gate, research training or native-speed evidence.

The common H06 scorer and transformer execute decoded FP32 contexts. Loaded
objects are inference-only; precision changes fail before mutating the model.
Arbitrary context overrides are rejected. Export verifies reference/decoded
logits and losses on supplied nonprotected byte blocks before publication,
recording block hash/count/targets, errors, backend and forward/pass costs.
The generated CLI probe is independently authored synthetic byte content.
It opens no official test data. Failed parity publishes no artifact. Failed
partial writes remain unusable unless their complete authenticated format is
verified; existing destinations are immutable.

The initialized core CLI fixture demonstrates packed mechanics and measured
file size only. No genuine selected C/LR/Bstar final research model exists;
owner E01/reopened WP06 must produce one, then export/final artifact parity
must be rerun. No score, trained-performance, compression-as-speed or deployed
FP4 memory claim follows. Unknown enclosing/common engineering cost stays null.

E06 administrative closure has an explicit root-authorized Git-volume location
exception. NEW receipts live at the original checkout's ignored
`artifacts/ralup-administration/20261012/e06-v1`, outside the linked worktree.
The typed `ralup.e06_compact_administration.v1` startup binds both roots, its
ownership marker, original immutable Data numerical/source/test archive, and
separate NEW administrative sources. The complete Data inventory sidecar is
`gzip_json_utf8_mtime0`, with both raw/compressed byte lengths and SHA256 hashes.
Authenticate each compressed file, decompress it, then authenticate the exact
raw bytes before reading its full inventory. This is administrative encoding;
H07 runtime/source compatibility and scientific authority remain unchanged.
E07 must discover and preserve both roots through future workspace cleanup.
The transaction owns zero Data writes, records the existing Data shortfall and
prior HOLDs, and requires a newly measured <=16MiB Git physical footprint and
the stronger1GiB Git floor. It does not restore the failed prior Data forecast.
