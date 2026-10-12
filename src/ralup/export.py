"""S35 data-only packed E2M1 deployment artifacts and reference-forward proof.

Numerical ordinals are converted to OCP S EE M nibbles, low nibble first,
row-major. Zero exports as nibble 0; nibble 8 decodes to positive zero.
Unused high nibbles are zero. Scales/auxiliaries are exact little-endian FP32.
This portable decoder executes ordinary FP32 arithmetic, not native FP4.
"""
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from types import MappingProxyType

import torch
from torch import Tensor, nn

from .config import canonical_bytes
from .data.batches import LogicalBatch
from .evaluate import evaluation_contexts, score_blocks, validate_batch
from .models.transformer import ByteTransformer, ModelConfig
from .quantization.context import QuantizationContext
from .quantization.formats import E2M1_NIBBLES, _preserved_reconstruction
from .quantization.scales import grouping_for
from .records import ArtifactRef, sha256_file

MAGIC = b'RALUPE2\x01'
SCHEMA = 'ralup.packed_e2m1.v1'
MAX_BYTES = 128*1024**2
MAX_HEADER = 4*1024**2
ENCODING = {'format':'E2M1_S_EE_M', 'order':'row_major_low_nibble_first',
            'zero':'export_positive_0_decode_8_as_positive_0', 'odd_padding':'high_nibble_0',
            'fp32':'little_endian', 'execution':'portable_reference_FP32_W4A32'}


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError('Packed export requires exact versioned fields')


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Nonnegative integer count/offset required')
    return value


def pack_e2m1_codes(codes: Tensor) -> bytes:
    """Pack validated uint8 numerical ordinals [0,14], not encoding nibbles."""
    if codes.dtype != torch.uint8:
        raise TypeError('Packed E2M1 requires uint8 numerical ordinals')
    if not bool((codes <= 14).all()):
        raise ValueError('E2M1 ordinal outside [0,14]')
    flat = codes.detach().cpu().reshape(-1).long()
    nibbles = torch.tensor(E2M1_NIBBLES, dtype=torch.uint8)[flat]
    if nibbles.numel()%2:
        nibbles = torch.cat((nibbles, torch.zeros(1, dtype=torch.uint8)))
    return bytes((nibbles[::2] | (nibbles[1::2] << 4)).tolist())


def unpack_e2m1_codes(payload: bytes, count: int) -> Tensor:
    """Decode both signed zeros to canonical ordinal 7, checking odd padding."""
    _integer(count)
    if type(payload) is not bytes or len(payload) != (count+1)//2:
        raise ValueError('Packed E2M1 length/count mismatch')
    if count%2 and payload[-1] & 0xf0:
        raise ValueError('Nonzero unused E2M1 padding nibble')
    if not count:
        return torch.empty(0, dtype=torch.uint8)
    raw = torch.tensor(list(payload), dtype=torch.uint8)
    nibbles = torch.stack((raw & 15, raw >> 4), dim=1).reshape(-1)[:count].long()
    return torch.tensor((7,8,9,10,11,12,13,14,7,6,5,4,3,2,1,0), dtype=torch.uint8)[nibbles]


def analytical_payload(config: ModelConfig, grouping: str) -> dict:
    """Derived byte ledger excludes all versioned container metadata."""
    if grouping not in ('row', 'tensor'):
        raise ValueError('Explicit row/tensor grouping required')
    l, d = config.layers, config.width
    internal = 12*l*d*d
    auxiliary = 4*l*d+2*d+2*257*d+256*d
    scales = 9*l*d if grouping == 'row' else 6*l
    code_bytes = 4*l*((d*d+1)//2)+4*l*d*d
    return {'code_bytes':code_bytes, 'auxiliary_bytes':4*auxiliary,
            'scale_bytes':4*scales, 'payload_bytes':code_bytes+4*auxiliary+4*scales,
            'scale_count':scales, 'auxiliary_parameters':auxiliary,
            'quantized_parameters':internal, 'total_parameters':internal+auxiliary,
            'quantized_fraction':internal/(internal+auxiliary), 'grouping':grouping}


def _config(value):
    _keys(value, ('layers','width','heads'))
    config = ModelConfig(**value)
    if config.layers > 6 or config.width > 512 or analytical_payload(config, 'row')['total_parameters'] > 20_000_000:
        raise ValueError('Unsupported bounded model geometry')
    return config


def _fp32_bytes(value):
    if sys.byteorder != 'little':
        raise ValueError('FP32 exporter requires little-endian host')
    if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise ValueError('Finite exact FP32 tensors required')
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


class DecodedExport(nn.Module):
    """Inference-only common transformer with exact decoded storage contexts."""
    _recipe = None

    def __init__(self, model, contexts, metadata):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.contexts = MappingProxyType(contexts)
        self.metadata = metadata

    def _apply(self, fn, recurse=True):
        converted = {}
        for name,c in self.contexts.items():
            values = {k:fn(getattr(c,k)) for k in ('scales','codes','storage_values','storage_u','gradient_mask','zero_groups')}
            if any(values[k].dtype != torch.float32 for k in ('scales','storage_values','storage_u','gradient_mask')):
                raise ValueError('Decoded inference preserves exact FP32 precision')
            converted[name] = replace(c, **values)
        super()._apply(fn, recurse)
        self.contexts = MappingProxyType(converted)
        return self

    def forward(self, inputs, positions, padding_mask=None, *, contexts=None):
        if contexts:
            raise ValueError('Decoded artifact owns its exact inference contexts')
        return self.model(inputs, positions, padding_mask, contexts=self.contexts)


def _decode(raw, *, require_parity=True):
    if len(raw) > MAX_BYTES or len(raw) < 16 or raw[:8] != MAGIC:
        raise ValueError('Invalid bounded packed E2M1 container')
    length = struct.unpack('<Q', raw[8:16])[0]
    if length > MAX_HEADER or 16+length > len(raw):
        raise ValueError('Invalid bounded metadata length')
    def pairs(items):
        value = {}
        for key,item in items:
            if key in value: raise ValueError('Duplicate JSON metadata key')
            value[key] = item
        return value
    metadata = json.loads(raw[16:16+length], object_pairs_hook=pairs)
    if canonical_bytes(metadata) != raw[16:16+length]:
        raise ValueError('Canonical finite JSON metadata required')
    _keys(metadata, ('schema','encoding','model_config','matrices','auxiliaries','analytical_payload',
        'actual_container_bytes','source_checkpoint','producer_sha256','source_configuration_sha256',
        'evaluation_context','context_sha256','scale_owner','parity','scientific_final_model','limits'))
    if metadata['schema'] != SCHEMA or metadata['encoding'] != ENCODING or metadata['scientific_final_model'] is not False:
        raise ValueError('Unknown packed export schema/encoding/scope')
    if _integer(metadata['actual_container_bytes']) != len(raw):
        raise ValueError('Actual container size mismatch')
    _keys(metadata['source_checkpoint'], ('path','sha256'))
    ArtifactRef(**metadata['source_checkpoint'])
    producer = metadata['producer_sha256']
    if type(producer) is not dict or not {'requirements.lock','artifacts/environment/environment.json'} <= producer.keys():
        raise ValueError('Producer source identity missing')
    for name,digest in producer.items():
        if name not in ('requirements.lock','artifacts/environment/environment.json') and not (
                name.startswith('src/ralup/') and name.endswith('.py') and '..' not in Path(name).parts):
            raise ValueError('Unknown producer source path')
        ArtifactRef(name,digest)
    ArtifactRef('.',metadata['source_configuration_sha256'])
    owner = metadata['scale_owner']
    _keys(owner, ('kind','backend','contract','derivation'))
    if owner['backend'] not in ('cpu','mps') or owner['kind'] not in ('learned_physical_cache','shared_H06_finalization'):
        raise ValueError('Unknown physical-scale owner/backend')
    expected_derivation = 'declared_backend_exp_log_bitwise' if owner['kind']=='learned_physical_cache' else 'declared_S14_recipe_input'
    if owner['derivation'] != expected_derivation:
        raise ValueError('Unknown scale derivation')
    if owner['kind']=='learned_physical_cache':
        if (type(owner['contract']) is not dict or owner['contract'].get('range') != [-6,6]
                or owner['contract'].get('export_precision') != 'float32'):
            raise ValueError('Full-range FP32 learned scale contract required')
    elif owner['contract'] is not None:
        raise ValueError('Shared finalization has no learned-scale contract')
    evaluation = metadata['evaluation_context']
    if type(evaluation) is not dict or evaluation.get('partition') not in ('fixture','development','meta-train','meta-development','target-training'):
        raise ValueError('Nonprotected parity evaluation context required')
    if require_parity:
        parity = metadata['parity']
        _keys(parity, ('contract','backend','atol','rtol','code_mismatches','logits_max_error','nll_sum','bpb',
            'target_count','block_count','blocks_sha256','costs'))
        if (parity['contract'] != 'exact_same_backend_FP32_codes_scales_logits_H06_loss_BPB.v1'
                or parity['backend'] != owner['backend'] or any(parity[k] != 0 for k in ('atol','rtol','code_mismatches','logits_max_error'))):
            raise ValueError('Exact decoded parity contract failed')
        _integer(parity['target_count'],1); _integer(parity['block_count'],1)
        ArtifactRef('.',parity['blocks_sha256'])
        if (type(parity['nll_sum']) not in (int,float) or not math.isfinite(parity['nll_sum']) or parity['nll_sum'] < 0
                or parity['bpb'] != parity['nll_sum']/(math.log(2)*parity['target_count'])):
            raise ValueError('Finite H06 loss/BPB parity evidence required')
        costs = parity['costs']
        _keys(costs, ('reference_logit_forwards','decoded_logit_forwards','reference_loss_forwards','decoded_loss_forwards',
            'processed_loss_targets','additional_forward_context_tokens','backward_targets','hessian_passes',
            'common_training_cost','enclosing_engineering_seconds'))
        for k in ('reference_logit_forwards','decoded_logit_forwards','reference_loss_forwards','decoded_loss_forwards'):
            if _integer(costs[k],1) != parity['block_count']: raise ValueError('Parity forward cost mismatch')
        if (_integer(costs['processed_loss_targets']) != 2*parity['target_count']
                or _integer(costs['backward_targets']) != 0 or _integer(costs['hessian_passes']) != 0
                or _integer(costs['additional_forward_context_tokens'],1) < 4*parity['target_count']
                or costs['enclosing_engineering_seconds'] is not None):
            raise ValueError('Parity target/pass/unknown cost mismatch')
    config = _config(metadata['model_config'])
    with torch.device('meta'):
        prototype = ByteTransformer(config)
    manifest = prototype.quantized_matrix_manifest()
    matrices = metadata['matrices']
    if type(matrices) is not list or [m.get('name') for m in matrices] != [m['name'] for m in manifest]:
        raise ValueError('Internal matrix manifest mismatch')
    if metadata['context_sha256'] != _hash(canonical_bytes({'matrices':matrices, 'evaluation_context':metadata['evaluation_context'], 'scale_owner':metadata['scale_owner']})):
        raise ValueError('Final context identity mismatch')
    body = raw[16+length:]
    offset = 0

    def blob(descriptor, shape, dtype, count):
        nonlocal offset
        _keys(descriptor, ('shape','dtype','offset','bytes','sha256'))
        needed = (count+1)//2 if dtype == 'e2m1_nibbles' else 4*count
        if (descriptor['shape'] != shape or descriptor['dtype'] != dtype
                or _integer(descriptor['offset']) != offset or _integer(descriptor['bytes']) != needed):
            raise ValueError('Shape/dtype/contiguous byte payload mismatch')
        data = body[offset:offset+needed]
        if len(data) != needed or _hash(data) != descriptor['sha256']:
            raise ValueError('Payload checksum/length mismatch')
        offset += needed
        if dtype == 'e2m1_nibbles':
            return unpack_e2m1_codes(data, count).reshape(shape)
        value = torch.frombuffer(bytearray(data), dtype=torch.float32).clone().reshape(shape)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite FP32 auxiliary/scale payload')
        return value

    contexts = {}
    grouping = None
    for item, expected in zip(matrices, manifest):
        _keys(item, ('name','shape','rotation','recipe','grouping','step','scale_metadata','padding_count','codes','scales'))
        if item['shape'] != expected['shape'] or item['rotation'] not in ('identity','full_hadamard'):
            raise ValueError('Matrix shape/rotation mismatch')
        width = item['shape'][1]
        if item['rotation'] == 'full_hadamard' and width & (width-1):
            raise ValueError('Full Hadamard requires power-of-two input width')
        declared = grouping_for(item['recipe'])
        if item['grouping'] != declared or grouping not in (None,declared):
            raise ValueError('Unsupported mixed/inconsistent scale grouping')
        grouping = declared
        count = math.prod(item['shape'])
        if _integer(item['padding_count']) != count%2:
            raise ValueError('Padding count mismatch')
        codes = blob(item['codes'], item['shape'], 'e2m1_nibbles', count)
        scale_shape = [item['shape'][0],1] if grouping == 'last_axis_row' else [1,1]
        scales = blob(item['scales'], scale_shape, 'float32_le', math.prod(scale_shape))
        if not bool((scales > 0).all()):
            raise ValueError('Physical scales must be finite positive')
        values = _preserved_reconstruction(codes.long(), scales.view(torch.int32))
        contexts[item['name']] = QuantizationContext(scales,codes,values,values.clone(),torch.zeros_like(values),
            torch.zeros(scale_shape,dtype=torch.bool),None,item['rotation'],grouping,'ste',item['recipe'],
            _integer(item['step']),MappingProxyType(item['scale_metadata']))
    ledger = analytical_payload(config, 'row' if grouping == 'last_axis_row' else 'tensor')
    if canonical_bytes(metadata['analytical_payload']) != canonical_bytes(ledger):
        raise ValueError('Analytical byte ledger mismatch')
    internal = {m['name'] for m in manifest}
    auxiliaries = metadata['auxiliaries']
    expected_aux = {n:list(p.shape) for n,p in prototype.named_parameters() if n not in internal}
    if type(auxiliaries) is not list or [a.get('name') for a in auxiliaries] != list(expected_aux):
        raise ValueError('FP32 auxiliary manifest mismatch')
    weights = {m['name']:torch.zeros(m['shape'],dtype=torch.float32) for m in manifest}
    for auxiliary in auxiliaries:
        _keys(auxiliary, ('name','tensor'))
        name = auxiliary['name']; shape = expected_aux[name]
        weights[name] = blob(auxiliary['tensor'], shape, 'float32_le', math.prod(shape))
    if offset != len(body) or offset != ledger['payload_bytes']:
        raise ValueError('Trailing or missing tensor payload')
    # Meta construction plus to_empty initializes no random numbers.
    model = prototype.to_empty(device='cpu')
    model.load_state_dict(weights, strict=True)
    return DecodedExport(model, contexts, metadata)


def load_export(reference: ArtifactRef) -> DecodedExport:
    if reference.path.stat().st_size > MAX_BYTES:
        raise ValueError('Packed artifact exceeds bounded size')
    raw = reference.path.read_bytes()
    if _hash(raw) != reference.sha256:
        raise ValueError('Packed artifact checksum mismatch')
    try:
        return _decode(raw)
    except (KeyError, TypeError, RuntimeError, json.JSONDecodeError) as error:
        raise ValueError(f'Packed export rejected: {error}') from error


def _serialize(metadata, body):
    for _ in range(16):
        header = canonical_bytes(metadata)
        size = 16+len(header)+len(body)
        if metadata['actual_container_bytes'] == size:
            if size > MAX_BYTES or len(header) > MAX_HEADER:
                raise ValueError('Packed export exceeds bounded container size')
            return MAGIC+struct.pack('<Q',len(header))+header+body
        metadata['actual_container_bytes'] = size
    raise ValueError('Container size did not converge')


@torch.no_grad()
def export_model(model, context: dict, destination: Path) -> ArtifactRef:
    """Authenticate source, finalize via H06, verify decoded forward, then publish.

    context requires source_checkpoint ArtifactRef, a nonprotected
    evaluation_context, and independently supplied parity_blocks. This route
    establishes export mechanics only; genuine final-arm authority is absent.
    """
    from .state import load_snapshot, restore_snapshot, _validated, _validate_model_owners
    from .train import _rng_state, _restore_rng
    _keys(context, ('source_checkpoint','evaluation_context','parity_blocks'))
    destination = Path(destination)
    if destination.exists(): raise FileExistsError('Immutable export destination exists')
    if type(model) is not ByteTransformer: raise ValueError('Only trusted ByteTransformer exports supported')
    _validate_model_owners(model)
    device = next(model.parameters()).device.type
    snapshot = load_snapshot(context['source_checkpoint'])
    payload = _validated(snapshot)
    rng = _rng_state()
    try:
        state = restore_snapshot(snapshot,device)
        if state.model.config != model.config or state.model._recipe != model._recipe or any(
                not torch.equal(p,model.state_dict()[n]) for n,p in state.model.state_dict().items()):
            raise ValueError('Export model differs from source checkpoint')
        # Runtime scale/input handoffs must match the authenticated producer.
        from .state import _Encoder
        def runtime_hash(value):
            e = _Encoder(); data = e.encode(value)
            return _hash(canonical_bytes({'metadata':data,'tensors':{n:_hash(t.contiguous().view(torch.uint8).numpy().tobytes()) for n,t in e.tensors.items()}}))
        if runtime_hash(state.model._runtime) != runtime_hash(model._runtime):
            raise ValueError('Model runtime differs from source checkpoint')
        contexts = evaluation_contexts(state.model, context['evaluation_context'])
        if not contexts: raise ValueError('Packed export requires configured quantized matrices')
        blocks = list(context['parity_blocks'])
        validate_batch(LogicalBatch(blocks,sum(sum(b.score_mask) for b in blocks)))
        body = bytearray()
        def append(raw, shape, dtype):
            record = {'shape':list(shape),'dtype':dtype,'offset':len(body),'bytes':len(raw),'sha256':_hash(raw)}
            body.extend(raw); return record
        matrices = []
        for n,c in contexts.items():
            codes = append(pack_e2m1_codes(c.codes),c.codes.shape,'e2m1_nibbles')
            scales = append(_fp32_bytes(c.scales),c.scales.shape,'float32_le')
            matrices.append({'name':n,'shape':list(c.codes.shape),'rotation':c.rotation,'recipe':c.recipe,
                'grouping':c.grouping,'step':c.step,'scale_metadata':dict(c.metadata),'padding_count':c.codes.numel()%2,
                'codes':codes,'scales':scales})
        auxiliaries = [{'name':n,'tensor':append(_fp32_bytes(p),p.shape,'float32_le')}
            for n,p in state.model.named_parameters() if n not in contexts]
        learned = payload['method_schema'] == 'learned_scale_adapter.v1'
        scale_owner = {'kind':'learned_physical_cache' if learned else 'shared_H06_finalization',
            'backend':device,'contract':state.method.scale_contract if learned else None,
            'derivation':'declared_backend_exp_log_bitwise' if learned else 'declared_S14_recipe_input'}
        metadata = {'schema':SCHEMA,'encoding':ENCODING,'model_config':asdict(model.config),
            'matrices':matrices,'auxiliaries':auxiliaries,
            'analytical_payload':analytical_payload(model.config,'row' if matrices[0]['grouping']=='last_axis_row' else 'tensor'),
            'actual_container_bytes':0, 'source_checkpoint':{'path':str(context['source_checkpoint'].path),'sha256':context['source_checkpoint'].sha256},
            'producer_sha256':{n:r.sha256 for n,r in payload['identities']['automatic'].items()},
            'source_configuration_sha256':payload['configuration_sha256'], 'evaluation_context':context['evaluation_context'],
            'scale_owner':scale_owner,'context_sha256':_hash(canonical_bytes({'matrices':matrices,'evaluation_context':context['evaluation_context'],'scale_owner':scale_owner})),
            'parity':{},'scientific_final_model':False,
            'limits':['export mechanics; no selected research final model (owner E01/reopened WP06)',
                      'portable FP32 decoded execution; no native FP4 speed or memory claim',
                      'parity only on the recorded nonprotected blocks; no protected scoring']}
        decoded = _decode(_serialize(metadata,body),require_parity=False).to(device)
        mismatches = sum(int((decoded.contexts[n].codes != c.codes).sum()) for n,c in contexts.items())
        max_error = 0.
        for b in blocks:
            inputs = torch.tensor([b.inputs],device=device); positions = torch.tensor([b.positions],device=device)
            left = state.model(inputs,positions,contexts=contexts); right = decoded(inputs,positions)
            torch.testing.assert_close(left,right,atol=0,rtol=0)
            max_error = max(max_error,float((left-right).abs().max()))
        reference = score_blocks(state.model,blocks,evaluation_context=context['evaluation_context'])
        actual = score_blocks(decoded,blocks,evaluation_context=context['evaluation_context'])
        if actual != reference or mismatches: raise ValueError('Exact decoded logits/loss/BPB parity failed')
        block_identity = [{k:list(v) if isinstance(v,tuple) else v for k,v in asdict(b).items()} for b in blocks]
        metadata['parity'] = {'contract':'exact_same_backend_FP32_codes_scales_logits_H06_loss_BPB.v1',
            'backend':device,'atol':0.,'rtol':0.,'code_mismatches':mismatches,'logits_max_error':max_error,
            'nll_sum':actual.nll_sum,'bpb':actual.bpb,'target_count':actual.target_count,
            'block_count':len(blocks),'blocks_sha256':_hash(canonical_bytes({'blocks':block_identity})),
            'costs':{'reference_logit_forwards':len(blocks),'decoded_logit_forwards':len(blocks),
                'reference_loss_forwards':len(blocks),'decoded_loss_forwards':len(blocks),
                'processed_loss_targets':2*actual.target_count,'additional_forward_context_tokens':4*sum(len(b.inputs) for b in blocks),
                'backward_targets':0,'hessian_passes':0,'common_training_cost':'source checkpoint ledger, not inferred from export',
                'enclosing_engineering_seconds':None}}
        raw = _serialize(metadata,body)
        _decode(raw)  # Verify final metadata/representation before publication.
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        return ArtifactRef(destination,_hash(raw))
    finally:
        _restore_rng(rng)


def export_checkpoint(checkpoint: Path, destination: Path) -> ArtifactRef:
    """CLI mechanics probe on generated fixture bytes, never protected data."""
    from .state import load_snapshot, restore_snapshot, _validated
    from .train import _rng_state, _restore_rng
    from .data.articles import Article
    from .data.blocks import tile_article
    reference = ArtifactRef(checkpoint,sha256_file(checkpoint))
    snapshot = load_snapshot(reference); payload = _validated(snapshot)
    device = payload['method']['scale_backend'] if payload['method_schema']=='learned_scale_adapter.v1' else 'cpu'
    rng = _rng_state()
    try:
        state = restore_snapshot(snapshot,device or 'cpu')
        raw = b'\x00E06 export CLI mechanics fixture\xff\n'
        return export_model(state.model,{'source_checkpoint':reference,
            'evaluation_context':{'partition':'fixture','seed':606,'step':state.schedule['step']},
            'parity_blocks':tile_article(Article(raw,[],_hash(raw)))},destination)
    finally:
        _restore_rng(rng)
