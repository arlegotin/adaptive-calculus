"""Complete, registered JSON/safetensors snapshots and atomic publication.

Only trusted code registers constructors. Artifacts contain data, never imports,
callables or pickle opcodes. All captured mutable payloads own CPU storage.
"""
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from types import MappingProxyType
import uuid

import torch
from safetensors.torch import load as load_tensors, save as tensor_bytes

from .config import canonical_bytes
from .data.cursor import BatchCursor
from .models.transformer import ByteTransformer, ModelConfig
from .optim.adamw import AdamWUpdateRule
from .optim.base import MethodAdapter, ParameterGroup, SchedulePoint
from .quantization.context import QuantizationContext
from .records import ArtifactRef, RunLedger, WorkLedger, sha256_file
from .train import TrainingState, _rng_state, _restore_rng

SCHEMA = 'ralup.training_snapshot.v1'
ROOT = Path(__file__).resolve().parents[2]


class IntegrityError(ValueError):
    """Unknown, incomplete, malformed or unauthenticated snapshot data."""


class IncompleteStep(IntegrityError):
    """Only completed logical transactions may be published."""


# Exact fields bind schema versions: adding scientific state must fail closed.
_SCHEMAS = {}
_METHODS = {}


def register_state_schema(name, cls, field_names):
    """Register a trusted dataclass constructor and its exact versioned fields."""
    if name in _SCHEMAS or not is_dataclass(cls) or tuple(f.name for f in fields(cls)) != tuple(field_names):
        raise IntegrityError('State schema must be new and declare every dataclass field')
    _SCHEMAS[name] = (cls,tuple(field_names))


def register_method_schema(name, cls, field_names, constructor):
    """Declare every adapter attribute except parameters/groups/rule.

    constructor(parameters, groups, rule) is trusted local code; no constructor
    reference is read from disk. New nested dataclasses need state registration.
    """
    if name in _METHODS or not isinstance(name,str) or not name or not callable(constructor):
        raise IntegrityError('Invalid or duplicate registered method schema')
    names = tuple(field_names)
    if len(set(names)) != len(names) or set(names)&{'parameters','groups','rule'}:
        raise IntegrityError('Invalid method state fields')
    _METHODS[name] = (cls,names,constructor)


for _name,_cls in [('model_config.v1',ModelConfig),('schedule_point.v1',SchedulePoint),
                   ('parameter_group.v1',ParameterGroup),('quantization_context.v1',QuantizationContext)]:
    register_state_schema(_name,_cls,tuple(f.name for f in fields(_cls)))
register_method_schema('ordinary_adapter.v1',MethodAdapter,
                       ('optimizer_states','proposals','snapshot_schema'),MethodAdapter)


@dataclass
class TrainingSnapshot:
    metadata: dict
    tensors: dict[str,torch.Tensor]
    milestone: bool = False


class _Encoder:
    def __init__(self): self.tensors = {}

    def encode(self,value):
        if isinstance(value,torch.Tensor):
            if value.layout != torch.strided or value.is_sparse:
                raise IntegrityError('Only dense registered tensor arrays are supported')
            owned = value.detach().to('cpu').contiguous().clone()
            if (owned.is_floating_point() or owned.is_complex()) and not bool(torch.isfinite(owned).all()):
                raise IntegrityError('Nonfinite tensor state')
            key = f't{len(self.tensors):08d}'
            self.tensors[key] = owned
            return {'type':'tensor','key':key}
        if type(value) is random.Random:
            return {'type':'python_rng','state':self.encode(value.getstate())}
        if type(value) is torch.Generator:
            return {'type':'torch_rng','device':str(value.device),'state':self.encode(value.get_state())}
        if type(value) is ArtifactRef:
            return {'type':'artifact','path':str(value.path.resolve()),'sha256':value.sha256}
        if type(value) in (dict,MappingProxyType):
            if any(type(k) is not str for k in value): raise IntegrityError('State keys must be strings')
            return {'type':'mapping','frozen':type(value) is MappingProxyType,
                    'items':{key:self.encode(value[key]) for key in sorted(value)}}
        if type(value) in (list,tuple):
            return {'type':'tuple' if type(value) is tuple else 'list','items':[self.encode(v) for v in value]}
        for name,(cls,names) in _SCHEMAS.items():
            if type(value) is cls:
                if tuple(vars(value)) != names: raise IntegrityError('Unknown registered dataclass fields')
                return {'type':'schema','schema':name,'fields':{n:self.encode(getattr(value,n)) for n in names}}
        if value is None or type(value) in (str,int,float,bool):
            try: canonical_bytes({'value':value})
            except (TypeError,ValueError) as error: raise IntegrityError('Nonfinite/plain metadata required') from error
            return value
        raise IntegrityError(f'Unregistered state type: {type(value).__name__}')


def _keys(value,names):
    if type(value) is not dict or set(value) != set(names):
        raise IntegrityError('Unknown or missing registered schema fields')


def _decode(value,tensors,used=None):
    if type(value) is not dict:
        canonical_bytes({'value':value})
        return value
    kind = value.get('type')
    if kind == 'tensor':
        _keys(value,('type','key'))
        key = value['key']
        if used is not None: used.add(key)
        return tensors[key].detach().clone()
    if kind == 'artifact':
        _keys(value,('type','path','sha256'))
        return ArtifactRef(value['path'],value['sha256'])
    if kind in ('python_rng','torch_rng'):
        _keys(value,('type','state') if kind == 'python_rng' else ('type','device','state'))
        state = _decode(value['state'],tensors,used)
        # Validation/load need not initialize a target-device generator.
        return {'__rng_kind__':kind,'device':value.get('device'),'state':state}
    if kind == 'mapping':
        _keys(value,('type','frozen','items'))
        if type(value['frozen']) is not bool or type(value['items']) is not dict: raise IntegrityError('Invalid mapping')
        result = {k:_decode(v,tensors,used) for k,v in value['items'].items()}
        return MappingProxyType(result) if value['frozen'] else result
    if kind in ('tuple','list'):
        _keys(value,('type','items'))
        if type(value['items']) is not list: raise IntegrityError('Invalid sequence')
        result = [_decode(v,tensors,used) for v in value['items']]
        return tuple(result) if kind == 'tuple' else result
    if kind == 'schema':
        _keys(value,('type','schema','fields'))
        cls,names = _SCHEMAS[value['schema']]
        _keys(value['fields'],names)
        return cls(**{n:_decode(v,tensors,used) for n,v in value['fields'].items()})
    raise IntegrityError('Unregistered state schema')


def _authenticate(ref):
    try:
        if sha256_file(ref.path) != ref.sha256: raise IntegrityError(f'Artifact hash mismatch: {ref.path}')
    except OSError as error: raise IntegrityError(f'Artifact hash unavailable: {ref.path}') from error


def _references(value):
    if type(value) is ArtifactRef: _authenticate(value)
    elif isinstance(value,(dict,MappingProxyType)):
        if set(value) == {'path','sha256'}: _authenticate(ArtifactRef(**value))
        else:
            for item in value.values(): _references(item)
    elif isinstance(value,(list,tuple)):
        for item in value: _references(item)


def _automatic_identities():
    paths = sorted((ROOT/'src/ralup').rglob('*.py')) + [ROOT/'requirements.lock',ROOT/'artifacts/environment/environment.json']
    return {str(p.relative_to(ROOT)):ArtifactRef(p,sha256_file(p)) for p in paths}


def capture_snapshot(state: TrainingState) -> TrainingSnapshot:
    if not state.checkpointable: raise IncompleteStep('Partial selection/update or failed state cannot be captured')
    expected = {'model','method','optimizer_state','quantizer_state','histories','schedule','cursor','rng_streams',
                'ledger','microbatch','clip_norm','failed','configuration','identities','process_audit','_pending','_work'}
    if {f.name for f in fields(TrainingState)} != expected: raise IntegrityError('TrainingState schema registration is out of date')
    if set(vars(state))|{'_pending','_work'} != expected: raise IntegrityError('Unknown training state fields require registered schema')
    if type(state.model) is not ByteTransformer: raise IntegrityError('Unregistered model schema')
    model_fields = set(vars(torch.nn.Module()))|{'config','_recipe','_runtime','_contexts','_context_step'}
    if set(vars(state.model)) != model_fields: raise IntegrityError('Unknown model fields require registered schema')
    matches = [(n,entry) for n,entry in _METHODS.items() if type(state.method) is entry[0]]
    if len(matches) != 1: raise IntegrityError('Unregistered method schema')
    method_name,(_,method_fields,_) = matches[0]
    if set(vars(state.method)) != set(method_fields)|{'parameters','groups','rule'}:
        raise IntegrityError('Unknown adapter fields require registered method schema')
    if type(state.method.rule) is not AdamWUpdateRule or set(vars(state.method.rule)) != {'betas','eps'}:
        raise IntegrityError('Unknown update rule requires registered schema')
    parameters = dict(state.model.named_parameters())
    if set(parameters) != set(state.method.parameters) or any(state.method.parameters[n] is not p for n,p in parameters.items()):
        raise IntegrityError('Adapter must retain authoritative original Parameters')
    if state.optimizer_state is not state.method.optimizer_states: raise IntegrityError('Optimizer alias mismatch')
    if state.cursor is not None and type(state.cursor) is not BatchCursor: raise IntegrityError('Unregistered cursor')
    if type(state.ledger) is not WorkLedger or set(vars(state.ledger)) != {'stream','coverage','passes','elapsed','events','run_ledger'}:
        raise IntegrityError('Unknown ledger fields require registered schema')
    _references(state.identities); _references(state.configuration)
    identities = {'automatic':_automatic_identities(),'declared':state.identities}
    durable = state.ledger.run_ledger
    run = None if durable is None else {'path':str(durable.path.resolve()),'metadata':ArtifactRef(durable.path/'run.json',sha256_file(durable.path/'run.json'))}
    payload = {'model_config':state.model.config,'weights':dict(state.model.state_dict()),
        'gradients':{n:None if p.grad is None else p.grad for n,p in parameters.items()},
        'model_training':state.model.training,
        'model_cache':{n:getattr(state.model,n) for n in ('_recipe','_runtime','_contexts','_context_step')},
        'method_schema':method_name,'groups':state.method.groups,'rule':vars(state.method.rule),
        'method':{n:getattr(state.method,n) for n in method_fields},
        'quantizer_state':state.quantizer_state,'histories':state.histories,'schedule':state.schedule,
        'cursor':None if state.cursor is None else state.cursor.state,
        'cursor_audit':None if state.cursor is None else {'materialized_blocks':state.cursor.materialized_blocks,'source_bytes_read':state.cursor.source_bytes_read},
        'rng_streams':state.rng_streams,'global_rng':_rng_state(),
        'ledger':{n:getattr(state.ledger,n) for n in ('stream','coverage','passes','elapsed','events')},
        'run_ledger':run,'microbatch':state.microbatch,'clip_norm':state.clip_norm,
        'configuration':state.configuration,'configuration_sha256':hashlib.sha256(canonical_bytes(state.configuration)).hexdigest(),
        'identities':identities,'process_audit':state.process_audit}
    encoder = _Encoder()
    metadata = encoder.encode(payload)
    snapshot = TrainingSnapshot(metadata,encoder.tensors)
    _validated(snapshot)
    return snapshot


_PAYLOAD = {'model_config','weights','gradients','model_training','model_cache','method_schema','groups','rule','method',
            'quantizer_state','histories','schedule','cursor','cursor_audit','rng_streams','global_rng','ledger','run_ledger',
            'microbatch','clip_norm','configuration','configuration_sha256','identities','process_audit'}


def _validated(snapshot):
    try:
        if type(snapshot) is not TrainingSnapshot or type(snapshot.milestone) is not bool: raise IntegrityError('Unregistered snapshot')
        canonical_bytes(snapshot.metadata)
        for tensor in snapshot.tensors.values():
            if tensor.device.type != 'cpu' or tensor.layout != torch.strided or not tensor.is_contiguous():
                raise IntegrityError('Snapshot arrays must be CPU dense contiguous')
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()): raise IntegrityError('Nonfinite tensor state')
        used = set()
        payload = _decode(snapshot.metadata,snapshot.tensors,used)
        if used != set(snapshot.tensors): raise IntegrityError('Unreferenced tensor payload')
        _keys(payload,_PAYLOAD)
        if hashlib.sha256(canonical_bytes(payload['configuration'])).hexdigest() != payload['configuration_sha256']:
            raise IntegrityError('Configuration hash mismatch')
        _references(payload['identities']); _references(payload['configuration']); _references(payload['run_ledger'])
        _keys(payload['model_cache'],('_recipe','_runtime','_contexts','_context_step'))
        _keys(payload['rule'],('betas','eps'))
        _keys(payload['ledger'],('stream','coverage','passes','elapsed','events'))
        # Validate architecture/array identities on meta before any device transfer.
        if type(payload['model_config']) is not ModelConfig: raise IntegrityError('Unregistered model geometry')
        with torch.device('meta'):
            prototype = ByteTransformer(payload['model_config'])
        expected = dict(prototype.named_parameters())
        _keys(payload['weights'],expected)
        _keys(payload['gradients'],expected)
        for name,parameter in expected.items():
            for array in (payload['weights'][name],payload['gradients'][name]):
                if array is not None and (not isinstance(array,torch.Tensor) or array.shape != parameter.shape or array.dtype != parameter.dtype):
                    raise IntegrityError('Model/gradient shape or dtype mismatch')
            if payload['weights'][name] is None: raise IntegrityError('Missing model array')
        _keys(payload['method']['optimizer_states'],expected)
        completed_step = payload['schedule']['step']
        if type(completed_step) is not int or completed_step < 0: raise IntegrityError('Invalid completed step')
        for name,values in payload['method']['optimizer_states'].items():
            if completed_step and not values: raise IntegrityError('Completed step is missing optimizer moments')
            if values:
                _keys(values,AdamWUpdateRule.optimizer_state_fields)
                counter = values['step']
                if counter.shape != torch.Size([]) or counter.dtype != torch.float32 or counter.item() < 0 or counter.item() != int(counter.item()):
                    raise IntegrityError('Invalid optimizer scalar counter')
                for field in ('exp_avg','exp_avg_sq'):
                    if values[field].shape != expected[name].shape or values[field].dtype != torch.float32:
                        raise IntegrityError('Optimizer original-W shape/dtype mismatch')
        _,names,_ = _METHODS[payload['method_schema']]
        _keys(payload['method'],names)
        if payload['method']['proposals']: raise IncompleteStep('Uncommitted proposals in snapshot')
        if payload['cursor'] is not None: BatchCursor(payload['cursor'])
        return payload
    except IntegrityError: raise
    except (KeyError,TypeError,ValueError,RuntimeError,OSError) as error:
        raise IntegrityError(f'Invalid registered snapshot: {error}') from error


def _device_tree(value,device):
    if isinstance(value,torch.Tensor): return value.to(device).clone()
    if isinstance(value,(dict,MappingProxyType)):
        result = {k:_device_tree(v,device) for k,v in value.items()}
        return MappingProxyType(result) if isinstance(value,MappingProxyType) else result
    if isinstance(value,(list,tuple)): return type(value)(_device_tree(v,device) for v in value)
    if is_dataclass(value): return type(value)(**{f.name:_device_tree(getattr(value,f.name),device) for f in fields(value)})
    return value


def restore_snapshot(snapshot: TrainingSnapshot, device: str) -> TrainingState:
    if device not in ('cpu','mps'): raise IntegrityError('Unsupported restore device')
    if device == 'mps' and (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0' or not torch.backends.mps.is_available()):
        raise IntegrityError('Actual MPS requires availability and fallback disabled before startup')
    payload = _validated(snapshot)
    try:
        model = ByteTransformer(payload['model_config']).to(device)
        model.load_state_dict(payload['weights'],strict=True)
        model.train(payload['model_training'])
        for name,value in payload['model_cache'].items(): setattr(model,name,_device_tree(value,device))
        parameters = dict(model.named_parameters())
        if set(parameters) != set(payload['gradients']): raise IntegrityError('Gradient manifest mismatch')
        for name,p in parameters.items():
            grad = payload['gradients'][name]
            if grad is not None and (grad.shape != p.shape or grad.dtype != p.dtype): raise IntegrityError('Gradient shape/dtype mismatch')
            p.grad = None if grad is None else grad.to(device).clone()
        rule = AdamWUpdateRule(**payload['rule'])
        _,_,constructor = _METHODS[payload['method_schema']]
        method = constructor(parameters,payload['groups'],rule)
        for name,value in payload['method'].items(): setattr(method,name,_device_tree(value,device))
        # Noncapturable scalar counters stay CPU; moments remain original-W device.
        if set(method.optimizer_states) != set(parameters): raise IntegrityError('Optimizer manifest mismatch')
        for name,values in method.optimizer_states.items():
            if values:
                _keys(values,rule.optimizer_state_fields)
                values['step'] = values['step'].cpu().clone()
        streams = {}
        for name,rng in payload['rng_streams'].items():
            _keys(rng,('__rng_kind__','device','state'))
            if rng['__rng_kind__'] == 'python_rng':
                generator = random.Random(); generator.setstate(rng['state'])
            else:
                generator = torch.Generator(device=rng['device']); generator.set_state(rng['state'])
            streams[name] = generator
        ledger = WorkLedger()
        for name,value in payload['ledger'].items(): setattr(ledger,name,value)
        if payload['run_ledger'] is not None: ledger.run_ledger = RunLedger(Path(payload['run_ledger']['path']))
        state = TrainingState(model,method,quantizer_state=_device_tree(payload['quantizer_state'],device),
            histories=_device_tree(payload['histories'],device),schedule=payload['schedule'],
            cursor=None if payload['cursor'] is None else BatchCursor(payload['cursor']),rng_streams=streams,ledger=ledger,
            microbatch=payload['microbatch'],clip_norm=payload['clip_norm'],configuration=payload['configuration'],
            identities=payload['identities']['declared'],process_audit=payload['process_audit'])
        if state.cursor is not None:
            for name,value in payload['cursor_audit'].items(): setattr(state.cursor,name,value)
        state.process_audit.append({'kind':'resume','created_utc':datetime.now(timezone.utc).isoformat(),
                                    'pid':os.getpid(),'process_elapsed_seconds':0.,'past_step_seconds':ledger.elapsed['seconds']})
        _restore_rng(payload['global_rng'])
        return state
    except IntegrityError: raise
    except (KeyError,ValueError,TypeError,RuntimeError) as error: raise IntegrityError(f'Restore rejected: {error}') from error


def _flush_write(path,data):
    with path.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def _fsync_directory(path):
    descriptor = os.open(path,os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _manifest(snapshot):
    data = tensor_bytes(snapshot.tensors)
    return {'schema':SCHEMA,'metadata':snapshot.metadata,'milestone':snapshot.milestone,
            'tensor_sha256':hashlib.sha256(data).hexdigest(),
            'tensors':{key:{'shape':list(t.shape),'dtype':str(t.dtype)} for key,t in snapshot.tensors.items()}},data


def save_snapshot(snapshot, destination: Path) -> ArtifactRef:
    _validated(snapshot)
    destination = Path(destination)
    missing = []
    parent = destination
    while not parent.exists():
        missing.append(parent); parent = parent.parent
    for directory in reversed(missing):
        directory.mkdir(); _fsync_directory(directory.parent)
    # One writer, immutable generations; latest is the only publication point.
    staging = Path(tempfile.mkdtemp(prefix='.pending-',dir=destination))
    generation = destination/f'g-{time.time_ns():020d}-{uuid.uuid4().hex}'
    pointer = destination/f'.latest-{uuid.uuid4().hex}'
    committed = False
    try:
        manifest,data = _manifest(snapshot)
        manifest_bytes = canonical_bytes(manifest)
        _flush_write(staging/'tensors.safetensors',data)
        _flush_write(staging/'manifest.json',manifest_bytes)
        _fsync_directory(staging)
        os.replace(staging,generation)
        _fsync_directory(destination)
        ref = ArtifactRef(generation/'manifest.json',hashlib.sha256(manifest_bytes).hexdigest())
        # Read/validate the actual published arrays before exposing a reference.
        load_snapshot(ref)
        _flush_write(pointer,canonical_bytes({'path':str(ref.path.resolve()),'sha256':ref.sha256}))
        os.replace(pointer,destination/'latest.json')
        committed = True
        _fsync_directory(destination)
    except BaseException:
        # If pointer replacement happened, its valid generation must survive even
        # a subsequent parent-fsync error. Recovery validates actual data again.
        if not committed and generation.exists(): shutil.rmtree(generation)
        raise
    finally:
        if staging.exists(): shutil.rmtree(staging)
        if pointer.exists(): pointer.unlink()
    _flush_write(generation/'published.json',canonical_bytes({'path':str(ref.path.resolve()),'sha256':ref.sha256}))
    _fsync_directory(generation)
    _retain(destination)
    return ref


def load_snapshot(reference: ArtifactRef) -> TrainingSnapshot:
    try:
        _authenticate(reference)
        manifest = json.loads(reference.path.read_bytes())
        _keys(manifest,('schema','metadata','milestone','tensor_sha256','tensors'))
        if manifest['schema'] != SCHEMA: raise IntegrityError('Unknown snapshot schema')
        data = (reference.path.parent/'tensors.safetensors').read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['tensor_sha256']: raise IntegrityError('Tensor payload hash mismatch')
        tensors = load_tensors(data)
        if set(tensors) != set(manifest['tensors']): raise IntegrityError('Tensor manifest mismatch')
        for key,tensor in tensors.items():
            spec = manifest['tensors'][key]; _keys(spec,('shape','dtype'))
            if list(tensor.shape) != spec['shape'] or str(tensor.dtype) != spec['dtype']:
                raise IntegrityError('Tensor shape/dtype mismatch')
        snapshot = TrainingSnapshot(manifest['metadata'],tensors,manifest['milestone'])
        _validated(snapshot)
        return snapshot
    except IntegrityError: raise
    except Exception as error: raise IntegrityError(f'Checkpoint rejected: {error}') from error


def _valid_generations(destination):
    result = []
    for generation in sorted(Path(destination).glob('g-*'),reverse=True):
        manifest = generation/'manifest.json'
        try:
            ref = ArtifactRef(**json.loads((generation/'published.json').read_bytes()))
            if ref.path.resolve() != manifest.resolve(): raise IntegrityError('Publication reference mismatch')
            snapshot = load_snapshot(ref)
            result.append((ref,snapshot.milestone))
        except (IntegrityError,OSError,ValueError,TypeError): continue
    return result


def recover_snapshot(destination: Path) -> ArtifactRef:
    destination = Path(destination)
    try:
        ref = ArtifactRef(**json.loads((destination/'latest.json').read_bytes()))
        if ref.path.parent.parent.resolve() != destination.resolve(): raise IntegrityError('Pointer outside snapshot store')
        load_snapshot(ref)
        return ref
    except (OSError,ValueError,TypeError):
        valid = _valid_generations(destination)
        if not valid: raise IntegrityError('No valid snapshot generation to recover')
        return valid[0][0]


def _retain(destination):
    valid = _valid_generations(destination)
    for ref,milestone in valid[2:]:
        if not milestone: shutil.rmtree(ref.path.parent)
    _fsync_directory(destination)


def exact_cpu_state_hash(state: TrainingState) -> str:
    """Scientific replay hash; elapsed/process/run metadata are audit costs.

    Captured storage is always CPU; same-device MPS replay inherits duplicate
    bounds separately. This hash makes no CPU-versus-MPS parity claim.
    """
    payload = _validated(capture_snapshot(state))
    payload.pop('process_audit'); payload.pop('run_ledger'); payload.pop('cursor_audit')
    payload['ledger'].pop('elapsed')
    for row in payload['ledger']['events']: row.pop('elapsed_seconds',None)
    encoder = _Encoder()
    # _decode leaves RNG states as plain data, suitable for deterministic hashing.
    metadata = encoder.encode(payload)
    return hashlib.sha256(canonical_bytes(metadata)+tensor_bytes(encoder.tensors)).hexdigest()
