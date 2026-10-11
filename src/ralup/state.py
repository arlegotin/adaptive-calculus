"""Complete, registered JSON/safetensors snapshots and atomic publication.

Only trusted code registers constructors. Artifacts contain data, never imports,
callables or pickle opcodes. All captured mutable payloads own CPU storage.
"""
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
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
from .optim.muon import MuonUpdateRule
from .optim.muon_plus import MuonPlusUpdateRule
from .optim.soap import SoapUpdateRule,TIMING_FIELDS
from .optim.madam import MadamUpdateRule
from .optim.matrix_adapter import MatrixMethodAdapter
from .baselines.ewgs import EwgsAdapter,ADAPTER_FIELDS,trusted_constructor,validate_owned_state
from .baselines.trs import TrsState,TrsBackboneAdapter,ADAPTER_FIELDS as TRS_FIELDS,trusted_constructor as trs_constructor,validate_owned_state as validate_trs_state
from .baselines.qar import QarAdjacentAdapter,ADAPTER_FIELDS as QAR_FIELDS,trusted_constructor as qar_constructor,validate_owned_state as validate_routing_state
from .baselines.rounding import RoundingAdapter,trusted_constructor as rounding_constructor
from .baselines.noise import NoiseAdapter,ADAPTER_FIELDS as NOISE_FIELDS,trusted_constructor as noise_constructor,validate_owned_state as validate_noise_state
from .baselines.learned_scale import LearnedScaleAdapter,ADAPTER_FIELDS as SCALE_FIELDS,trusted_constructor as scale_constructor,validate_owned_state as validate_scale_state
from .optim.base import MethodAdapter, ParameterGroup, SchedulePoint, default_parameter_groups
from .quantization.scales import grouping_for
from .quantization.surrogates import BACKWARD_RULES
from .quantization.context import QuantizationContext
from .records import ArtifactRef, RunLedger, WorkLedger, sha256_file
from .train import TrainingState, _rng_state, _restore_rng

SCHEMA = 'ralup.training_snapshot.v2'
ROOT = Path(__file__).resolve().parents[2]


class IntegrityError(ValueError):
    """Unknown, incomplete, malformed or unauthenticated snapshot data."""


class IncompleteStep(IntegrityError):
    """Only completed logical transactions may be published."""


# Exact fields bind schema versions: adding scientific state must fail closed.
_SCHEMAS = {}
_METHODS = {}
_SHARED_FIELDS = {}


def register_state_schema(name, cls, field_names):
    """Register a trusted dataclass constructor and its exact versioned fields."""
    if name in _SCHEMAS or not is_dataclass(cls) or tuple(f.name for f in fields(cls)) != tuple(field_names):
        raise IntegrityError('State schema must be new and declare every dataclass field')
    _SCHEMAS[name] = (cls,tuple(field_names))


def register_method_schema(name, cls, field_names, constructor, *, shared_fields=None):
    """Declare every adapter attribute except parameters/groups/rule.

    constructor(parameters, groups, rule) is trusted local code; no constructor
    reference is read from disk. New nested dataclasses need state registration.
    """
    if name in _METHODS or not isinstance(name,str) or not name or not callable(constructor):
        raise IntegrityError('Invalid or duplicate registered method schema')
    names = tuple(field_names)
    if len(set(names)) != len(names) or set(names)&{'parameters','groups','rule'}:
        raise IntegrityError('Invalid method state fields')
    shared_fields = {} if shared_fields is None else shared_fields
    if set(shared_fields)-{'histories','quantizer_state','runtime'} or any(
            not isinstance(v,(tuple,list)) or any(type(n) is not str for n in v) or len(set(v)) != len(v)
            for v in shared_fields.values()):
        raise IntegrityError('Invalid declared shared state fields')
    _METHODS[name] = (cls,names,constructor)
    _SHARED_FIELDS[name] = {key:tuple(shared_fields.get(key,())) for key in ('histories','quantizer_state','runtime')}


for _name,_cls in [('model_config.v1',ModelConfig),('schedule_point.v1',SchedulePoint),
                   ('parameter_group.v1',ParameterGroup),('quantization_context.v1',QuantizationContext)]:
    register_state_schema(_name,_cls,tuple(f.name for f in fields(_cls)))
register_method_schema('ordinary_adapter.v1',MethodAdapter,
                       ('optimizer_states','proposals','snapshot_schema'),MethodAdapter)

register_method_schema('matrix_adapter.v1',MatrixMethodAdapter,
                       ('optimizer_states','proposals','snapshot_schema'),MatrixMethodAdapter)


register_method_schema('ewgs_adapter.v2',EwgsAdapter,ADAPTER_FIELDS,trusted_constructor)
register_state_schema('trs_state.v1',TrsState,('r','K','previous_codes','observations'))
register_method_schema('trs_adapter.v1',TrsBackboneAdapter,TRS_FIELDS,trs_constructor)
register_method_schema('qar_adapter.v1',QarAdjacentAdapter,QAR_FIELDS,qar_constructor)
register_method_schema('rounding_adapter.v1',RoundingAdapter,QAR_FIELDS,rounding_constructor)
register_method_schema('noise_adapter.v1',NoiseAdapter,NOISE_FIELDS,noise_constructor)
register_method_schema('learned_scale_adapter.v1',LearnedScaleAdapter,SCALE_FIELDS,scale_constructor)


def _rule_payload(rule):
    if type(rule) is AdamWUpdateRule and set(vars(rule)) == {'betas','eps'}:
        # Preserve the historical AdamW representation and strict source boundary.
        return dict(vars(rule))
    if type(rule) in (MuonUpdateRule,MuonPlusUpdateRule,SoapUpdateRule,MadamUpdateRule) and set(vars(rule)) == {'config'}:
        value = {'schema':rule.state_schema,'config':dict(rule.configuration)}
        _construct_rule(value)
        return value
    raise IntegrityError('Unknown update rule requires registered schema')


def _construct_rule(payload):
    if set(payload) == {'betas','eps'}:
        return AdamWUpdateRule(**payload)
    _keys(payload,('schema','config'))
    constructors = {MuonUpdateRule.state_schema:MuonUpdateRule,MuonPlusUpdateRule.state_schema:MuonPlusUpdateRule,SoapUpdateRule.state_schema:SoapUpdateRule,MadamUpdateRule.state_schema:MadamUpdateRule}
    if payload['schema'] not in constructors:
        raise IntegrityError('Unknown update rule requires registered schema')
    constructor = constructors[payload['schema']]
    _keys(payload['config'],constructor.defaults)
    return constructor(dict(payload['config']))


def _validated_adapter(payload, prototype, rule):
    _,_,constructor = _METHODS[payload['method_schema']]
    if constructor in (MatrixMethodAdapter,trusted_constructor,trs_constructor,qar_constructor,rounding_constructor,noise_constructor,scale_constructor):
        if payload['groups'] != default_parameter_groups(prototype):
            raise IntegrityError('Matrix groups must match exact internal projection manifest and no-decay auxiliaries')
    elif type(rule) is not AdamWUpdateRule:
        raise IntegrityError('Matrix rules require the registered matrix adapter')
    adapter = constructor(dict(prototype.named_parameters()),payload['groups'],rule)
    if payload['method']['snapshot_schema'] != adapter.snapshot_schema:
        raise IntegrityError('Method state schema mismatch')
    return adapter


@dataclass
class TrainingSnapshot:
    metadata: dict
    tensors: dict[str,torch.Tensor]
    milestone: bool = False
    producer_files: dict[str,bytes] = field(default_factory=dict)


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


def _references(value,producer=None):
    if type(value) is ArtifactRef:
        archived = None if producer is None else producer.get(str(value.path.resolve()))
        if archived is None and producer is not None and not value.path.is_absolute():
            archived = producer.get(value.path.as_posix())
        if archived is not None:
            if value.sha256 != archived: raise IntegrityError('Producer reference hash mismatch')
        else: _authenticate(value)
    elif isinstance(value,(dict,MappingProxyType)):
        if set(value) == {'path','sha256'}: _references(ArtifactRef(**value),producer)
        else:
            for item in value.values(): _references(item,producer)
    elif isinstance(value,(list,tuple)):
        for item in value: _references(item,producer)


def _source_paths():
    return sorted((ROOT/'src/ralup').rglob('*.py')) + [ROOT/'requirements.lock',ROOT/'artifacts/environment/environment.json']


def _archive_producer():
    identities,files = {},{}
    for path in _source_paths():
        name = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        files[name] = data
        identities[name] = ArtifactRef(path,hashlib.sha256(data).hexdigest())
    return identities,files


def _automatic_identities():
    return _archive_producer()[0]


def _validate_producer(snapshot,identities):
    _keys(identities,('automatic','declared'))
    refs = identities['automatic']
    if not refs or set(refs) != set(snapshot.producer_files): raise IntegrityError('Producer archive inventory mismatch')
    if not {'requirements.lock','artifacts/environment/environment.json'} <= refs.keys():
        raise IntegrityError('Producer lock/environment archive missing')
    for name,ref in refs.items():
        if type(ref) is not ArtifactRef or type(snapshot.producer_files[name]) is not bytes:
            raise IntegrityError('Invalid producer archive identity/bytes')
        if not (name in ('requirements.lock','artifacts/environment/environment.json') or
                (name.startswith('src/ralup/') and name.endswith('.py') and '..' not in Path(name).parts)):
            raise IntegrityError('Unknown producer archive path')
        if hashlib.sha256(snapshot.producer_files[name]).hexdigest() != ref.sha256:
            raise IntegrityError('Producer archive hash mismatch')
    producer = {str(ref.path.resolve()):ref.sha256 for ref in refs.values()}
    producer.update({name:ref.sha256 for name,ref in refs.items()})
    _references(identities['declared'],producer)
    return producer


def _strict_runtime(payload):
    try: current = _automatic_identities()
    except OSError as error: raise IntegrityError('Current runtime identity unavailable') from error
    producer = payload['identities']['automatic']
    if current.keys() != producer.keys() or any(current[k].sha256 != producer[k].sha256 for k in current):
        raise IntegrityError('Current runtime incompatible with archived producer identity; no migration is registered')


def _validate_model_owners(model):
    """The supported architecture admits numeric weights/cache, never callbacks."""
    with torch.device('meta'):
        prototype = ByteTransformer(model.config)
    modules,expected = dict(model.named_modules()),dict(prototype.named_modules())
    if modules.keys() != expected.keys(): raise IntegrityError('Unregistered model child modules')
    caches = {'config','_recipe','_runtime','_contexts','_context_step'}
    for name,module in modules.items():
        reference = expected[name]
        if type(module) is not type(reference) or vars(module).keys() != vars(reference).keys():
            raise IntegrityError(f'Unregistered model child owner: {name}')
        for field,value in vars(module).items():
            if field == 'training':
                if type(value) is not bool: raise IntegrityError('Invalid module training mode')
            elif field in ('_parameters','_buffers','_modules'):
                other = getattr(reference,field)
                if value.keys() != other.keys(): raise IntegrityError('Unregistered model parameter/buffer/child owner')
                if field == '_parameters' and any((v is None) != (other[k] is None) for k,v in value.items()):
                    raise IntegrityError('Model Parameter ownership mismatch')
            elif name == '' and field in caches:
                continue
            elif value != getattr(reference,field):
                raise IntegrityError(f'Unregistered model setting or executable hook: {name}.{field}')
    for parameter in model.parameters():
        if parameter._backward_hooks or parameter._post_accumulate_grad_hooks:
            raise IntegrityError('Executable Parameter hooks cannot be snapshotted')


def capture_snapshot(state: TrainingState) -> TrainingSnapshot:
    if not state.checkpointable: raise IncompleteStep('Partial selection/update or failed state cannot be captured')
    expected = {'model','method','optimizer_state','quantizer_state','histories','schedule','cursor','rng_streams',
                'ledger','microbatch','clip_norm','failed','configuration','identities','process_audit','_pending','_work'}
    if {f.name for f in fields(TrainingState)} != expected: raise IntegrityError('TrainingState schema registration is out of date')
    if set(vars(state))|{'_pending','_work'} != expected: raise IntegrityError('Unknown training state fields require registered schema')
    if type(state.model) is not ByteTransformer: raise IntegrityError('Unregistered model schema')
    _validate_model_owners(state.model)
    if 'data' in state.rng_streams: raise IntegrityError('Named data RNG belongs only to cursor.rng_states.data')
    model_fields = set(vars(torch.nn.Module()))|{'config','_recipe','_runtime','_contexts','_context_step'}
    if set(vars(state.model)) != model_fields: raise IntegrityError('Unknown model fields require registered schema')
    matches = [(n,entry) for n,entry in _METHODS.items() if type(state.method) is entry[0]]
    if len(matches) != 1: raise IntegrityError('Unregistered method schema')
    method_name,(_,method_fields,_) = matches[0]
    if method_name=='learned_scale_adapter.v1':
        try:state.method.validate_live_scales()
        except (ValueError,RuntimeError) as error:raise IntegrityError(str(error)) from error
        for name,p in state.method.log_scales.items():
            if type(p) is not torch.nn.Parameter or not p.requires_grad or p._backward_hooks or p._post_accumulate_grad_hooks:
                raise IntegrityError('Learned scales require hook-free trainable Parameters')
            if p.grad is not state.method.scale_gradients.get(name):
                raise IntegrityError('Learned scale gradient owner alias mismatch')
    if set(vars(state.method)) != set(method_fields)|{'parameters','groups','rule'}:
        raise IntegrityError('Unknown adapter fields require registered method schema')
    rule_payload = _rule_payload(state.method.rule)
    parameters = dict(state.model.named_parameters())
    if set(parameters) != set(state.method.parameters) or any(state.method.parameters[n] is not p for n,p in parameters.items()):
        raise IntegrityError('Adapter must retain authoritative original Parameters')
    if state.optimizer_state is not state.method.optimizer_states: raise IntegrityError('Optimizer alias mismatch')
    if state.cursor is not None and type(state.cursor) is not BatchCursor: raise IntegrityError('Unregistered cursor')
    if type(state.ledger) is not WorkLedger or set(vars(state.ledger)) != {'stream','coverage','passes','elapsed','events','run_ledger'}:
        raise IntegrityError('Unknown ledger fields require registered schema')
    _references(state.identities); _references(state.configuration)
    producer,producer_files = _archive_producer()
    identities = {'automatic':producer,'declared':state.identities}
    durable = state.ledger.run_ledger
    run = None if durable is None else {'path':str(durable.path.resolve()),'metadata':ArtifactRef(durable.path/'run.json',sha256_file(durable.path/'run.json'))}
    payload = {'model_config':state.model.config,'weights':dict(state.model.state_dict()),
        'gradients':{n:None if p.grad is None else p.grad for n,p in parameters.items()},
        'model_training':state.model.training,'module_training':{n:m.training for n,m in state.model.named_modules()},
        'model_cache':{n:getattr(state.model,n) for n in ('_recipe','_runtime','_contexts','_context_step')},
        'method_schema':method_name,'groups':state.method.groups,'rule':rule_payload,
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
    snapshot = TrainingSnapshot(metadata,encoder.tensors,producer_files=producer_files)
    _validated(snapshot)
    return snapshot


_PAYLOAD = {'model_config','weights','gradients','model_training','module_training','model_cache','method_schema','groups','rule','method',
            'quantizer_state','histories','schedule','cursor','cursor_audit','rng_streams','global_rng','ledger','run_ledger',
            'microbatch','clip_norm','configuration','configuration_sha256','identities','process_audit'}


def _array(value,shape,dtype,label):
    if not isinstance(value,torch.Tensor) or tuple(value.shape) != tuple(shape) or value.dtype != dtype:
        raise IntegrityError(f'{label} shape/dtype mismatch')
    if dtype == torch.uint8 and bool((value > 14).any()): raise IntegrityError(f'{label} invalid code ordinal')


def _validate_quantized_owners(payload,prototype):
    """Versioned H06 shared owners; future extras require method declarations."""
    cache = payload['model_cache']
    recipe = cache['_recipe']
    shared = _SHARED_FIELDS[payload['method_schema']]
    if recipe is None:
        if type(cache['_contexts']) is not dict or cache['_contexts'] or cache['_context_step'] is not None:
            raise IntegrityError('FP32 context cache must be an empty mapping with no marker')
        for key,value in (('histories',payload['histories']),('quantizer_state',payload['quantizer_state']),('runtime',cache['_runtime'])):
            allowed = set(shared[key]) | ({'loss_ema','loss_updates'} if key == 'histories' else set())
            if type(value) is not dict or set(value)-allowed: raise IntegrityError('FP32 owner has undeclared state')
        step = payload['schedule']['step']
        history = payload['histories']
        if not step and ({'loss_ema','loss_updates'} & history.keys()):
            raise IntegrityError('Unconfigured initial FP32 state has completed history')
        if step and (type(history.get('loss_updates')) is not int or history['loss_updates'] != step
                     or type(history.get('loss_ema')) not in (int,float)
                     or not math.isfinite(history['loss_ema']) or history['loss_ema'] < 0):
            raise IntegrityError('Completed FP32 loss history missing or invalid')
        return
    if type(recipe) is not dict: raise IntegrityError('Quantization recipe must be explicit')
    if set(recipe)-{'name','rotation','backward'} or 'name' not in recipe: raise IntegrityError('Unknown recipe fields')
    grouping = grouping_for(recipe['name'])
    rotation = recipe.get('rotation','identity')
    backward = recipe.get('backward','trust' if recipe['name'].startswith('quest') else 'ste')
    if backward=='ewgs' and payload['method_schema']!='ewgs_adapter.v2':raise IntegrityError('EWGS backward requires trusted EWGS owner')
    if rotation not in ('identity','full_hadamard') or backward not in BACKWARD_RULES: raise IntegrityError('Invalid context recipe')
    names = {item['name'] for item in prototype.quantized_matrix_manifest()}
    shared = _SHARED_FIELDS[payload['method_schema']]
    histories,quantizer,runtime = payload['histories'],payload['quantizer_state'],cache['_runtime']
    for owner,allowed in ((histories,names|{'loss_ema','loss_updates'}|set(shared['histories'])),
                          (quantizer,names|set(shared['quantizer_state'])),(runtime,names|set(shared['runtime']))):
        if type(owner) is not dict or set(owner)-allowed: raise IntegrityError('Unknown undeclared quantizer/history/runtime owner')
    contexts = cache['_contexts']
    if type(contexts) is not dict or set(contexts)-names: raise IntegrityError('Unknown matrix context owner')
    step = payload['schedule']['step']
    if step:
        if set(contexts) != names or cache['_context_step'] != step-1: raise IntegrityError('Completed context cache missing or stale')
        if not names|{'loss_ema','loss_updates'} <= histories.keys(): raise IntegrityError('Completed history/EMA owner missing')
        if type(histories['loss_ema']) not in (float,int) or not math.isfinite(histories['loss_ema']) or histories['loss_ema'] < 0:
            raise IntegrityError('Invalid completed loss EMA')
        if type(histories['loss_updates']) is not int or histories['loss_updates'] != step: raise IntegrityError('Completed history update counter mismatch')
    elif contexts:
        if set(contexts) != names or cache['_context_step'] != 0: raise IntegrityError('Initial context cache is partial')
    elif cache['_context_step'] is not None: raise IntegrityError('Initial context cache marker without contexts')
    fixed = 'fixed' in recipe['name']
    if step and fixed and (not names <= runtime.keys() or not names <= quantizer.keys()):
        raise IntegrityError('Completed fixed-scale handoff owner missing')
    if not fixed and set(quantizer)&names: raise IntegrityError('Nonfixed recipe has undeclared quantizer handoffs')
    parameters = dict(prototype.named_parameters())
    for name in names:
        shape = parameters[name].shape
        scale_shape = (1,1) if grouping == 'whole_tensor' else (shape[0],1)
        context = contexts.get(name)
        if context is not None:
            if type(context) is not QuantizationContext: raise IntegrityError('Unregistered matrix context')
            if (context.recipe,context.rotation,context.grouping,context.backward,context.step) != (
                    recipe['name'],rotation,grouping,backward,step-1 if step else 0):
                raise IntegrityError('Context recipe/step identity mismatch')
            _array(context.scales,scale_shape,torch.float32,'Context scales')
            if not bool((context.scales > 0).all()): raise IntegrityError('Context scales must be positive')
            for field in ('storage_u','storage_values','gradient_mask'): _array(getattr(context,field),shape,torch.float32,field)
            _array(context.codes,shape,torch.uint8,'Context codes')
            _array(context.zero_groups,scale_shape,torch.bool,'Context zero groups')
            if step or context.scale_zero_groups is not None: _array(context.scale_zero_groups,scale_shape,torch.bool,'Context initialization zero groups')
        if name in histories:
            history = histories[name]
            _keys(history,('previous_codes','two_steps_ago_codes','cached_codes','scales'))
            _array(history['previous_codes'],shape,torch.uint8,'History previous codes')
            _array(history['cached_codes'],shape,torch.uint8,'History cached codes')
            _array(history['scales'],scale_shape,torch.float32,'History scales')
            if step > 1: _array(history['two_steps_ago_codes'],shape,torch.uint8,'History two-step codes')
            elif history['two_steps_ago_codes'] is not None: raise IntegrityError('First-step two-step history must be null')
            if context is None or not torch.equal(history['previous_codes'],context.codes) or not torch.equal(history['scales'],context.scales):
                raise IntegrityError('Context/history cross-owner mismatch')
        for owner in (runtime,quantizer):
            if name not in owner: continue
            handoff = owner[name]
            if step and fixed:
                _keys(handoff,('previous_scales','previous_zero_groups'))
            elif type(handoff) is not dict or set(handoff)-{'previous_scales','previous_zero_groups','warm_start','codes'}:
                raise IntegrityError('Unknown runtime handoff fields')
            for field,value in handoff.items():
                dtype = torch.bool if field == 'previous_zero_groups' else torch.uint8 if field == 'codes' else torch.float32
                _array(value,shape if field in ('warm_start','codes') else scale_shape,dtype,'Runtime '+field)
                if field == 'previous_scales' and (not fixed or not bool((value > 0).all())): raise IntegrityError('Invalid fixed scale handoff')
            if step and fixed and ((payload['method_schema']!='learned_scale_adapter.v1' and not torch.equal(handoff['previous_scales'],context.scales)) or
                                   not torch.equal(handoff['previous_zero_groups'],context.scale_zero_groups)):
                raise IntegrityError('Fixed-scale cross-owner handoff mismatch')


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
        producer = _validate_producer(snapshot,payload['identities'])
        _references(payload['configuration'],producer); _references(payload['run_ledger'])
        _keys(payload['model_cache'],('_recipe','_runtime','_contexts','_context_step'))
        rule = _construct_rule(payload['rule'])
        _keys(payload['ledger'],('stream','coverage','passes','elapsed','events'))
        # Validate architecture/array identities on meta before any device transfer.
        if type(payload['model_config']) is not ModelConfig: raise IntegrityError('Unregistered model geometry')
        with torch.device('meta'):
            prototype = ByteTransformer(payload['model_config'])
        expected = dict(prototype.named_parameters())
        _keys(payload['module_training'],dict(prototype.named_modules()))
        if any(type(v) is not bool for v in payload['module_training'].values()): raise IntegrityError('Invalid module training modes')
        if payload['model_training'] != payload['module_training']['']: raise IntegrityError('Root module training mismatch')
        if 'data' in payload['rng_streams']: raise IntegrityError('Named data RNG belongs only to cursor.rng_states.data')
        _keys(payload['weights'],expected)
        _keys(payload['gradients'],expected)
        for name,parameter in expected.items():
            for array in (payload['weights'][name],payload['gradients'][name]):
                if array is not None and (not isinstance(array,torch.Tensor) or array.shape != parameter.shape or array.dtype != parameter.dtype):
                    raise IntegrityError('Model/gradient shape or dtype mismatch')
            if payload['weights'][name] is None: raise IntegrityError('Missing model array')
        adapter = _validated_adapter(payload,prototype,rule)
        _keys(payload['method']['optimizer_states'],expected)
        completed_step = payload['schedule']['step']
        if type(completed_step) is not int or completed_step < 0: raise IntegrityError('Invalid completed step')
        for name,values in payload['method']['optimizer_states'].items():
            if completed_step and not values: raise IntegrityError('Completed step is missing optimizer moments')
            if values:
                parameter_rule = adapter.rule_for_parameter(name)
                _keys(values,parameter_rule.optimizer_state_fields)
                if type(parameter_rule) in (SoapUpdateRule,MadamUpdateRule):
                    parameter_rule.validate_state(values,payload['weights'][name])
                    continue
                if type(parameter_rule) is AdamWUpdateRule:
                    counter = values['step']
                    if counter.shape != torch.Size([]) or counter.dtype != torch.float32 or counter.item() < 0 or counter.item() != int(counter.item()):
                        raise IntegrityError('Invalid optimizer scalar counter')
                for field in (set(parameter_rule.optimizer_state_fields)-{'step'}):
                    if values[field].shape != expected[name].shape or values[field].dtype != torch.float32:
                        raise IntegrityError('Optimizer original-W shape/dtype mismatch')
        _,names,_ = _METHODS[payload['method_schema']]
        _keys(payload['method'],names)
        if payload['method']['proposals']: raise IncompleteStep('Uncommitted proposals in snapshot')
        _validate_quantized_owners(payload,prototype)
        if payload['method_schema']=='ewgs_adapter.v2':
            if type(rule) is not AdamWUpdateRule:raise IntegrityError('EWGS owns original-W AdamW only')
            if payload['model_cache']['_recipe'] is None or payload['model_cache']['_recipe'].get('backward')!='ewgs':raise IntegrityError('EWGS requires explicit backward recipe')
            if payload['groups']!=default_parameter_groups(prototype):raise IntegrityError('EWGS groups mismatch')
            validate_owned_state(payload['method'],payload['groups'],completed_step,payload['model_cache']['_contexts'])
        if payload['method_schema']=='trs_adapter.v1':
            validate_trs_state(payload['method'],payload['groups'],completed_step,payload['model_cache']['_contexts'],rule)
        if payload['method_schema']=='noise_adapter.v1':
            validate_noise_state(payload['method'],payload['groups'],completed_step,rule,payload['model_cache']['_recipe'])
        if payload['method_schema']=='learned_scale_adapter.v1':
            validate_scale_state(payload['method'],payload['groups'],completed_step,rule,payload['model_cache']['_recipe'],payload['weights'],payload['model_cache']['_runtime'],payload['quantizer_state'])
        if payload['method_schema'] in ('qar_adapter.v1','rounding_adapter.v1'):
            validate_routing_state(payload['method'],payload['groups'],completed_step,payload['model_cache']['_contexts'],rule,payload['weights'],payload['rng_streams'],payload['model_cache']['_recipe'],rounding=payload['method_schema']=='rounding_adapter.v1')
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
    _strict_runtime(payload)
    if payload['method_schema']=='learned_scale_adapter.v1' and payload['method']['scale_backend'] not in (None,device):
        raise IntegrityError('Learned scales require the declared producing backend; cross-backend restoration unsupported')
    try:
        model = ByteTransformer(payload['model_config']).to(device)
        model.load_state_dict(payload['weights'],strict=True)
        model.train(payload['model_training'])
        for name,module in model.named_modules(): module.training = payload['module_training'][name]
        for name,value in payload['model_cache'].items(): setattr(model,name,_device_tree(value,device))
        parameters = dict(model.named_parameters())
        if set(parameters) != set(payload['gradients']): raise IntegrityError('Gradient manifest mismatch')
        for name,p in parameters.items():
            grad = payload['gradients'][name]
            if grad is not None and (grad.shape != p.shape or grad.dtype != p.dtype): raise IntegrityError('Gradient shape/dtype mismatch')
            p.grad = None if grad is None else grad.to(device).clone()
        rule = _construct_rule(payload['rule'])
        _,_,constructor = _METHODS[payload['method_schema']]
        method = constructor(parameters,payload['groups'],rule)
        for name,value in payload['method'].items(): setattr(method,name,_device_tree(value,device))
        if payload['method_schema']=='learned_scale_adapter.v1':
            method.restore_owned_parameters()
            method.validate_live_scales()
        if payload['method_schema']=='noise_adapter.v1':method.mask_rng_state=payload['method']['mask_rng_state'].clone()
        if payload['method_schema']=='ewgs_adapter.v2':method.probe_rng_state=payload['method']['probe_rng_state'].clone()
        # Noncapturable scalar counters stay CPU; moments remain original-W device.
        if set(method.optimizer_states) != set(parameters): raise IntegrityError('Optimizer manifest mismatch')
        for name,values in method.optimizer_states.items():
            if values:
                parameter_rule = method.rule_for_parameter(name)
                _keys(values,parameter_rule.optimizer_state_fields)
                if type(parameter_rule) is SoapUpdateRule:
                    parameter_rule.restore_basis_layout(values)
                if type(parameter_rule) is AdamWUpdateRule:
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
            'producer_files':{name:{'path':f'producer/{hashlib.sha256(raw).hexdigest()}.bin',
                              'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)}
                              for name,raw in snapshot.producer_files.items()},
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
        (staging/'producer').mkdir()
        for name,entry in manifest['producer_files'].items():
            archive = staging/entry['path']
            if not archive.exists(): _flush_write(archive,snapshot.producer_files[name])
        _fsync_directory(staging/'producer')
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
        _keys(manifest,('schema','metadata','milestone','tensor_sha256','tensors','producer_files'))
        if manifest['schema'] != SCHEMA: raise IntegrityError('Unknown snapshot schema')
        data = (reference.path.parent/'tensors.safetensors').read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['tensor_sha256']: raise IntegrityError('Tensor payload hash mismatch')
        tensors = load_tensors(data)
        if set(tensors) != set(manifest['tensors']): raise IntegrityError('Tensor manifest mismatch')
        for key,tensor in tensors.items():
            spec = manifest['tensors'][key]; _keys(spec,('shape','dtype'))
            if list(tensor.shape) != spec['shape'] or str(tensor.dtype) != spec['dtype']:
                raise IntegrityError('Tensor shape/dtype mismatch')
        producer_files = {}
        for name,entry in manifest['producer_files'].items():
            _keys(entry,('path','sha256','size'))
            digest = entry['sha256']
            ArtifactRef(reference.path,digest)  # Validate digest syntax before path use.
            if entry['path'] != f'producer/{digest}.bin' or type(entry['size']) is not int or entry['size'] < 0:
                raise IntegrityError('Invalid producer archive descriptor')
            raw = (reference.path.parent/entry['path']).read_bytes()
            if len(raw) != entry['size'] or hashlib.sha256(raw).hexdigest() != digest:
                raise IntegrityError('Producer archive hash/length mismatch')
            producer_files[name] = raw
        snapshot = TrainingSnapshot(manifest['metadata'],tensors,manifest['milestone'],producer_files)
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
    for row in payload['ledger']['events']:
        row.pop('elapsed_seconds',None)
        for name in TIMING_FIELDS: row.get('optimizer_work',{}).pop(name,None)
        row.get('ewgs_work',{}).pop('seconds',None)
        row.get('learned_scale_work',{}).pop('scale_seconds',None)
        row.get('noise_work',{}).pop('mask_seconds',None)
        for name in ('context_seconds','probability_seconds'):
            row.get('routing_rounding_work',{}).pop(name,None)
    if payload['rule'].get('schema') == SoapUpdateRule.state_schema:
        for values in payload['method']['optimizer_states'].values():
            for name in TIMING_FIELDS: values.get('telemetry',{}).pop(name,None)
    if payload['method_schema']=='ewgs_adapter.v2':
        for values in payload['method']['refresh_telemetry'].values():values.pop('elapsed_seconds',None)
    if payload['method_schema']=='learned_scale_adapter.v1':payload['method'].pop('scale_validation_telemetry')
    encoder = _Encoder()
    # _decode leaves RNG states as plain data, suitable for deterministic hashing.
    metadata = encoder.encode(payload)
    return hashlib.sha256(canonical_bytes(metadata)+tensor_bytes(encoder.tensors)).hexdigest()
