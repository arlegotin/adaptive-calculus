"""S15 logical transaction, with original-W adapter commit and three ledgers.

select_logical_batch stages data selection; train_step publishes it only after
all scientific updates succeed. Corpus indexes are shared immutable data; each
staged cursor owns RNG/carry and copies its permutation only at epoch rollover.
H07 owns durable snapshots. Transaction copy/restore costs remain charged in the
whole-step clock and must be measured by H08.

Exceptional adapters use state.record_work for extra passes (including work that
fails). Do not also report/charge ordinary callbacks: summed_nll charges forward
launches and its autograd hook charges backward launches automatically.
"""
import copy
from dataclasses import dataclass, field, fields, is_dataclass
import math
import random
import time
from types import MappingProxyType

import torch

from .data.batches import LogicalBatch, next_logical_batch, pack_microbatches
from .evaluate import summed_nll, validate_batch
from .optim.base import SchedulePoint
from .quantization.context import prepare_context
from .records import WorkLedger
from .policy.history import MatrixHistory, LossObservation, commit_history, resolve_realized_history
from .policy.features import collect_step_features


def _clone(value):
    """Own mutable scientific payloads without executable serialization."""
    if isinstance(value,torch.Tensor):
        return value.detach().clone()
    if isinstance(value,dict):
        return {key:_clone(item) for key,item in value.items()}
    if isinstance(value,MappingProxyType):
        return MappingProxyType({key:_clone(item) for key,item in value.items()})
    if isinstance(value,(list,tuple)):
        return type(value)(_clone(item) for item in value)
    if is_dataclass(value):
        result = copy.copy(value)
        for item in fields(value):
            object.__setattr__(result,item.name,_clone(getattr(value,item.name)))
        return result
    return copy.deepcopy(value)


def _rng_state():
    return {'torch_cpu':torch.get_rng_state().clone(), 'python':random.getstate(),
            'torch_mps':torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None}


def _restore_rng(state):
    torch.set_rng_state(state['torch_cpu'])
    random.setstate(state['python'])
    if state['torch_mps'] is not None:
        torch.mps.set_rng_state(state['torch_mps'])


def _synchronize(model):
    if next(model.parameters()).device.type == 'mps':
        torch.mps.synchronize()


@dataclass
class TrainingState:
    model: object
    method: object
    optimizer_state: dict = field(default_factory=dict)
    quantizer_state: dict = field(default_factory=dict)
    histories: dict = field(default_factory=dict)
    schedule: dict = field(default_factory=lambda:{'step':0,'point':SchedulePoint(.001,0.,0.)})
    cursor: object = None
    rng_streams: dict = field(default_factory=dict)
    ledger: WorkLedger = field(default_factory=WorkLedger)
    microbatch: int = 1
    clip_norm: float | None = None
    failed: bool = False
    configuration: dict = field(default_factory=dict)
    identities: dict = field(default_factory=dict)
    process_audit: list = field(default_factory=list)
    _pending: object = field(default=None,init=False,repr=False)
    _work: dict | None = field(default=None,init=False,repr=False)

    def __post_init__(self):
        if self.optimizer_state and self.optimizer_state is not self.method.optimizer_states:
            raise ValueError('Optimizer state must be the adapter-owned named state')
        self.optimizer_state = self.method.optimizer_states
        if type(self.microbatch) is not int or self.microbatch <= 0:
            raise ValueError('Physical microbatch must be a positive integer')
        if self.clip_norm is not None and (not math.isfinite(self.clip_norm) or self.clip_norm <= 0):
            raise ValueError('Clip norm must be finite and positive')

    @property
    def checkpointable(self):
        return not self.failed and self._work is None and self._pending is None and not self.method.proposals

    def record_work(self, *, additional_loss=0, backward=0, query=0, forward_context=0, optimizer_work=None, loss_targets=0, extra_pass_event=None, ewgs_seconds=0.):
        """Charge actual launched extra work before an exceptional pass.

        Counts overlap and cannot be summed as a total. Ordinary forward/backward
        callbacks charge themselves; explicit calls are for other pass paths.
        """
        if self._work is None:
            raise ValueError('Extra work requires an active logical transaction')
        if extra_pass_event is not None:
            if type(extra_pass_event) is not dict or set(extra_pass_event)!={'event_id','execution_id','kind','target_bytes'} or extra_pass_event['kind'] not in ('extra_loss','extra_loss_backward','gradient_only','hvp') or type(extra_pass_event['event_id']) is not str or type(extra_pass_event['execution_id']) is not str or type(extra_pass_event['target_bytes']) is not int or extra_pass_event['target_bytes']<=0:
                raise ValueError('Invalid extra pass event')
            events=self._work.setdefault('extra_pass_events',[])
            if any(e['event_id']==extra_pass_event['event_id'] for e in events):raise ValueError('Duplicate extra pass event')
            events.append(dict(extra_pass_event))
            work=self._work.setdefault('ewgs_work',{'extra_loss_target_bytes':0,'loss_backward_target_bytes':0,'gradient_only_target_bytes':0,'hvp_target_bytes':0,'seconds':0.})
            work[('loss_backward' if extra_pass_event['kind']=='extra_loss_backward' else extra_pass_event['kind'])+'_target_bytes']+=extra_pass_event['target_bytes']
        if type(ewgs_seconds) not in (int,float) or not math.isfinite(ewgs_seconds) or ewgs_seconds<0:raise ValueError('Invalid EWGS seconds')
        if ewgs_seconds:
            self._work.setdefault('ewgs_work',{'extra_loss_target_bytes':0,'loss_backward_target_bytes':0,'gradient_only_target_bytes':0,'hvp_target_bytes':0,'seconds':0.})['seconds']+=ewgs_seconds
        if optimizer_work is not None:
            from .optim.soap import TELEMETRY_FIELDS,TIMING_FIELDS
            if type(optimizer_work) is not dict or set(optimizer_work) != set(TELEMETRY_FIELDS):
                raise ValueError('Unknown optimizer work inventory')
            for name,value in optimizer_work.items():
                if type(value) not in (int,float) or not math.isfinite(value) or value < 0 or (name not in TIMING_FIELDS and type(value) is not int):
                    raise ValueError('Invalid optimizer work measurement')
            work = self._work.setdefault('optimizer_work',{k:0. if k in TIMING_FIELDS else 0 for k in TELEMETRY_FIELDS})
            for name,value in optimizer_work.items(): work[name] += value
        for name,value in {'additional_loss':additional_loss,'backward':backward,'query':query,'forward_context':forward_context,'loss_targets':loss_targets}.items():
            if type(value) is not int or value < 0:
                raise ValueError('Work counts must be nonnegative integers')
            self._work[name] += value

    def collect_policy_features(self, proposals, schedule):
        return collect_step_features(self, proposals, schedule)

    def summed_nll(self, batch_slice, *, parameter_overrides=None):
        validate_batch(batch_slice)
        if self._work is not None:
            self._work['forward_context'] += sum(len(b.inputs) for b in batch_slice.block_slices)
            self._work['loss_targets'] += batch_slice.target_count
        loss = summed_nll(self.model,batch_slice,parameter_overrides=parameter_overrides)
        if self._work is not None:
            # Hook runs when this graph's scalar backward actually starts.
            work = self._work
            loss.register_hook(lambda grad: _charge_backward(work,batch_slice.target_count,grad))
        return loss


def _charge_backward(work, count, gradient):
    work['backward'] += count
    return gradient


@dataclass(frozen=True)
class StepResult:
    nll_sum: float
    target_count: int
    work: dict
    telemetry: dict


class TrainingFailure(RuntimeError):
    """Failed work was charged; restored state cannot publish a checkpoint."""


def select_logical_batch(state: TrainingState, target_budget=8192) -> LogicalBatch:
    if state.failed or state._work is not None or state._pending is not None:
        raise ValueError('State cannot stage another logical batch')
    if state.cursor is None:
        raise ValueError('Training state requires a cursor for staged selection')
    _synchronize(state.model)
    selection_start = time.perf_counter()
    staged = _stage_cursor(state.cursor)
    batch = next_logical_batch(staged,target_budget)
    validate_batch(batch)
    state._pending = (batch,staged,selection_start)
    return batch


def _stage_cursor(cursor):
    if cursor is None:
        return None
    staged = copy.copy(cursor)
    staged._rng = random.Random()
    staged._rng.setstate(cursor._rng.getstate())
    staged._carry = copy.copy(cursor._carry)
    return staged


def _snapshot(state):
    # Retain actual Parameter objects so even illegal replacement can be undone.
    parameters = dict(state.model.named_parameters())
    owned = getattr(state.method,'transaction_parameters',lambda:{})()
    return {'parameters':parameters,'weights':_clone(state.model.state_dict()),'owned_parameters':dict(owned),
            'grads':{name:None if p.grad is None else p.grad.detach().clone() for name,p in parameters.items()},
            'model_context':{name:_clone(getattr(state.model,name)) for name in ('_contexts','_context_step','_runtime','_recipe')},
            'method':{name:_clone(value) for name,value in state.method.__dict__.items()
                      if name not in ('parameters','groups','rule')},
            'state':{name:_clone(getattr(state,name)) for name in ('quantizer_state','histories','schedule','rng_streams','configuration','identities')},
            'cursor':_stage_cursor(state.cursor),'cursor_object':state.cursor,
            'rule_object':state.method.rule,'rule_state':_clone(vars(state.method.rule)),
            'groups':state.method.groups,'ledger_object':state.ledger,
            'ledger':{name:_clone(getattr(state.ledger,name)) for name in ('stream','passes','elapsed','coverage')},
            'ledger_events':len(state.ledger.events),
            'rng':_rng_state()}


def _restore(state, snapshot):
    for name,p in snapshot['parameters'].items():
        parent,_,leaf = name.rpartition('.')
        module = state.model.get_submodule(parent) if parent else state.model
        module._parameters[leaf] = p
    state.model.load_state_dict(snapshot['weights'])
    for name,p in snapshot['parameters'].items():
        p.grad = snapshot['grads'][name]
    for name,value in snapshot['model_context'].items():
        setattr(state.model,name,value)
    for name in list(state.method.__dict__):
        if name not in ('parameters','groups','rule'):
            delattr(state.method,name)
    state.method.__dict__.update(snapshot['method'])
    if hasattr(state.method,'restore_owned_parameters'):
        state.method.restore_owned_parameters(snapshot['owned_parameters'])
    state.method.parameters = dict(snapshot['parameters'])
    state.method.groups = snapshot['groups']
    state.method.rule = snapshot['rule_object']
    vars(state.method.rule).clear()
    vars(state.method.rule).update(snapshot['rule_state'])
    state.optimizer_state = state.method.optimizer_states
    for name,value in snapshot['state'].items():
        setattr(state,name,value)
    state.cursor = snapshot['cursor_object']
    state.ledger = snapshot['ledger_object']
    if snapshot['cursor'] is not None:
        state.cursor.__dict__.update(snapshot['cursor'].__dict__)
    for name,value in snapshot['ledger'].items():
        setattr(state.ledger,name,value)
    del state.ledger.events[snapshot['ledger_events']:]
    _restore_rng(snapshot['rng'])


def _contexts(state, step):
    # Exceptional endpoint/noise preparation is within snapshot/rollback/time.
    old_histories = _clone(state.histories)
    prepare = getattr(state.method,'prepare_contexts',None)
    contexts = (prepare(state,step) if prepare is not None else
                ({} if state.model._recipe is None else
                 state.model.prepare_quantized_views(step,state.rng_streams.get('quantizer'))))
    if not _equal(old_histories,state.histories):
        raise ValueError('Adapter changed shared history during context preparation')
    resolve_start = time.perf_counter()
    resolved = 0
    for name,context in contexts.items():
        if name in state.histories:
            resolved += int(state.histories[name].pending)
            state.histories[name] = resolve_realized_history(state.histories[name],context)
        if 'fixed' in context.recipe:
            handoff = {'previous_scales':context.scales.clone(),
                       'previous_zero_groups':context.scale_zero_groups.clone()}
            state.quantizer_state[name] = handoff
            state.model._runtime[name] = handoff
    state._work['history_work'] = {'resolve_seconds':time.perf_counter()-resolve_start,'matrix_refreshes':resolved}
    return contexts


def _transition(state, contexts, nll, count):
    from dataclasses import replace
    commit_start = time.perf_counter()
    telemetry = {}
    parameters = dict(state.model.named_parameters())
    for name,context in contexts.items():
        old = state.histories.get(name,MatrixHistory())
        # Actual pre-step W is captured after trusted endpoint initialization.
        # The transaction-entry snapshot still owns exact rollback operands.
        before = state._work['before_weights'][name]
        new = commit_history(old,before,parameters[name],context)
        new = replace(new,common_gradient_ema=state._work.get('common_gradient_ema',{}).get(name,old.common_gradient_ema),
                      last_features=state._work.get('feature_tensors',{}).get(name,old.last_features))
        state.histories[name] = new
        telemetry[name] = {'cached_transitions':int((context.codes!=new.cached_codes).sum()),
                           'realized_refresh_transitions':None if not old.realized_valid else round(old.disagreement_fraction*context.codes.numel()),
                           'scale_changed':False if old.previous_scales is None else not torch.equal(old.previous_scales,context.scales),
                           'reversal_fraction':new.reversal_fraction,'reversal_coverage':new.reversal_coverage,
                           'numerical_endpoint_displacement':new.numerical_displacement,
                           'refresh_endpoint_displacement':old.refresh_endpoint_displacement,'history_bytes':new.tensor_bytes}
    state.histories['loss_ema'] = LossObservation(state.histories.get('loss_ema'),nll/count).prospective_ema
    state.histories['loss_updates'] = state.histories.get('loss_updates',0)+1
    state._work['history_work'].update(commit_seconds=time.perf_counter()-commit_start,
                                       matrix_commits=len(contexts),cached_quantizer_calls=len(contexts),
                                       tensor_bytes=sum(state.histories[name].tensor_bytes for name in contexts))
    return telemetry


def train_step(state: TrainingState, batch: LogicalBatch) -> StepResult:
    # Mandatory entry guard precedes contexts, snapshots, clearing, or work costs.
    validate_batch(batch)
    if state.failed:
        raise ValueError('Failed training state cannot continue or checkpoint')
    if state._work is not None:
        raise ValueError('Logical transaction already active')
    if state._pending is not None and state._pending[0] is not batch:
        raise ValueError('Batch differs from staged cursor selection')
    _synchronize(state.model)
    start = time.perf_counter() if state._pending is None else state._pending[2]
    snapshot = _snapshot(state)
    step = state.schedule['step']
    state._work = {'stream':0,'unique':0,'additional_loss':0,'backward':0,'query':0,'forward_context':0,'loss_targets':0}
    nll_values = []
    try:
        contexts = _contexts(state,step)
        prepared_history = _clone(state.histories)
        state._work['before_weights'] = {name:state.method.parameters[name].detach().clone() for name in contexts}
        # First-step owners may be constructed during context preparation inside
        # this transaction. Bind the prepared inventory before backward/commit,
        # while retaining every owner that already existed at transaction entry.
        prepared_owned = dict(getattr(state.method,'transaction_parameters',lambda:{})())
        if any(prepared_owned.get(name) is not p for name,p in snapshot['owned_parameters'].items()):
            raise ValueError('Method changed method-owned Parameter inventory or identity during preparation')
        state.model.zero_grad(set_to_none=True)
        state.method.prepare_step(state,batch)
        for microbatch in pack_microbatches(batch,state.microbatch):
            report = state.method.backward_microbatch(state,microbatch)
            nll_values.append(report['summed_nll'])
        nll = math.fsum(nll_values)
        if not math.isfinite(nll):
            raise ValueError('Nonfinite training NLL')
        for p in state.model.parameters():
            if p.grad is not None:
                p.grad.div_(batch.target_count)
                if not torch.isfinite(p.grad).all():
                    raise ValueError('Nonfinite normalized gradient')
        norm = None
        if state.clip_norm is not None:
            norm = float(torch.nn.utils.clip_grad_norm_(state.model.parameters(),state.clip_norm,error_if_nonfinite=True))
        # H05 separately constructs/validates all proposals and explicitly updates
        # named internal/auxiliary groups; identity uses rule.commit addcdiv order.
        state._work['training_loss'] = nll/batch.target_count
        before_history = prepared_history
        updates = state.method.finish_step(state,state.schedule['point'])
        if state.histories.keys() != before_history.keys() or any(
                not _equal(state.histories[k],before_history[k]) for k in before_history):
            raise ValueError('Adapter changed shared history before its single commit')
        for name in ('configuration','identities'):
            if not _equal(getattr(state,name),snapshot['state'][name]):
                raise ValueError('Adapter changed snapshot configuration/identity during commit')
        if not _equal(state.schedule,snapshot['state']['schedule']):
            raise ValueError('Adapter advanced the shared schedule before transaction commit')
        if _cursor_marker(state.cursor) != _cursor_marker(snapshot['cursor']):
            raise ValueError('Adapter advanced the shared cursor before transaction commit')
        if any(not _equal(getattr(state.ledger,k),v) for k,v in snapshot['ledger'].items()) or len(state.ledger.events) != snapshot['ledger_events']:
            raise ValueError('Adapter advanced the shared ledger before transaction commit')
        if not _finite(state.method.optimizer_states):
            raise ValueError('Nonfinite committed optimizer state')
        if {n:id(p) for n,p in state.model.named_parameters()} != {n:id(p) for n,p in snapshot['parameters'].items()}:
            raise ValueError('Method replaced a Parameter in place of preserving identity')
        committed_owned = getattr(state.method,'transaction_parameters',lambda:{})()
        if {n:id(p) for n,p in committed_owned.items()} != {n:id(p) for n,p in prepared_owned.items()}:
            raise ValueError('Method changed method-owned Parameter inventory or identity during commit')
        if not all(bool(torch.isfinite(p).all()) for p in state.model.parameters()):
            raise ValueError('Nonfinite committed model')
        matrices = _transition(state,contexts,nll,batch.target_count)
        state.optimizer_state = state.method.optimizer_states
        state.schedule['step'] = step+1
        unique = _unique_direct(state.ledger,batch) if state._pending is None else 0
        if state._pending is not None:
            staged = state._pending[1]
            unique = staged.unique_target_count-state.cursor.unique_target_count
        features = state._work.get('features',{})
        feature_work = state._work.get('feature_work')
        public_work = {k:v for k,v in state._work.items() if k not in ('before_weights','training_loss','features','feature_tensors','common_gradient_ema')}
        if feature_work is not None:
            public_work['policy_seconds'] = feature_work['scan_seconds']+public_work.get('policy_seconds',0.)
        if feature_work is not None and not feature_work['detailed_trace']:features = {}
        work = {**public_work,'step':step,'stream':batch.target_count,'unique':unique,'nll_sum':nll,
                'nll_count':batch.target_count,'status':'completed_step','nonfinite_count':0,
                'matrices':matrices,'candidate_validation_updates':updates.get('candidate_validation_updates',0)}
        _synchronize(state.model)
        work['elapsed_seconds'] = time.perf_counter()-start
        state.ledger.record(work)
        if state._pending is not None:
            state.cursor.__dict__.update(state._pending[1].__dict__)
        state._pending = None
        return StepResult(nll,batch.target_count,work,{'gradient_norm_before_clip':norm,'matrices':matrices,'features':features,**updates})
    except Exception as error:
        work = {k:v for k,v in state._work.items() if k not in ('before_weights','training_loss','features','feature_tensors','common_gradient_ema')}
        nonfinite = _nonfinite_count((tuple(state.model.parameters()),tuple(p.grad for p in state.model.parameters()),
                                      state.method.optimizer_states,state.method.proposals))+sum(not math.isfinite(v) for v in nll_values)
        _restore(state,snapshot)
        state.failed = True
        state._pending = None
        _synchronize(state.model)
        state.ledger.record({**work,'step':step,'elapsed_seconds':time.perf_counter()-start,
                             'status':'failed','nonfinite_count':nonfinite,
                             'failure':f'{type(error).__name__}: {error}'})
        raise TrainingFailure(f'Logical step failed and was rolled back: {error}') from error
    finally:
        state._work = None


def _unique_direct(ledger, batch):
    """Coalesce selected offsets by document; never record unscored context."""
    new_unique = 0
    for block in batch.block_slices:
        existing = ledger.coverage.get(block.document_id,[])
        previous = sum(stop-start for start,stop in existing)
        selected = [(offset,offset+1) for offset in block.scored_offsets()]
        merged = []
        for start,stop in sorted([*existing,*selected]):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0],max(stop,merged[-1][1]))
            else:
                merged.append((start,stop))
        ledger.coverage[block.document_id] = merged
        new_unique += sum(stop-start for start,stop in merged)-previous
    return new_unique


def _cursor_marker(cursor):
    if cursor is None:
        return None
    return (cursor.epoch,cursor.permutation_position,cursor.cumulative_target_count,
            cursor._carry,cursor._rng.getstate(),id(cursor._permutation))


def _nonfinite_count(value):
    if isinstance(value,torch.Tensor):
        return int((~torch.isfinite(value)).sum())
    if isinstance(value,dict):
        return sum(_nonfinite_count(v) for v in value.values())
    if isinstance(value,(list,tuple)):
        return sum(_nonfinite_count(v) for v in value)
    if is_dataclass(value):
        return sum(_nonfinite_count(getattr(value,f.name)) for f in fields(value))
    return 0


def _finite(value):
    if isinstance(value,torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value,dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value,(list,tuple)):
        return all(_finite(item) for item in value)
    if is_dataclass(value):
        return all(_finite(getattr(value,f.name)) for f in fields(value))
    return not isinstance(value,float) or math.isfinite(value)


def _equal(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):
        return left.keys() == right.keys() and all(_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):
        return len(left) == len(right) and all(_equal(a,b) for a,b in zip(left,right))
    if is_dataclass(left):
        return type(left) is type(right) and all(_equal(getattr(left,f.name),getattr(right,f.name)) for f in fields(left))
    return left == right


def check_engine(device):
    """Bounded synthetic accumulation/scorer evidence; no scientific acceptance."""
    import hashlib
    from .data.articles import Article
    from .data.blocks import tile_article
    from .models.transformer import ByteTransformer, ModelConfig
    from .optim.adamw import AdamWUpdateRule
    from .optim.base import MethodAdapter, default_parameter_groups
    from .evaluate import score_blocks
    from .models.checks import TOLERANCES
    articles = [Article(raw,[],hashlib.sha256(raw).hexdigest()) for raw in (b'abcdefg',b'x',b'1234')]
    blocks = [block for a in articles for block in tile_article(a)]
    batch = LogicalBatch(blocks,12)
    def create(microbatch):
        with torch.random.fork_rng():
            torch.manual_seed(606)
            model = ByteTransformer(ModelConfig(1,8,2)).to(device)
        model.configure_quantization({'name':'row_dynamic_max'})
        return TrainingState(model,MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule()),
                             microbatch=microbatch,clip_norm=.2)
    one,three,duplicate,reference = create(1),create(3),create(1),create(3)
    max_gradient_error = max_parameter_error = 0.
    for step in range(3):
        reference.model.prepare_quantized_views(step,None)
        reference.model.zero_grad(set_to_none=True)
        (reference.summed_nll(batch)/batch.target_count).backward()
        torch.nn.utils.clip_grad_norm_(reference.model.parameters(),.2,error_if_nonfinite=True)
        reference.method.finish_step(reference,reference.schedule['point'])
        for state in (one,three,duplicate):
            train_step(state,batch)
        for name,p in one.model.named_parameters():
            peer = dict(three.model.named_parameters())[name]
            oracle = dict(reference.model.named_parameters())[name]
            for value in (peer.grad,oracle.grad):
                torch.testing.assert_close(p.grad,value,atol=TOLERANCES['gradient_atol'],rtol=TOLERANCES['gradient_rtol'])
                max_gradient_error = max(max_gradient_error,float((p.grad-value).abs().max()))
            torch.testing.assert_close(p,peer,atol=TOLERANCES['gradient_atol'],rtol=TOLERANCES['gradient_rtol'])
            torch.testing.assert_close(p,oracle,atol=TOLERANCES['gradient_atol'],rtol=TOLERANCES['gradient_rtol'])
            max_parameter_error = max(max_parameter_error,float((p-peer).detach().abs().max()),float((p-oracle).detach().abs().max()))
            assert torch.equal(p,dict(duplicate.model.named_parameters())[name])
            assert one.method.optimizer_states[name]['step'].item() == step+1
    score = score_blocks(one.model,blocks,evaluation_context={'partition':'fixture','seed':606})
    repeated = score_blocks(one.model,blocks,evaluation_context={'partition':'fixture','seed':606})
    assert score == repeated and score.target_count == 12
    assert one.histories['loss_updates'] == 3
    class Degraded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((),device=device))
        def forward(self, inputs, positions, padding_mask=None, *, contexts=None):
            logits = torch.zeros((*inputs.shape,257),device=device)
            logits[...,256] = 20
            return logits
    degraded = score_blocks(Degraded(),blocks,evaluation_context={'partition':'fixture'})
    assert degraded.bpb > score.bpb
    return {'accumulation':{'status':'pass','synthetic':True,'steps':3,'scored_targets_per_step':12,
                            'microbatches':[1,3],'gradient_max_error':max_gradient_error,
                            'parameter_max_error':max_parameter_error,'duplicate_trajectory_exact':True,
                            'oracle':'one full-batch SUM/12, one clip, authoritative H05 update',
                            'tolerances':dict(TOLERANCES),'three_ledgers':{'stream':one.ledger.stream,
                                'passes':one.ledger.passes,'elapsed':one.ledger.elapsed}},
            'scoring':{'status':'pass','nll_sum':score.nll_sum,'target_count':score.target_count,'bpb':score.bpb,
                       'duplicate_equal':True,'degraded_bpb':degraded.bpb,
                       'reduction':'per-block FP32 losses -> explicit host FP64 math.fsum; sorted document/interval/offset'}}
