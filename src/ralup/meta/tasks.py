"""H06/H07 rollout boundary: complete reset, causal deployment, disjoint queries."""
from dataclasses import dataclass
import copy
import hashlib
import math
import os
from pathlib import Path
import time
import torch
from ..config import canonical_bytes
from ..data.batches import LogicalBatch
from ..data.blocks import TargetBlock
from ..evaluate import score_blocks,validate_batch
from ..policy.adapter import PolicyAdapter
from ..policy.controllers import ControllerState,Affine45,LearnedLR49,Static3
from ..records import ArtifactRef
from ..state import TrainingSnapshot,capture_snapshot,restore_snapshot,exact_cpu_state_hash,save_snapshot
from ..train import _rng_state,_restore_rng,_synchronize,select_logical_batch,train_step


@dataclass
class MetaTask:
    snapshot: TrainingSnapshot
    inner_batches: list[LogicalBatch]
    query_blocks: list[TargetBlock]
    rng_bundle: dict
    provenance: dict

    def validate(self,horizon):
        if type(horizon) is not int or not 0<horizon<=len(self.inner_batches):raise ValueError('Invalid rollout horizon')
        if self.provenance.get('partition')!='meta_train':raise ValueError('Outer objectives require META-TRAIN only')
        if set(self.rng_bundle)!={'query_seed'} or type(self.rng_bundle['query_seed']) is not int:raise ValueError('Explicit query evaluation seed required')
        if not self.query_blocks:raise ValueError('Nonempty query list required')
        for batch in self.inner_batches:validate_batch(batch)
        inner={b.article_hash for batch in self.inner_batches for b in batch.block_slices}
        query={b.article_hash for b in self.query_blocks}
        if inner&query:raise ValueError('Inner/query documents must be disjoint')
        partitions=self.provenance.get('document_partitions',{})
        if any(partitions.get(h)!='meta_train' for h in inner|query):raise ValueError('Every document must belong to META-TRAIN')


@dataclass(frozen=True)
class RolloutResult:
    objective: float
    initial_query: float
    final_query: float
    work: dict
    final_snapshot_ref: ArtifactRef | None


def _input_digest(task,horizon):
    def block(b):return {k:list(v) if isinstance(v,tuple) else v for k in ('article_hash','document_id','target_start','target_end','inputs','labels','positions','score_mask') for v in [getattr(b,k)]}
    return hashlib.sha256(canonical_bytes({'inner':[[block(b) for b in batch.block_slices] for batch in task.inner_batches[:horizon]],'query':[block(b) for b in task.query_blocks],'evaluation':task.rng_bundle})).hexdigest()


def authenticate_queries(cursor,query_blocks):
    """Verify source-scoped IDs and materialized bytes without scoring."""
    if cursor is None or cursor._source.get('kind')!='indexed' or cursor._source.get('partition')!='meta-train':raise ValueError('Genuine indexed META-TRAIN snapshot required')
    from ..data.blocks import make_block
    prefix=cursor._source['hashes']['source_sha256']+':'
    documents={prefix+d['id']:d for d in cursor._documents}
    for b in query_blocks:
        d=documents.get(b.document_id)
        if d is None or d['content_sha256']!=b.article_hash or not 0<=b.target_start<b.target_end<=d['bytes']:raise ValueError('Query is outside authenticated META-TRAIN documents')
        start=max(0,b.target_start-128)
        with Path(d['content_path']).open('rb') as f:
            if cursor._stat_key(os.fstat(f.fileno()))!=cursor._content_stats[d['content_path']]:raise ValueError('Query source changed')
            f.seek(d['content_start']+start);raw=f.read(b.target_end-start)
        expected=make_block(b.article_hash,b.target_start,b.target_end,raw[:b.target_start-start],raw[b.target_start-start:],document_id=b.document_id)
        if (b.inputs,b.labels,b.positions)!=(expected.inputs,expected.labels,expected.positions) or len(b.score_mask)!=len(expected.score_mask) or any(x and not y for x,y in zip(b.score_mask,expected.score_mask)):raise ValueError('Query bytes/masks differ from authenticated source')


def rollout(task: MetaTask,controller,horizon: int) -> RolloutResult:
    """Sequential rollout. Output checkpoints are optional declared milestones.

    Source state and global RNG survive unchanged. Rebinding a candidate keeps
    every original-W moment/history/cursor owner; only frozen deployment changes.
    No query outcomes enter the adapter, state histories or gradient callback.
    """
    task.validate(horizon)
    if type(controller) is not ControllerState:raise ValueError('Frozen finite controller required')
    outer_rng=_rng_state();start=time.perf_counter()
    journal=None
    if task.provenance.get('journal'):
        journal=Path(task.provenance['journal']).open('a')
    def record(value):
        if journal is not None:
            journal.write(canonical_bytes(value).decode()+'\n');journal.flush();os.fsync(journal.fileno())
    def deadline():
        limit=task.provenance.get('deadline_monotonic')
        if limit is not None:
            if type(limit) not in (int,float) or not math.isfinite(limit):
                raise ValueError('Finite rollout deadline required')
            if time.perf_counter()>limit:
                raise RuntimeError('Frozen rollout deadline exhausted')
    try:
        deadline()
        record({'event':'rollout_start','family':controller.family,'parameters':controller.parameters.tolist(),'horizon':horizon})
        state=restore_snapshot(task.snapshot,task.provenance.get('device','cpu'))
        if task.provenance.get('synthetic') is not True:
            authenticate_queries(state.cursor,task.query_blocks)
        # A fork must not append to its source run ledger.
        state.ledger.run_ledger=None
        initial_hash=exact_cpu_state_hash(state)
        old=state.method
        if type(old) is not PolicyAdapter:raise ValueError('Pool snapshots require instrumented frozen policy/reference adapter')
        training_rng=_rng_state()
        fresh=LearnedLR49(max_multiplier=4 if controller.family.endswith('shared_range') else 32) if controller.family.startswith('learned_lr49') else Static3() if controller.family=='static3' else Affine45()
        offset=0
        with torch.no_grad():
            for p in fresh.parameters():
                p.copy_(controller.parameters[offset:offset+p.numel()].reshape(p.shape));offset+=p.numel()
        method=PolicyAdapter(old.rule,fresh,old.normalizer,normalizer_ref=old.normalizer_ref,family=controller.family).bind(dict(state.model.named_parameters()),old.groups)
        _restore_rng(training_rng)
        method.optimizer_states=old.optimizer_states
        method.last_actions=copy.deepcopy(old.last_actions)
        state.method=method;state.optimizer_state=method.optimizer_states
        full=state.configuration['policy_features']['full_horizon'];step=state.schedule['step']
        if step+horizon>full:raise ValueError('Rollout exceeds intended full continuation horizon')
        _synchronize(state.model);reset_seconds=time.perf_counter()-start
        query_seconds=0.;query_targets=0
        def query():
            nonlocal query_seconds,query_targets
            rng=_rng_state();t=time.perf_counter()
            try:
                def progress(value):
                    record({'event':'query_progress',**value})
                    deadline()
                scored=score_blocks(state.model,task.query_blocks,evaluation_context={'partition':'meta-train','seed':task.rng_bundle['query_seed']},progress_callback=progress)
                _synchronize(state.model);query_targets+=scored.target_count
                return scored.nll_sum/scored.target_count
            finally:
                _restore_rng(rng);query_seconds+=time.perf_counter()-t
        initial=query();steps=[]
        for expected in task.inner_batches[:horizon]:
            deadline()
            actual=select_logical_batch(state,expected.target_count)
            if actual!=expected:raise ValueError('Frozen inner batch list differs from complete snapshot cursor replay')
            result=train_step(state,actual)
            steps.append(result.work)
            record({'event':'step','work':result.work,'inputs':[{'document_id':b.document_id,'offsets':list(b.scored_offsets())} for b in actual.block_slices]})
        final=query();snapshot_start=time.perf_counter()
        final_hash=exact_cpu_state_hash(state)
        ref=None
        if task.provenance.get('final_snapshot_directory'):
            ref=save_snapshot(capture_snapshot(state),task.provenance['final_snapshot_directory'])
        snapshot_seconds=time.perf_counter()-snapshot_start
        work={'optimizer_steps':horizon,'training_targets':sum(b.target_count for b in task.inner_batches[:horizon]),
            'query_targets':query_targets,'query_evaluations':2,'initial_query_reused':False,
            'reset_seconds':reset_seconds,'query_seconds':query_seconds,'snapshot_seconds':snapshot_seconds,
            'elapsed_seconds':time.perf_counter()-start,'initial_state_hash':initial_hash,'final_state_hash':final_hash,
            'input_sha256':_input_digest(task,horizon),'progress_start':step/full,'progress_end':state.schedule['step']/full,'steps':steps,
            'retention':'complete source snapshot plus frozen controller/input/RNG reconstruct successful disposable final state' if ref is None else 'declared final milestone'}
        return RolloutResult((final-initial)/max(initial,1e-6),initial,final,work,ref)
    except BaseException as error:
        record({'event':'failure','error':str(error),'elapsed_seconds':time.perf_counter()-start,'unknown_partial_work':True})
        raise
    finally:
        _restore_rng(outer_rng)
        if journal is not None:journal.close()


def sample_meta_task(pool,rng,recipe):
    """Sample only a prospectively weighted, complete baseline/policy age pool."""
    weights=recipe['weights']
    if len(pool)!=len(weights) or not pool or any(type(w) not in (int,float) or not torch.isfinite(torch.tensor(w)) or w<0 for w in weights) or sum(weights)<=0:raise ValueError('Invalid frozen pool mixture')
    task=rng.choices(pool,weights=weights,k=1)[0]
    task.validate(recipe['horizon'])
    return task
