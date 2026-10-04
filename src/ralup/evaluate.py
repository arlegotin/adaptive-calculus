"""Common raw-byte SUM loss and read-only, stable-order evaluation.

This module has no official-test source loader. E03 will own that gateway.
Evaluation constructs numerical contexts from actual weights independently of
training caches/history, and sums per-block FP64 target losses on the host with
math.fsum in (document identity, target interval, score offsets) order.
"""
from dataclasses import dataclass
import math
import random

import torch
from torch.nn import functional as F

from .data.batches import LogicalBatch
from .data.blocks import TARGET_BYTES
from .quantization.context import prepare_context


@dataclass(frozen=True)
class ScoreResult:
    nll_sum: float
    target_count: int
    bpb: float
    per_article: dict


def validate_batch(batch: LogicalBatch) -> None:
    count = 0
    for block in batch.block_slices:
        if (type(block.target_start) is not int or type(block.target_end) is not int
                or block.target_start < 0 or not 0 < block.target_end-block.target_start <= TARGET_BYTES):
            raise ValueError('Invalid original target interval: require a positive span <=128 bytes')
        length = len(block.inputs)
        if not 0 < length <= 256 or any(len(v) != length for v in (block.labels,block.positions,block.score_mask)):
            raise ValueError('Invalid byte block dimensions')
        if block.positions != tuple(range(length)):
            raise ValueError('Byte block position IDs must restart at zero')
        if any(type(v) is not int or not 0 <= v <= 256 for v in block.inputs):
            raise ValueError('Invalid input byte/boundary token')
        if any(type(v) is not int or not 0 <= v <= 255 for v in block.labels):
            raise ValueError('Targets must be raw bytes only')
        if any(type(v) is not bool for v in block.score_mask):
            raise ValueError('Loss masks must be boolean')
        if any(not block.target_start <= offset < block.target_end for offset in block.scored_offsets()):
            raise ValueError('Selected byte offset lies outside the original target interval')
        count += sum(block.score_mask)
    if type(batch.target_count) is not int or batch.target_count != count:
        raise ValueError('Declared selected-target count differs from masks')
    if count == 0:
        raise ValueError('Cannot optimize or score an empty logical batch')


def _target_losses(model, batch: LogicalBatch, *, contexts=None, parameter_overrides=None):
    validate_batch(batch)
    device = next(model.parameters()).device
    length = max(len(b.inputs) for b in batch.block_slices)
    shape = (len(batch.block_slices),length)
    inputs = torch.zeros(shape,dtype=torch.int64,device=device)
    positions = torch.zeros_like(inputs)
    labels = torch.zeros_like(inputs)
    padding = torch.ones(shape,dtype=torch.bool,device=device)
    mask = torch.zeros_like(padding)
    for row,block in enumerate(batch.block_slices):
        n = len(block.inputs)
        inputs[row,:n] = torch.tensor(block.inputs,device=device)
        positions[row,:n] = torch.tensor(block.positions,device=device)
        labels[row,:n] = torch.tensor(block.labels,device=device)
        padding[row,:n] = False
        mask[row,:n] = torch.tensor(block.score_mask,device=device)
    if parameter_overrides is None:
        logits = model(inputs,positions,padding,contexts=contexts)
    else:
        # M06's identity-storage learned-weight graph uses the identical model,
        # inputs, activation arithmetic and target scorer. Only the six named
        # projection weights differ; fresh graphs preserve original-W leaves.
        if set(parameter_overrides) != {m['name'] for m in model.quantized_matrix_manifest()}:
            raise ValueError('Functional weight views require exact projection manifest')
        logits = torch.func.functional_call(model,parameter_overrides,(inputs,positions,padding),{'contexts':{}})
    return F.cross_entropy(logits[mask],labels[mask],reduction='none')


def summed_nll(model, batch: LogicalBatch, *, parameter_overrides=None):
    """Scalar SUM for fresh training graphs; H06 alone owns normalization."""
    return _target_losses(model,batch,parameter_overrides=parameter_overrides).sum()


def evaluation_contexts(model, evaluation_context: dict):
    if evaluation_context.get('partition') not in ('fixture','development','meta-train','meta-development','target-training'):
        raise ValueError('Official test targets are protected until the E03 gateway')
    recipe = getattr(model,'_recipe',None)
    if recipe is None:
        return {}
    rng = random.Random(evaluation_context.get('seed',0))
    parameters = dict(model.named_parameters())
    # Fixed scales are runtime scientific state, not recomputed evaluation scales.
    runtime = getattr(model,'_runtime',{})
    return {item['name']:prepare_context(parameters[item['name']],
            {**recipe,**{k:v for k,v in runtime.get(item['name'],{}).items()
                        if k in ('previous_scales','previous_zero_groups','warm_start')}},
            evaluation_context.get('step',0),rng)
            for item in model.quantized_matrix_manifest()}


@torch.no_grad()
def score_blocks(model, blocks, *, evaluation_context: dict) -> ScoreResult:
    contexts = evaluation_contexts(model,evaluation_context)
    # Cursor-backed callers materialize only the declared development/query list.
    ordered = sorted(blocks,key=lambda b:(b.document_id,b.target_start,b.target_end,b.scored_offsets()))
    if not ordered:
        raise ValueError('Cannot score an empty logical batch')
    values, counts, article_values, article_counts = [], [], {}, {}
    seen = set()
    for block in ordered:
        for offset in block.scored_offsets():
            key = (block.document_id,offset)
            if key in seen:
                raise ValueError('Evaluation target byte selected more than once')
            seen.add(key)
        count = sum(block.score_mask)
        losses = _target_losses(model,LogicalBatch([block],count),contexts=contexts)
        # MPS lacks float64: this is an explicit evaluation-only host reduction.
        value = math.fsum(losses.detach().cpu().double().tolist())
        if not math.isfinite(value):
            raise ValueError('Nonfinite evaluation NLL')
        values.append(value)
        counts.append(count)
        article_values.setdefault(block.document_id,[]).append(value)
        article_counts[block.document_id] = article_counts.get(block.document_id,0)+count
    nll, count = math.fsum(values),sum(counts)
    per_article = {key:{'nll_sum':math.fsum(article_values[key]),'target_count':article_counts[key],
                        'bpb':math.fsum(article_values[key])/(math.log(2)*article_counts[key])}
                   for key in sorted(article_values)}
    return ScoreResult(nll,count,nll/(math.log(2)*count),per_article)
