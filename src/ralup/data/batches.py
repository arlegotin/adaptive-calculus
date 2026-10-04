"""Logical byte selection is independent of physical packing."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .blocks import TargetBlock

if TYPE_CHECKING:
    from .cursor import BatchCursor


@dataclass(frozen=True)
class LogicalBatch:
    block_slices: list[TargetBlock]
    target_count: int


def next_logical_batch(cursor: 'BatchCursor', target_budget: int = 8192) -> LogicalBatch:
    if type(target_budget) is not int or target_budget <= 0:
        raise ValueError('Target budget must be a positive integer')
    remaining = target_budget
    if cursor.remaining_run_targets is not None:
        remaining = min(remaining, cursor.remaining_run_targets)
    slices = []
    selected = 0
    while remaining and cursor.block_count:
        block = cursor.take(remaining)
        count = sum(block.score_mask)
        slices.append(block)
        selected += count
        remaining -= count
    return LogicalBatch(slices, selected)


def pack_microbatches(batch: LogicalBatch, sequences_per_microbatch: int) -> list[LogicalBatch]:
    """Group selected original sequences without padding or reselection.

    Tensor consumers may pad these groups with false score masks. Their summed
    NLL denominator remains the logical batch's target_count, including a short
    final run batch. No microbatch field enters the cursor state.
    """
    if type(sequences_per_microbatch) is not int or sequences_per_microbatch <= 0:
        raise ValueError('Physical packing must be a positive integer')
    return [LogicalBatch(group, sum(sum(b.score_mask) for b in group))
            for start in range(0, len(batch.block_slices), sequences_per_microbatch)
            if (group := batch.block_slices[start:start + sequences_per_microbatch])]
