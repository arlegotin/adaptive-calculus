"""Raw byte sums, duplicate equality, and read-only actual-weight scoring."""
import math
import pytest
import torch
from test_logical_step import article, fixture, batch
from ralup.data.blocks import tile_article


def test_duplicate_degraded_and_byte_coverage_are_read_only():
    from ralup.evaluate import score_blocks
    from ralup.train import train_step
    state = fixture()
    train_step(state,batch())
    articles = [article(b'abc'),article(b'x'*129)]
    blocks = [block for a in articles for block in tile_article(a)]
    before = {n:p.clone() for n,p in state.model.named_parameters()}
    contexts = state.model.quantization_contexts
    history = {n:v for n,v in state.histories.items()}
    rng = torch.get_rng_state().clone()
    score = score_blocks(state.model,blocks,evaluation_context={'partition':'fixture','seed':4})
    duplicate = score_blocks(state.model,blocks,evaluation_context={'partition':'fixture','seed':4})
    assert score == duplicate
    assert score.target_count == sum(len(a.content) for a in articles)
    assert score.bpb == score.nll_sum/(math.log(2)*score.target_count)
    assert sum(a['target_count'] for a in score.per_article.values()) == score.target_count
    assert torch.equal(torch.get_rng_state(),rng)
    assert state.model.quantization_contexts == contexts and state.histories == history
    for n,p in state.model.named_parameters():
        assert torch.equal(p,before[n])
    class Degraded(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
        def forward(self,inputs,positions,padding_mask=None,*,contexts=None):
            logits = torch.zeros((*inputs.shape,257),device=inputs.device)
            logits[...,256] = 20
            return logits
    assert score_blocks(Degraded(),blocks,evaluation_context={'partition':'fixture'}).bpb > score.bpb


def test_official_test_and_nonbyte_target_rejected():
    from ralup.evaluate import score_blocks
    from dataclasses import replace
    state = fixture()
    with pytest.raises(ValueError,match='protected'):
        score_blocks(state.model,batch().block_slices,evaluation_context={'partition':'test'})
    block = batch().block_slices[0]
    with pytest.raises(ValueError,match='raw.byte'):
        score_blocks(state.model,[replace(block,labels=(256,)+block.labels[1:])],evaluation_context={'partition':'fixture'})


def test_block_progress_cap_stops_before_next_forward_and_preserves_state():
    from ralup.evaluate import score_blocks
    from ralup.state import exact_cpu_state_hash
    from ralup.train import train_step
    state = fixture(); train_step(state,batch())
    blocks = tile_article(article(b'x'*270))
    before = exact_cpu_state_hash(state); forwards=[]; observed=[]
    hook = state.model.register_forward_pre_hook(lambda *args: forwards.append(True))
    def monitor(row):
        observed.append(row)
        if row['completed_blocks'] == 1:
            raise TimeoutError('declared block boundary cap')
    try:
        with pytest.raises(TimeoutError,match='block boundary cap'):
            score_blocks(state.model,blocks,evaluation_context={'partition':'fixture'},progress_callback=monitor)
    finally: hook.remove()
    assert len(forwards) == 1
    assert observed[-1]['completed_targets'] == 128
    assert observed[-1]['block_targets'] == 128
    assert observed[-1]['forward_context'] == len(blocks[0].inputs)
    assert exact_cpu_state_hash(state) == before


def test_progress_observation_preserves_exact_full_score_and_actual_counts():
    from ralup.evaluate import score_blocks
    state = fixture(); blocks=tile_article(article(b'x'*270))
    baseline=score_blocks(state.model,blocks,evaluation_context={'partition':'fixture'})
    rows=[]
    observed=score_blocks(state.model,blocks,evaluation_context={'partition':'fixture'},progress_callback=rows.append)
    assert observed == baseline
    completed=[r for r in rows if r['phase']=='block_completed']
    assert [r['block_targets'] for r in completed] == [128,128,14]
    assert rows[-1]['completed_targets'] == 270
    assert rows[-1]['forward_context'] == sum(len(b.inputs) for b in blocks)
    assert sum(r['block_nll_sum'] for r in completed) == pytest.approx(baseline.nll_sum,abs=1e-12)


def test_evaluation_hashes_complete_training_state_unchanged():
    import hashlib
    import random
    from dataclasses import fields, is_dataclass
    from collections.abc import Mapping
    from ralup.evaluate import score_blocks
    from ralup.train import train_step
    def digest(value):
        if isinstance(value,torch.Tensor):
            return (str(value.dtype),tuple(value.shape),value.detach().cpu().reshape(-1).tolist())
        if isinstance(value,Mapping):
            return tuple((key,digest(item)) for key,item in sorted(value.items()))
        if isinstance(value,(list,tuple)):
            return tuple(digest(item) for item in value)
        if is_dataclass(value):
            return tuple((f.name,digest(getattr(value,f.name))) for f in fields(value))
        if isinstance(value,torch.Generator):
            return digest(value.get_state())
        if isinstance(value,random.Random):
            return digest(value.getstate())
        return value
    def state_hash(state):
        data = {'model':state.model.state_dict(),'gradients':{n:p.grad for n,p in state.model.named_parameters()},
                'contexts':state.model.quantization_contexts,'optimizer':state.optimizer_state,
                'quantizer':state.quantizer_state,'histories':state.histories,'schedule':state.schedule,
                'cursor':None if state.cursor is None else state.cursor.state,'rng':state.rng_streams,
                'global_rng':torch.get_rng_state(),'python_rng':random.getstate(),
                'ledger':{'stream':state.ledger.stream,'passes':state.ledger.passes,'elapsed':state.ledger.elapsed,'events':state.ledger.events}}
        return hashlib.sha256(repr(digest(data)).encode()).hexdigest()
    state = fixture()
    state.rng_streams = {'data':random.Random(7),'evaluation_unused':torch.Generator().manual_seed(4)}
    train_step(state,batch())
    before = state_hash(state)
    score_blocks(state.model,batch().block_slices,evaluation_context={'partition':'fixture','seed':3})
    assert state_hash(state) == before


@pytest.mark.parametrize('device',['cpu','mps'])
def test_partial_carry_and_short_final_block_preserve_original_offset_semantics(device):
    from dataclasses import replace
    import os
    from ralup.evaluate import score_blocks
    from ralup.data.batches import LogicalBatch
    from ralup.train import train_step
    if device == 'mps':
        if not torch.backends.mps.is_available():
            pytest.skip('Actual MPS unavailable')
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    blocks = tile_article(article(b'a'*260))
    original = blocks[1]  # Full context and original interval128..256.
    offsets = {155,156,157,158,159,160,161}
    mask = tuple(selected and offset in offsets for selected,offset in
                 zip(original.score_mask,range(1,256)))
    carried = replace(original,score_mask=mask)
    assert carried.scored_offsets() == tuple(sorted(offsets))
    assert carried.inputs == original.inputs and carried.positions == original.positions
    short = blocks[2]
    assert short.scored_offsets() == (256,257,258,259)
    state = fixture(device=device)
    result = train_step(state,LogicalBatch([carried,short],11))
    score = score_blocks(state.model,[carried,short],evaluation_context={'partition':'fixture'})
    assert result.target_count == score.target_count == 11
    assert state.ledger.stream['unique'] == 11
