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
