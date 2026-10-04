"""Shared engine regressions: normalization, commit identity, and rollback."""
import hashlib

import pytest
import torch

from ralup.data.articles import Article
from ralup.data.batches import LogicalBatch
from ralup.data.blocks import tile_article
from ralup.data.cursor import BatchCursor
from ralup.models.transformer import ByteTransformer, ModelConfig
from ralup.optim.adamw import AdamWUpdateRule
from ralup.optim.base import MethodAdapter, SchedulePoint, default_parameter_groups


def article(raw):
    return Article(raw, [], hashlib.sha256(raw).hexdigest())


def fixture(device='cpu', microbatch=1):
    from ralup.train import TrainingState
    with torch.random.fork_rng():
        torch.manual_seed(606)
        model = ByteTransformer(ModelConfig(1,8,2)).to(device)
    model.configure_quantization({'name':'row_dynamic_max'})
    method = MethodAdapter(dict(model.named_parameters()), default_parameter_groups(model), AdamWUpdateRule())
    return TrainingState(model, method, schedule={'step':0,'point':SchedulePoint(.001,.01,0.)},
                         microbatch=microbatch, clip_norm=.2)


def batch():
    blocks = [tile_article(article(raw))[0] for raw in [b'abcdefg', b'x', b'1234']]
    return LogicalBatch(blocks,12)


def test_sum_accumulation_matches_full_mean_and_single_clipped_update(monkeypatch):
    from ralup.train import train_step
    one, three, reference = fixture(), fixture(microbatch=3), fixture()
    reference.model.prepare_quantized_views(0,None)
    reference.summed_nll(batch()).div(12).backward()
    calls = []
    original = torch.nn.utils.clip_grad_norm_
    def clip(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',clip)
    result1 = train_step(one,batch())
    result3 = train_step(three,batch())
    assert len(calls) == 2
    original(reference.model.parameters(),.2)
    reference.method.finish_step(reference, reference.schedule['point'])
    for name,p in one.model.named_parameters():
        torch.testing.assert_close(p.grad,dict(reference.model.named_parameters())[name].grad,atol=1e-7,rtol=1e-5)
        torch.testing.assert_close(p,dict(three.model.named_parameters())[name],atol=1e-7,rtol=1e-5)
        torch.testing.assert_close(p,dict(reference.model.named_parameters())[name],atol=1e-7,rtol=1e-5)
        assert one.method.optimizer_states[name]['step'] == 1
    assert result1.target_count == result3.target_count == 12
    assert result1.work['stream'] == 12 and result1.work['backward'] == 12
    assert one.histories['loss_updates'] == 1 and one.schedule['step'] == 1
    # An unweighted mean of the unequal slice means is detectably different.
    wrong = fixture()
    wrong.model.prepare_quantized_views(0,None)
    for block in batch().block_slices:
        (wrong.summed_nll(LogicalBatch([block],sum(block.score_mask)))/sum(block.score_mask)/3).backward()
    assert any(not torch.allclose(p.grad,dict(wrong.model.named_parameters())[n].grad) for n,p in reference.model.named_parameters())


def test_empty_guard_precedes_context_adapter_and_all_advancement():
    from ralup.train import train_step
    state = fixture()
    with pytest.raises(ValueError,match='empty'):
        train_step(state,LogicalBatch([],0))
    assert not state.model.quantization_contexts and not state.method.proposals
    assert state.schedule['step'] == 0 and not state.ledger.events


@pytest.mark.parametrize('failure',['after_commit','replacement','history','nonfinite'])
def test_failed_transaction_restores_scientific_state_and_charges_work(failure):
    from ralup.train import train_step, select_logical_batch, TrainingFailure
    state = fixture()
    state.cursor = BatchCursor.from_articles([article(b'abc')],seed=17)
    # Force selection across rollover and verify rollback includes permutation/RNG/carry.
    before_cursor = state.cursor.state
    before = {n:p.detach().clone() for n,p in state.model.named_parameters()}
    identities = {n:id(p) for n,p in state.model.named_parameters()}
    chosen = select_logical_batch(state,7)
    assert state.cursor.state == before_cursor
    original = state.method.finish_step
    def broken(training, point):
        result = original(training,point)
        if failure == 'replacement':
            training.model.output.weight = torch.nn.Parameter(training.model.output.weight.detach().clone())
            return result
        if failure == 'history':
            training.histories['loss_updates'] = 2
            return result
        if failure == 'nonfinite':
            with torch.no_grad():
                training.model.output.weight.fill_(float('nan'))
            return result
        training.rng_streams['custom'].random()
        raise RuntimeError('injected after commit')
    import random
    state.rng_streams['custom'] = random.Random(44)
    rng_before = state.rng_streams['custom'].getstate()
    state.method.finish_step = broken
    with pytest.raises(TrainingFailure):
        train_step(state,chosen)
    assert state.cursor.state == before_cursor
    assert state.rng_streams['custom'].getstate() == rng_before
    assert state.schedule['step'] == 0 and not state.histories
    assert state.failed and not state.checkpointable
    assert state.ledger.events[-1]['status'] == 'failed'
    assert state.ledger.events[-1]['backward'] == 7 and state.ledger.stream['targets'] == 0
    for name,p in state.model.named_parameters():
        assert id(p) == identities[name]
        torch.testing.assert_close(p,before[name],atol=0,rtol=0)
    assert all(not v for v in state.method.optimizer_states.values())


def test_success_commits_staged_rollover_once_without_serializing_index(monkeypatch):
    from ralup.train import train_step, select_logical_batch
    state = fixture()
    state.cursor = BatchCursor.from_articles([article(b'abc')],seed=17)
    docs = state.cursor._documents
    monkeypatch.setattr(BatchCursor,'state',property(lambda self: (_ for _ in ()).throw(AssertionError('deep serialization'))))
    selected = select_logical_batch(state,7)
    train_step(state,selected)
    assert state.cursor._documents is docs
    assert state.cursor.cumulative_target_count == 7 and state.cursor.epoch == 2
    assert state.ledger.stream == {'targets':7,'unique':3,'updates':1}


def test_fixed_scale_survives_training_and_evaluation_uses_actual_weights():
    from ralup.train import train_step
    from ralup.evaluate import evaluation_contexts
    state = fixture()
    state.model.configure_quantization({'name':'row_fixed_max'})
    train_step(state,batch())
    scales = {name:c.scales.clone() for name,c in state.model.quantization_contexts.items()}
    actual = evaluation_contexts(state.model,{'partition':'fixture'})
    for name,p in state.model.named_parameters():
        if name in actual:
            assert torch.equal(actual[name].storage_u,p.detach())
            assert torch.equal(actual[name].scales,scales[name])
    train_step(state,batch())
    assert state.histories['loss_updates'] == 2
    for name,c in state.model.quantization_contexts.items():
        assert torch.equal(c.scales,scales[name])


def test_authoritative_commit_is_exact_torch_scalar_order():
    from ralup.train import train_step
    state = fixture()
    before = {name:torch.nn.Parameter(p.detach().clone()) for name,p in state.model.named_parameters()}
    groups = []
    for group in state.method.groups:
        groups.append({'params':[before[n] for n in group.parameter_names],
                       'weight_decay':.01 if group.weight_decay is None else group.weight_decay})
    optimizer = torch.optim.AdamW(groups,lr=.001,betas=(.9,.99),eps=1e-8,foreach=False,fused=False)
    train_step(state,batch())
    for name,p in state.model.named_parameters():
        before[name].grad = p.grad.clone()
    optimizer.step()
    for name,p in state.model.named_parameters():
        assert torch.equal(p,before[name])


def test_failure_restores_cursor_schedule_ledger_and_adapter_nested_mutables():
    from ralup.train import train_step, TrainingFailure
    import random
    state = fixture()
    state.cursor = BatchCursor.from_articles([article(b'abc')],seed=17)
    cursor_before = state.cursor.state
    state.method.custom = {'nested':torch.tensor([2.])}
    state.rng_streams['tensor'] = torch.Generator().manual_seed(17)
    rng_before = state.rng_streams['tensor'].get_state()
    def broken(training,point):
        training.cursor.take(2)
        training.schedule['step'] = 99
        training.ledger.stream['targets'] = 88
        training.method.custom['nested'].add_(3)
        torch.rand(1,generator=training.rng_streams['tensor'])
        training.record_work(additional_loss=3,backward=3,query=2,forward_context=5)
        raise RuntimeError('extra pass failed')
    state.method.finish_step = broken
    with pytest.raises(TrainingFailure):
        train_step(state,batch())
    assert state.cursor.state == cursor_before
    assert state.schedule['step'] == 0 and state.ledger.stream['targets'] == 0
    assert torch.equal(state.method.custom['nested'],torch.tensor([2.]))
    assert torch.equal(state.rng_streams['tensor'].get_state(),rng_before)
    assert state.ledger.passes['additional_loss'] == 3 and state.ledger.passes['backward'] == 15


def test_bootstrap_resolves_prior_engineering_without_future_budgets(tmp_path):
    from ralup.config import resolve_config, UnresolvedConfig
    from ralup.records import sha256_file
    import json
    from pathlib import Path
    template = json.loads((Path(__file__).parents[2]/'configs/core.json').read_bytes())
    artifact = tmp_path/'engineering.json'
    artifact.write_text('{"scope":"ENGINEERING acceptance fixture", "scientific_gate":null}')
    ref = {'path':str(artifact),'sha256':sha256_file(artifact)}
    measured = {'data':{'revision':'pinned','manifest':ref},
                'runtime':{'device':'cpu','dependency_lock':ref,'environment':ref,'tolerances':{'loss_atol':1e-4}},
                'quantization':{'rotation':'identity'},'training':{'microbatch':1,'recipe':'row_fixed_max'},
                'gates':{name:{'status':'pass','evidence':[ref],'blockers':[]} for name in ('WP00','WP01','H02','H04','H05')}}
    resolved = resolve_config(template,measured,purpose='pilot-bootstrap')
    assert resolved.payload['purpose'] == 'pilot-bootstrap'
    assert resolved.payload['training']['continuation_steps'] is None
    assert 'WP04' not in resolved.payload['evidence_gates'] and 'WP06' not in resolved.payload['evidence_gates']
    with pytest.raises(UnresolvedConfig):
        resolve_config(template,measured,purpose='train')
    artifact.write_text('tampered')
    with pytest.raises(ValueError,match='hash'):
        resolve_config(template,measured,purpose='pilot-bootstrap')


def test_cli_requires_explicit_purpose_envelope_and_never_falls_back(tmp_path,capsys):
    from ralup.__main__ import main
    import json
    from pathlib import Path
    template = json.loads((Path(__file__).parents[2]/'configs/smoke.json').read_bytes())
    config = tmp_path/'run.json'
    config.write_text(json.dumps({'config':template,'purpose':'smoke','pilot_target_budget':17,'output':str(tmp_path/'run')}))
    assert main(['run','--phase','explore','--config',str(config)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['synthetic'] is True and printed['disposition'] == 'synthetic_engineering_only'
    assert printed['stream_targets'] == 17
    assert (tmp_path/'run/run.json').exists() and (tmp_path/'run/metrics.jsonl').exists()
    for payload in ({'config':template}, {'config':template,'purpose':'smoke'},
                    {'config':json.loads((Path(__file__).parents[2]/'configs/core.json').read_bytes()),'purpose':'train','pilot_target_budget':17}):
        config.write_text(json.dumps(payload))
        with pytest.raises(SystemExit):
            main(['run','--phase','explore','--config',str(config)])


def test_harness_default_includes_accumulation_and_duplicate_scoring(tmp_path,capsys):
    from ralup.__main__ import main
    import json
    output = tmp_path/'harness.json'
    assert main(['test-harness','--output',str(output)]) == 0
    result = json.loads(output.read_bytes())
    assert result['checks']['coverage']['status'] == 'pass'
    assert result['checks']['model']['status'] == 'pass'
    assert result['checks']['accumulation']['status'] == 'pass'
    assert result['checks']['scoring']['duplicate_equal'] is True
    assert result['scientific_gate'] is None


@pytest.mark.parametrize('field',['history_backward','schedule','cursor','ledger','optimizer_nonfinite'])
def test_adapter_cannot_publish_shared_state_early_or_nonfinite_moments(field):
    from ralup.train import train_step, TrainingFailure
    state = fixture()
    state.cursor = BatchCursor.from_articles([article(b'abc')],seed=17)
    original = state.method.finish_step
    if field == 'history_backward':
        backward = state.method.backward_microbatch
        def changed(training,slice):
            training.histories['loss_updates'] = training.histories.get('loss_updates',0)+1
            return backward(training,slice)
        state.method.backward_microbatch = changed
    def changed_finish(training,point):
        result = original(training,point)
        if field == 'schedule':
            training.schedule['step'] += 1
        elif field == 'cursor':
            training.cursor.take(1)
        elif field == 'ledger':
            training.ledger.stream['updates'] += 1
        elif field == 'optimizer_nonfinite':
            next(iter(training.method.optimizer_states.values()))['exp_avg'].fill_(float('nan'))
        return result
    state.method.finish_step = changed_finish
    with pytest.raises(TrainingFailure):
        train_step(state,batch())
    assert state.schedule['step'] == 0 and state.cursor.cumulative_target_count == 0
    assert state.ledger.stream['updates'] == 0 and not state.histories


def test_cursor_free_repeated_fixture_counts_only_new_scored_intervals():
    from ralup.train import train_step
    state = fixture()
    for _ in range(3):
        train_step(state,batch())
    assert state.ledger.stream == {'targets':36,'unique':12,'updates':3}
    assert len(state.ledger.coverage) == 3


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('invalid',['overlong','empty_span','negative_start','context_offset','past_end'])
def test_invalid_original_target_interval_rejected_before_optimizer_mutation(device,invalid):
    from dataclasses import replace
    import os
    from ralup.data.blocks import TargetBlock
    from ralup.train import train_step
    if device == 'mps':
        if not torch.backends.mps.is_available():
            pytest.skip('Actual MPS unavailable')
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    if invalid == 'overlong':
        block = TargetBlock('fixture',0,129,(256,)+(0,)*128,(0,)*129,
                            tuple(range(129)),(True,)*129,'fixture:oversize')
    else:
        block = tile_article(article(b'a'*132))[1]  # Original interval128..132, with127 context labels.
        if invalid == 'empty_span':
            block = replace(block,target_end=block.target_start)
        elif invalid == 'negative_start':
            block = replace(block,target_start=-1)
        elif invalid == 'context_offset':
            block = replace(block,score_mask=(True,)+block.score_mask[1:])
        else:
            block = replace(block,target_end=131)  # Last selected offset131 lies beyond the shortened interval.
    state = fixture(device=device)
    before = {n:p.detach().clone() for n,p in state.model.named_parameters()}
    with pytest.raises(ValueError,match='target interval'):
        train_step(state,LogicalBatch([block],sum(block.score_mask)))
    from ralup.evaluate import score_blocks
    with pytest.raises(ValueError,match='target interval'):
        score_blocks(state.model,[block],evaluation_context={'partition':'fixture'})
    assert state.checkpointable and not state.failed
    assert state.schedule['step'] == 0 and not state.model.quantization_contexts
    assert not state.histories and not state.ledger.events
    assert all(not v for v in state.method.optimizer_states.values())
    for name,p in state.model.named_parameters():
        assert torch.equal(p,before[name]) and p.grad is None
