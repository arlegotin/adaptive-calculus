"""Real quantized H06/H07 inner/query isolation and replay (tiny CPU fixture)."""
import copy
import hashlib
import importlib.util
import random
import pytest
import torch
from ralup.data.articles import Article
from ralup.data.blocks import tile_article
from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
from ralup.train import select_logical_batch,train_step,_rng_state,_restore_rng,_equal
from ralup.policy.controllers import Affine45,freeze_controller
from test_policy_identity import policy


def task_fixture(tmp_path):
    assert importlib.util.find_spec('ralup.meta') is not None, 'M11 meta package required'
    from ralup.meta.tasks import MetaTask
    s=policy(tmp_path)
    for _ in range(2):train_step(s,select_logical_batch(s,5))
    snapshot=capture_snapshot(s)
    c=restore_snapshot(snapshot,'cpu');batches=[]
    for _ in range(4):
        b=select_logical_batch(c,5);batches.append(b);train_step(c,b)
    raw=b'query document excluded from inner training'
    q=tile_article(Article(raw,[],hashlib.sha256(raw).hexdigest()))
    docs={b.article_hash:'meta_train' for batch in batches for b in batch.block_slices}
    docs.update({b.article_hash:'meta_train' for b in q})
    return MetaTask(snapshot,batches,q,{'query_seed':17},{'partition':'meta_train','document_partitions':docs,'device':'cpu','synthetic':True}),s


def test_complete_clones_share_inputs_replay_and_preserve_source(tmp_path):
    task,s=task_fixture(tmp_path)
    from ralup.meta.tasks import rollout
    controller=freeze_controller(Affine45())
    before=copy.deepcopy(task.snapshot);rng=_rng_state()
    a=rollout(task,controller,4)
    b=rollout(task,controller,4)
    assert a.objective==b.objective
    assert a.initial_query==b.initial_query and a.final_query==b.final_query
    assert a.work['final_state_hash']==b.work['final_state_hash']
    assert a.work['optimizer_steps']+b.work['optimizer_steps']==8
    assert a.work['query_targets']==2*sum(sum(x.score_mask) for x in task.query_blocks)
    assert _equal(before,task.snapshot) and _equal(rng,_rng_state())
    assert a.work['initial_state_hash']==b.work['initial_state_hash']
    assert a.work['progress_start']==2/16 and a.work['progress_end']==6/16
    for step in a.work['steps']:
        coverage=step['feature_work']
        assert len(coverage['clipped_by_feature'])==14
        assert sum(coverage['clipped_by_feature'])==coverage['clipped']


def test_inner_query_overlap_and_protected_partition_rejected(tmp_path):
    task,_=task_fixture(tmp_path)
    from ralup.meta.tasks import rollout
    task.query_blocks=task.inner_batches[0].block_slices
    with pytest.raises(ValueError,match='disjoint'):rollout(task,freeze_controller(Affine45()),1)
    task,_=task_fixture(tmp_path)
    task.provenance['document_partitions'][task.query_blocks[0].article_hash]='development'
    with pytest.raises(ValueError,match='META-TRAIN'):rollout(task,freeze_controller(Affine45()),1)


def test_different_inner_list_fails_cursor_replay(tmp_path):
    task,_=task_fixture(tmp_path)
    from ralup.meta.tasks import rollout
    task.inner_batches[0],task.inner_batches[1]=task.inner_batches[1],task.inner_batches[0]
    with pytest.raises(ValueError,match='cursor'):rollout(task,freeze_controller(Affine45()),2)


def test_weight_only_reset_changes_future_loss(tmp_path):
    task,_=task_fixture(tmp_path)
    from ralup.meta.tasks import rollout
    correct=rollout(task,freeze_controller(Affine45()),4)
    cold=policy(tmp_path);warm=restore_snapshot(task.snapshot,'cpu')
    cold.model.load_state_dict(warm.model.state_dict())
    # Authentic but cold reset is NOT the same starting state or objective.
    cold.cursor=warm.cursor
    task.snapshot=capture_snapshot(cold)
    mutant=rollout(task,freeze_controller(Affine45()),4)
    assert correct.work['initial_state_hash']!=mutant.work['initial_state_hash']
    assert correct.final_query!=mutant.final_query


def test_generation_pairs_match_full_inputs_and_charge_all_work(tmp_path):
    task,_=task_fixture(tmp_path)
    from ralup.meta import es
    assert hasattr(es,'train_generation'),'M11 real antithetic generation required'
    state=freeze_controller(Affine45());outer=es.OuterAdam(state.parameters)
    eps=torch.randn(2,45,generator=torch.Generator().manual_seed(11))
    record=es.train_generation(task,state.family,outer,eps,2,.05)
    assert record['optimizer_steps']==8
    assert record['outer_step']==1
    for pair in record['pairs']:
        assert pair['plus']['work']['input_sha256']==pair['minus']['work']['input_sha256']
        assert pair['plus']['work']['initial_state_hash']==pair['minus']['work']['initial_state_hash']
    assert outer.phi.grad_fn is None


def test_nonsynthetic_task_cannot_use_fixture_or_self_declared_membership(tmp_path):
    task,_=task_fixture(tmp_path)
    from ralup.meta.tasks import rollout
    task.provenance['synthetic']=False
    with pytest.raises(ValueError,match='indexed'):rollout(task,freeze_controller(Affine45()),1)


def test_failed_rollout_retains_completed_work_journal(tmp_path):
    task,_=task_fixture(tmp_path)
    from ralup.meta.tasks import rollout
    task.provenance['journal']=str(tmp_path/'rollout.jsonl')
    task.inner_batches[1],task.inner_batches[2]=task.inner_batches[2],task.inner_batches[1]
    with pytest.raises(ValueError,match='cursor'):rollout(task,freeze_controller(Affine45()),3)
    import json
    rows=[json.loads(s) for s in (tmp_path/'rollout.jsonl').read_text().splitlines()]
    assert sum(r.get('work',{}).get('stream',0) for r in rows if r['event']=='step')==5
    assert rows[-1]['event']=='failure'


def campaign_fixture(tmp_path):
    import json
    from dataclasses import asdict
    from ralup.records import sha256_file
    from ralup.state import save_snapshot
    from ralup.search import source_identity
    task,_=task_fixture(tmp_path)
    ref=save_snapshot(task.snapshot,tmp_path/'pool')
    p=tmp_path/'queries.json';p.write_text(json.dumps({'blocks':[asdict(b) for b in task.query_blocks]}))
    return {'schema':'ralup.meta_campaign.v1','purpose':'synthetic-smoke','synthetic':True,'source_identity':source_identity(),
        'generations':1,'horizon':2,'pairs':2,'seed':17,'device':'cpu','output':str(tmp_path/'campaign'),
        'pool':[{'snapshot':{'path':str(ref.path),'sha256':ref.sha256},'queries':{'path':str(p),'sha256':sha256_file(p)},'provenance':task.provenance}],
        'family':'affine45','parameters':freeze_controller(Affine45()).parameters.tolist(),'weights':[1.],
        'sigma':.05,'outer_learning_rate':.01,'logical_batch_targets':5,'maximum_optimizer_steps':8,'maximum_seconds':60.}


def test_campaign_writes_complete_outer_rng_and_charged_results(tmp_path):
    import json
    from ralup.meta.es import run_campaign
    value=campaign_fixture(tmp_path);path=tmp_path/'campaign.json';path.write_text(json.dumps(value))
    result=run_campaign(path)
    assert result['status']=='synthetic_smoke_completed' and not result['confirmation_eligible']
    output=tmp_path/'campaign';state=json.loads((output/'outer-1.json').read_bytes())
    assert len(state['sampling_rng'])==3 and state['sampling_rng'][0]==3
    assert state['outer']['step_count']==1
    journal=[json.loads(s) for s in (output/'generation-1-journal.jsonl').read_text().splitlines()]
    assert sum(r.get('work',{}).get('stream',0) for r in journal if r['event']=='step')==40
    record=json.loads((output/'generation-1.json').read_bytes());assert record['optimizer_steps']==8


def test_campaign_rejects_self_declared_real_fixture_and_over_budget(tmp_path):
    import json
    from ralup.meta.es import run_campaign
    value=campaign_fixture(tmp_path);path=tmp_path/'campaign.json'
    value['maximum_optimizer_steps']=7;path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='budget'):run_campaign(path)
    value['maximum_optimizer_steps']=8;value['purpose']='exploratory-meta-training';value['synthetic']=False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='synthetic|Genuine|protocol'):run_campaign(path)
    assert not (tmp_path/'campaign').exists()


def test_campaign_resume_restores_both_rngs_and_outer_moments(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from ralup.meta import es
    from ralup.records import sha256_file
    value=campaign_fixture(tmp_path)
    value.update(generations=2,pairs=1,horizon=1,maximum_optimizer_steps=4)
    config=tmp_path/'campaign.json';config.write_text(json.dumps(value))
    complete=es.run_campaign(config)
    complete_state=json.loads((tmp_path/'campaign'/'outer-2.json').read_bytes())
    value['output']=str(tmp_path/'interrupted');config.write_text(json.dumps(value))
    real_generation=es.train_generation
    def interrupt_second(task,family,outer,*args,**kwargs):
        if outer.step_count==1:raise RuntimeError('deliberate interruption')
        return real_generation(task,family,outer,*args,**kwargs)
    monkeypatch.setattr(es,'train_generation',interrupt_second)
    with pytest.raises(RuntimeError,match='deliberate interruption'):es.run_campaign(config)
    checkpoint=tmp_path/'interrupted'/'outer-1.json'
    assert json.loads((tmp_path/'interrupted'/'failure.json').read_bytes())['completed_generations']==1
    value['output']=str(tmp_path/'resumed')
    value['resume']={'path':str(checkpoint),'sha256':sha256_file(checkpoint)}
    config.write_text(json.dumps(value));monkeypatch.setattr(es,'train_generation',real_generation)
    resumed=es.run_campaign(config)
    assert resumed['parameters']==complete['parameters']
    resumed_state=json.loads((Path(value['output'])/'outer-2.json').read_bytes())
    for key in ('outer','sampling_rng','perturbation_rng'):
        assert resumed_state[key]==complete_state[key]


def test_rollout_deadline_stops_before_spending_and_retains_failure(tmp_path):
    import json,time
    from ralup.meta.tasks import rollout
    task,_=task_fixture(tmp_path)
    journal=tmp_path/'expired.jsonl'
    task.provenance.update(deadline_monotonic=time.perf_counter()-1,journal=str(journal))
    with pytest.raises(RuntimeError,match='deadline'):
        rollout(task,freeze_controller(Affine45()),1)
    rows=[json.loads(s) for s in journal.read_text().splitlines()]
    assert len(rows)==1 and rows[0]['event']=='failure'


def test_controller_rebind_preserves_completed_action_telemetry(tmp_path,monkeypatch):
    from ralup.meta import tasks
    task,_=task_fixture(tmp_path)
    expected=copy.deepcopy(restore_snapshot(task.snapshot,'cpu').method.last_actions)
    assert expected
    real_step=tasks.train_step
    def observed_step(state,batch):
        assert _equal(state.method.last_actions,expected)
        return real_step(state,batch)
    monkeypatch.setattr(tasks,'train_step',observed_step)
    tasks.rollout(task,freeze_controller(Affine45()),1)


def test_rollout_retains_completed_actions_with_disposable_final_state(tmp_path):
    """S32 action/bound evidence survives deletion of successful final tensors."""
    import json
    from ralup.meta.tasks import rollout
    from ralup.policy.controllers import ControllerState
    task,_=task_fixture(tmp_path)
    task.provenance['journal']=str(tmp_path/'actions.jsonl')
    parameters=freeze_controller(Affine45()).parameters
    parameters[-3:]=torch.tensor([.2,.05,.875])
    result=rollout(task,ControllerState('affine45',parameters),2)
    assert result.final_snapshot_ref is None
    journal=[json.loads(line) for line in (tmp_path/'actions.jsonl').read_text().splitlines()]
    rows=[row['work'] for row in journal if row['event']=='step']
    assert len(rows)==2
    for work,retained in zip(rows,result.work['steps']):
        actions=work['policy_actions']
        assert actions==retained['policy_actions']
        assert set(actions)==set(work['matrices'])
        for info in actions.values():
            assert {'alpha','h','tau','assistance_bypassed','accepted_bound_excess_max'}<=set(info)
            assert info['alpha']>0 and info['tau']<1
