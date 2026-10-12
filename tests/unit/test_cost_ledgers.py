"""Hand-counted executed passes and shared ancestry, not additive role labels."""
import copy
import pytest


def event(name, targets, *, stream=0, kind='loss', start=0, roles=()):
    return {'event_id':name, 'kind':kind, 'targets':targets, 'stream_targets':stream,
            'classifications':list(roles), 'source_intervals':[['corpus','article',start,start+targets]]}


def run(name, events, *, parents=(), start=0, end=10, score=None, status='completed'):
    return {'run_id':name, 'family':name, 'status':status, 'parents':list(parents),
            'events':events, 'wall_intervals':[{'clock':'UTC','start':start,'end':end}],
            'wall_scope':{'complete':True,'boundary':'declared_synthetic_component'},
            'development_bpb':score, 'common_engineering_seconds':None}


def test_queries_roles_ancestry_and_nested_wall_are_counted_once():
    from ralup.analysis.cost import cost_quality_frontier
    parent=run('warm',[event('warm',4,stream=4)],end=4)
    a=run('extra',[event('train-a',6,stream=6,start=4),event('q',3,roles=('query','additional_loss','backward')),
                   event('hvp',3,kind='hvp')],parents=[parent],start=4,end=14,score=2.)
    b=run('ordinary',[event('train-b',6,stream=6,start=4)],parents=[parent],start=8,end=18,score=2.1)
    a['wall_intervals'].append({'clock':'UTC','start':5,'end':7})
    result=cost_quality_frontier([a,b],'total')
    assert result['totals']['stream_targets']==16
    assert result['totals']['loss_targets']==19
    assert result['totals']['hvp_targets']==3
    assert result['totals']['total_targets']==22
    assert result['totals']['unique_targets']==10
    assert result['totals']['seconds']==18
    assert result['totals']['common_engineering_seconds'] is None
    assert result['matched_curves'][0]['total_targets']==16
    assert result['matched_curves'][0]['stream_targets']==10
    assert result['matched_curves'][1]['total_targets']==10  # ordinary has no unused policy scans


def test_failures_charge_cost_without_becoming_quality_points():
    from ralup.analysis.cost import cost_quality_frontier
    runs=[run('failed',[event('partial',3)],end=5,status='failed'),
          run('good',[event('ok',4,stream=4)],start=5,end=12,score=3.)]
    result=cost_quality_frontier(runs,'seconds')
    assert result['frontier']==[{'run_id':'good','cost':12,'development_bpb':3.}]
    assert result['totals']['total_targets']==7
    assert result['failed_runs']==['failed']


def test_distinct_gradient_only_passes_are_not_lost_or_double_counted():
    from ralup.analysis.cost import cost_quality_frontier
    r=run('r',[event('loss',4,stream=4,roles=('backward',)),event('grad',4,kind='gradient_only')],score=1.)
    assert cost_quality_frontier([r],'total')['totals']['total_targets']==8
    r['events'].append(copy.deepcopy(r['events'][0]))
    assert cost_quality_frontier([r],'total')['totals']['total_targets']==8
    r['events'][-1]['targets']=5
    with pytest.raises(ValueError,match='identity'):
        cost_quality_frontier([r],'total')


def test_missing_historical_offsets_and_unknown_partial_work_remain_unknown():
    from ralup.analysis.cost import cost_quality_frontier
    r=run('partial',[event('known',3)],status='failed')
    r['events'][0]['source_intervals']=None
    r['unknown_partial_work']=True
    totals=cost_quality_frontier([r],'unique')['totals']
    assert totals['unique_targets'] is None
    assert totals['total_targets'] is None
    assert totals['known_total_targets']==3


@pytest.mark.parametrize('field,value',[('targets',-1),('targets',True),('kind','query'),('stream_targets',5)])
def test_invalid_counts_and_ambiguous_pass_kinds_reject(field,value):
    from ralup.analysis.cost import cost_quality_frontier
    e=event('bad',4);e[field]=value
    with pytest.raises(ValueError):cost_quality_frontier([run('bad',[e])],'total')


def test_h06_records_each_actual_loss_execution_and_backward_launch():
    import hashlib
    from ralup.models.transformer import ByteTransformer,ModelConfig
    from ralup.optim.base import MethodAdapter,default_parameter_groups
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.train import TrainingState,train_step
    from ralup.data.articles import Article
    from ralup.data.cursor import BatchCursor
    from ralup.data.batches import next_logical_batch
    model=ByteTransformer(ModelConfig(1,8,2))
    state=TrainingState(model,MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule()),microbatch=1)
    raw=b'four';article=Article(raw,[],hashlib.sha256(raw).hexdigest())
    cursor=BatchCursor.from_articles([article],seed=1,run_target_budget=4)
    result=train_step(state,next_logical_batch(cursor,4))
    events=result.work['loss_pass_events']
    assert len(events)==1
    assert events[0]['targets']==4 and events[0]['backward_launched'] is True
    assert events[0]['forward_completed'] is True
    assert events[0]['source_offsets'][0]['offsets']==[0,1,2,3]


def test_search_result_cost_uses_physical_microbatches_and_query_passes(tmp_path):
    from ralup.analysis.cost import trial_cost_record,cost_quality_frontier
    from ralup.search import _write
    report={'trial_id':'trial','family':'ordinary','status':'completed','curve':[{'development_bpb':2.}],
            'executions':[event('trial:mb0',4,stream=4),event('trial:mb1',4,stream=4,start=4),
                          event('trial:query0',3,roles=('query','additional_loss'))],
            'cost':{'parent':None,'wall_interval':{'clock':'UTC','start':0,'end':9},
                    'common_unknown_engineering_seconds':None},
            'unknown_partial_query_work':False,'unknown_partial_training_work':False}
    ref=_write(tmp_path/'report.json',report)
    run=trial_cost_record(ref)
    totals=cost_quality_frontier([run],'total')['totals']
    assert totals['physical_events']==3 and totals['total_targets']==11
    assert totals['stream_targets']==8 and totals['unique_targets']==8


def test_legacy_search_records_cannot_fabricate_historical_pass_ids(tmp_path):
    from ralup.analysis.cost import trial_cost_record,cost_quality_frontier
    from ralup.search import _write
    ref=_write(tmp_path/'legacy.json',{'trial_id':'old','family':'ordinary','status':'completed','curve':[],
        'cost':{'complete_seconds':9,'parent':None,'ledger':{'passes':{'loss_targets':99}}}})
    run=trial_cost_record(ref)
    assert run['events']==[]
    totals=cost_quality_frontier([run],'total')['totals']
    assert totals['total_targets'] is None and totals['seconds'] is None
    assert run['historical_raw_cost']['ledger']['passes']['loss_targets']==99


def test_missing_source_ranges_cannot_claim_zero_unique_exposure():
    from ralup.analysis.cost import cost_quality_frontier
    e=event('x',4);e['source_intervals']=[]
    with pytest.raises(ValueError,match='source'):
        cost_quality_frontier([run('x',[e])],'unique')


def test_ewgs_forward_backward_labels_share_execution_but_hvp_is_distinct():
    from ralup.analysis.cost import extra_execution_events,cost_quality_frontier
    raw=[{'event_id':'e1','execution_id':'loss','kind':'extra_loss','target_bytes':4},
         {'event_id':'e2','execution_id':'loss','kind':'extra_loss_backward','target_bytes':4},
         {'event_id':'e3','execution_id':'h1','kind':'hvp','target_bytes':4}]
    events=extra_execution_events('trial:0',raw)
    totals=cost_quality_frontier([run('adaptive',events)],'total')['totals']
    assert totals['total_targets']==8 and totals['physical_events']==2
    assert totals['loss_targets']==4 and totals['hvp_targets']==4
    assert totals['unique_targets'] is None


def test_loss_pass_logging_retains_partial_microbatch_failure():
    import hashlib
    from ralup.models.transformer import ByteTransformer,ModelConfig
    from ralup.optim.base import MethodAdapter,default_parameter_groups
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.train import TrainingState,train_step,TrainingFailure
    from ralup.data.articles import Article
    from ralup.data.cursor import BatchCursor
    from ralup.data.batches import next_logical_batch
    model=ByteTransformer(ModelConfig(1,8,2))
    state=TrainingState(model,MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule()),microbatch=1)
    articles=[Article(raw,[],hashlib.sha256(raw).hexdigest()) for raw in (b'four',b'five')]
    cursor=BatchCursor.from_articles(articles,seed=1,run_target_budget=8)
    forward=model.forward;calls=[]
    def fail_second(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('second physical forward failure')
        return forward(*args,**kwargs)
    model.forward=fail_second
    with pytest.raises(TrainingFailure):train_step(state,next_logical_batch(cursor,8))
    events=state.ledger.events[-1]['loss_pass_events']
    assert len(events)==2 and len({e['event_id'] for e in events})==2
    assert events[0]['forward_completed'] and events[0]['backward_launched']
    assert not events[1]['forward_completed'] and not events[1]['backward_launched']
    assert state.ledger.stream['targets']==0


def test_slow_crossing_curves_keep_actual_matched_stream_total_and_wall_cost():
    from ralup.analysis.cost import cost_quality_frontier
    fast=run('fast',[event('f',10,stream=10)],end=5,score=3.)
    slow=run('slow',[event('s1',10,stream=10),event('q1',2),event('s2',10,stream=10,start=10),event('q2',2)],start=5,end=20,score=2.)
    slow['checkpoints']=[{'event_ids':['s1','q1'],'development_bpb':4.,'wall_intervals':[{'clock':'UTC','start':5,'end':12}],
                         'wall_scope':{'complete':True,'boundary':'declared_synthetic_checkpoint_component'}},
                         {'event_ids':['s1','q1','s2','q2'],'development_bpb':2.,'wall_intervals':slow['wall_intervals'],
                          'wall_scope':slow['wall_scope']}]
    result=cost_quality_frontier([fast,slow],'total')
    assert [p['cost'] for p in result['frontier']]==[10,22,34]
    assert [p['development_bpb'] for p in result['frontier']]==[3.,3.,2.]
    assert [p['stream_targets'] for p in result['matched_curves']]==[10,10,20]
    assert [p['seconds'] for p in result['matched_curves']]==[5,7,15]


def test_unknown_ancestor_prevents_complete_stream_and_wall_cost_claim():
    from ralup.analysis.cost import cost_quality_frontier
    child=run('child',[event('known',4,stream=4)],start=5,end=10,score=2.)
    child['unknown_ancestor_cost']=True
    totals=cost_quality_frontier([child],'seconds')['totals']
    assert totals['seconds'] is None and totals['stream_targets'] is None
    assert totals['known_stream_targets']==4 and totals['known_interval_seconds']==5


def test_checkpoint_cannot_omit_earlier_executed_passes():
    from ralup.analysis.cost import cost_quality_frontier
    r=run('r',[event('first',4),event('last',4)],score=1.)
    r['checkpoints']=[{'event_ids':['last'],'development_bpb':1.,'wall_intervals':r['wall_intervals']}]
    with pytest.raises(ValueError,match='prefix'):cost_quality_frontier([r],'total')
