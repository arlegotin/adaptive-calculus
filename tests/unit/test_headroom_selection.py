"""S12/S26 advancement must survive short-curve and adapter-selection bias."""
import hashlib
import json
from pathlib import Path
import pytest
from ralup.records import ArtifactRef
from ralup.config import canonical_bytes


def record(tmp_path, name, **updates):
    row = dict(trial_id=name, family='AdamW', seed=17, budget=32768,
        status='completed', model='proxy', partition='target_train',
        curve=[{'budget':16384,'development_bpb':4.4}, {'budget':32768,'development_bpb':4.3}],
        development_interval=[4.2,4.4], settled=False, lr_bracketed=True,
        edge=False, missing_mechanism=False, eliminates_headroom=False,
        recipe={'method':'adamw','quantizer':'row_dynamic_max','grouping':'row',
                'rotation':'identity','backward':'ste','schedule':'constant',
                'learning_rate':.001,'normalization':'none','moment_coordinates':'original_W'},
        evidence_kind='synthetic_curve_fixture')
    row.update(updates)
    path=tmp_path/(name+'.json');path.write_bytes(canonical_bytes(row))
    return ArtifactRef(path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_unsettled_edge_overlap_and_headroom_eliminator_survive(tmp_path):
    from ralup.search import screen_headroom
    refs=[record(tmp_path,'leader',settled=True,development_interval=[3.,3.1]),
          record(tmp_path,'edge',edge=True,settled=True,development_interval=[5.,5.2]),
          record(tmp_path,'unsettled',development_interval=[6.,6.2]),
          record(tmp_path,'overlap',settled=True,development_interval=[3.05,3.2]),
          record(tmp_path,'no-gap',settled=True,eliminates_headroom=True,development_interval=[5.,5.2]),
          record(tmp_path,'dominated',settled=True,development_interval=[6.,6.2])]
    screen=screen_headroom(refs)
    assert 'edge' in screen['expand']
    assert {'edge','unsettled','overlap','no-gap'} <= set(screen['advance'])
    assert 'dominated' not in screen['advance']
    assert screen['selection_ready'] is False


def test_shortest_fixed_trial_count_winner_is_rejected(tmp_path):
    from ralup.search import select_backbone
    rows=[json.loads(record(tmp_path,'short-winner').path.read_bytes())]
    with pytest.raises(ValueError,match='unsettled|full-budget|measured'):
        select_backbone(rows,{'output':str(tmp_path/'b0.json'),'winner':'short-winner'})


def test_no_protected_score_can_enter_nested_selection(tmp_path):
    from ralup.search import screen_headroom, select_backbone
    bad=record(tmp_path,'bad',diagnostic={'test_bpb':2.})
    with pytest.raises(ValueError,match='protected'):
        screen_headroom([bad])
    with pytest.raises(ValueError,match='protected'):
        select_backbone([],{'output':str(tmp_path/'x'),'hidden':{'test_bpb':1.}})


def test_trial_hash_source_and_budget_are_binding(tmp_path,monkeypatch):
    from ralup.search import TrialSpec, run_search_trial
    monkeypatch.chdir(tmp_path)
    trial=TrialSpec('0'*64,'AdamW',17,8192,'8e5b2e1')
    with pytest.raises((ValueError,FileNotFoundError)):
        run_search_trial(trial)
    with pytest.raises(ValueError):
        TrialSpec('0'*64,'AdamW',17,0,'8e5b2e1')


def test_disposition_is_not_b0_or_reference_upper_bound(tmp_path):
    from ralup.search import select_backbone
    row=json.loads(record(tmp_path,'measured',evidence_kind='real_data',
        cost={'complete_seconds':2.},source_revision='actual',data_sha256='a'*64).path.read_bytes())
    ref=select_backbone([row],{'output':str(tmp_path/'disposition.json'),
        'disposition':'resource-limited','reason':'Measured horizon remains unsettled; all competitors retained.'})
    selection=json.loads(ref.path.read_bytes())
    assert selection['b0'] is None
    assert selection['reference_gap_is_upper_bound'] is False
    assert selection['wp04']=='inconclusive'


def test_meta_target_parent_partitions_cannot_be_crossed():
    from ralup.calibrate import validate_warmstart_partition
    validate_warmstart_partition('meta_train','meta_train')
    with pytest.raises(ValueError,match='partition'):
        validate_warmstart_partition('meta_train','target_train')


def test_fp32_engine_checkpoint_exact_resume():
    import torch
    from ralup.models.transformer import ByteTransformer,ModelConfig
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.optim.base import MethodAdapter,default_parameter_groups
    from ralup.train import TrainingState,train_step
    from ralup.data.articles import Article
    from ralup.data.cursor import BatchCursor
    from ralup.data.batches import next_logical_batch
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    torch.manual_seed(707)
    model=ByteTransformer(ModelConfig(1,8,2))
    state=TrainingState(model,MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule()))
    raw=b'ordinary fp32 reference'*3
    cursor=BatchCursor.from_articles([Article(raw,[],hashlib.sha256(raw).hexdigest())],seed=17,run_target_budget=8)
    batch=next_logical_batch(cursor,4)
    train_step(state,batch)
    snapshot=capture_snapshot(state)
    expected=exact_cpu_state_hash(state)
    restored=restore_snapshot(snapshot,'cpu')
    assert exact_cpu_state_hash(restored)==expected
    assert restored.model._recipe is None
    assert restored.histories['loss_updates']==1
    train_step(state,batch); a=exact_cpu_state_hash(state)
    train_step(restored,batch); b=exact_cpu_state_hash(restored)
    assert a==b


def test_selection_cannot_honor_arbitrary_winner_or_missing_competitor(tmp_path):
    from ralup.search import select_backbone
    rows=[]
    for family,score in [('better',3.),('worse',4.)]:
        for seed in [71,72]:
            rows.append(json.loads(record(tmp_path,f'{family}-{seed}',family=family,seed=seed,
                model='core',full_budget=True,settled=True,lr_bracketed=True,evidence_kind='real_data',
                development_interval=[score,score],recipe_key=family,
                curve=[{'budget':32768,'development_bpb':score}],
                cost={'complete_seconds':10.},source_revision='actual',data_sha256='a'*64).path.read_bytes()))
            rows[-1]['recipe']['learning_rate']=.001 if family=='better' else .002
    rationale=bind_selection(tmp_path,rows);rationale['winner']='worse-71'
    with pytest.raises(ValueError,match='winner|minimum|mean|pool'):
        select_backbone(rows,rationale)


def test_fp32_reference_does_not_block_best_eligible_quantized_backbone(tmp_path):
    from ralup.search import select_backbone
    rows=[]
    for family,quantizer,score in [('reference','fp32',2.),('quantized','row_dynamic_max',3.)]:
        for seed in [71,72]:
            row=json.loads(record(tmp_path,f'{family}-{seed}',family=family,seed=seed,
                model='core',full_budget=True,settled=True,lr_bracketed=True,evidence_kind='real_data',
                curve=[{'budget':32768,'development_bpb':score}],development_interval=[score,score],
                cost={'complete_seconds':10.},source_revision='fixture',data_sha256='a'*64).path.read_bytes())
            row['recipe']['quantizer']=quantizer
            row['recipe']['grouping']='none' if quantizer=='fp32' else 'row'
            rows.append(row)
    rationale=bind_selection(tmp_path,rows);rationale['winner']='quantized-71'
    result=select_backbone(rows,rationale)
    value=json.loads(result.path.read_bytes())
    assert value['b0']['recipe']['quantizer']=='row_dynamic_max'
    assert len(value['selection_inputs'])==4  # The stronger reference stays visible.


def test_adamw_recipe_cannot_silently_ignore_declared_overrides():
    from ralup.search import create_trial_state
    with pytest.raises(ValueError,match='AdamW.*(unsupported|resolved)'):
        create_trial_state({'recipe':{'method':'adamw','method_config':{'betas':[.95,.999]}}})


@pytest.mark.parametrize('recipe,field',[
    ({'method':'muon_project_fp32','quantizer':'row_dynamic_max','grouping':'none'},'grouping'),
    ({'method':'muon_project_fp32','quantizer':'quest_fp4_tensor_reference','grouping':'row'},'grouping'),
    ({'method':'adamw','quantizer':'fp32','grouping':'row'},'grouping'),
    ({'method':'muon_plus_row_col_fp32','quantizer':'row_dynamic_max','normalization':'original_W'},'normalization'),
    ({'method':'muon_project_fp32','quantizer':'row_dynamic_max','moment_coordinates':'rotated'},'coordinates'),
])
def test_public_construction_rejects_false_recipe_annotations_before_model_creation(recipe,field):
    from ralup.search import create_trial_state
    with pytest.raises(ValueError,match=field):
        create_trial_state({'recipe':recipe})


def test_recipe_metadata_distinguishes_grouping_normalization_and_moment_coordinates():
    from ralup.search import _validate_recipe_metadata
    for grouping in ('row','last_axis_row'):
        _validate_recipe_metadata({'method':'muon_plus_row_col_fp32','quantizer':'quest_fp4_row_reference',
            'grouping':grouping,'normalization':'row_col','moment_coordinates':'original_W'})
    _validate_recipe_metadata({'method':'adamw','quantizer':'fp32','grouping':'none','normalization':'none'})
    _validate_recipe_metadata({'method':'muon_project_fp32','quantizer':'quest_fp4_tensor_reference',
        'grouping':'whole_tensor','normalization':'none','moment_coordinates':'original_W'})


def test_final_selection_cannot_publish_a_false_grouping_label(tmp_path):
    from ralup.search import select_backbone
    rows=[]
    for seed in (71,72):
        row=json.loads(record(tmp_path,f'candidate-{seed}',seed=seed,model='core',full_budget=True,
            settled=True,lr_bracketed=True,evidence_kind='real_data',cost={'complete_seconds':1.}).path.read_bytes())
        row['recipe']['grouping']='none';rows.append(row)
    with pytest.raises(ValueError,match='grouping'):
        select_backbone(rows,{'output':str(tmp_path/'bad.json'),'winner':'candidate-71',
            'equal_initial_compute':True,'parity_evidence':['fixture'],'fresh_seeds':[71,72]})


@pytest.mark.parametrize('failure_at',['training','query','query_cap'])
def test_failed_public_trial_does_not_invent_or_reuse_exposures(tmp_path,monkeypatch,failure_at):
    import ralup.search as search
    import ralup.calibrate as calibration
    import ralup.train as training
    import ralup.evaluate as evaluation
    from ralup.models.transformer import ByteTransformer,ModelConfig
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.optim.base import MethodAdapter,default_parameter_groups
    from ralup.data.articles import Article
    from ralup.data.cursor import BatchCursor
    monkeypatch.chdir(tmp_path)
    raw=b'failure accounting fixture '*128
    def cursor():
        return BatchCursor.from_articles([Article(raw,[],hashlib.sha256(raw).hexdigest())],seed=17,run_target_budget=8192)
    model=ByteTransformer(ModelConfig(1,8,2))
    state=training.TrainingState(model,MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule()),cursor=cursor())
    monkeypatch.setattr(search,'create_trial_state',lambda config:state)
    monkeypatch.setattr(BatchCursor,'from_manifest',classmethod(lambda cls,*args,**kwargs:cursor()))
    monkeypatch.setattr(calibration,'_limits',lambda payload:{'resident_cap_bytes':32*2**30,'driver_cap_bytes':32*2**30})
    if failure_at=='query_cap':
        from types import SimpleNamespace
        clock=[0.]; actual_score=evaluation.score_blocks
        monkeypatch.setattr(search,'time',SimpleNamespace(perf_counter=lambda:clock[0]))
        def slow_score(*args,**kwargs):
            callback=kwargs['progress_callback']
            def advance(row):
                if row['phase']=='block_completed':clock[0]=31.
                callback(row)
            return actual_score(*args,**{**kwargs,'progress_callback':advance})
        monkeypatch.setattr(evaluation,'score_blocks',slow_score)
    else:
        def fail(*args,**kwargs):raise RuntimeError('injected failure before completed exposure record')
        monkeypatch.setattr(training if failure_at=='training' else evaluation,'train_step' if failure_at=='training' else 'score_blocks',fail)
    gate=tmp_path/'admission.json'
    gate.write_bytes(canonical_bytes({'status':'pass','source_identity':search.source_identity(),
        'scopes':[['core','adamw','fp32','identity','ste',1]],'maximum_steps':1}))
    config={'family':'AdamW','seed':17,'budget':8192,'source_revision':'fixture',
        'source_identity':search.source_identity(),'purpose':'development-bootstrap','finite_forecast_seconds':30.,
        'resolved':{'synthetic':False,'profile':'core','runtime':{'device':'cpu'},'data':{'manifest':{'path':'fixture-injected','sha256':'f'*64}}},
        'admission':{'path':str(gate),'sha256':hashlib.sha256(gate.read_bytes()).hexdigest()},
        'recipe':{'method':'adamw','quantizer':'fp32','rotation':'identity','backward':'ste','schedule':'constant','learning_rate':.001},
        'partition':'target_train','role':'continuation','packing':1,'output':str(tmp_path/'run'),
        'query_seed':17,'query_bytes':8,'milestones':[1]}
    config['resolved'].update(model={'layers':1,'width':8,'heads':2},optimizer_probe={'betas':[.9,.99],'eps':1e-8})
    config['resolved']['data'].update(context_length=256,logical_batch_bytes=8192,target_block_bytes=128,vocabulary_size=257,boundary_token=256)
    config['admission']=write_evidence(tmp_path,'bound-admission',admission_fixture(tmp_path,config))
    trial=search.register_trial(config)
    result=json.loads(search.run_search_trial(trial).path.read_bytes())
    assert result['status']=='failed'
    assert ('envelope' if failure_at=='query_cap' else 'injected failure') in result['failure']
    if failure_at=='training':
        rows=[json.loads(line) for line in (tmp_path/'run/steps.jsonl').read_text().splitlines()]
        assert rows[0]['events']==[]  # The successful initial query is not a training attempt.
        assert result['unknown_partial_training_work'] is True
    elif failure_at=='query':
        assert result['cost']['ledger']['passes']['query']==0  # Known completed lower bound only.
        assert result['unknown_partial_query_work'] is True
    else:
        assert result['cost']['ledger']['passes']['query']==8
        assert result['query_attempts'][0]['known_completed_targets']==8
        assert result['query_attempts'][0]['unknown_partial_work'] is False
        assert result['curve']==[]  # Capped partial pass cannot become a selection score.


def test_nan_interval_rejected(tmp_path):
    from ralup.search import screen_headroom
    ref=record(tmp_path,'nan',development_interval=[0.,1.])
    raw=json.loads(ref.path.read_bytes());raw['development_interval']=[float('nan'),1.]
    ref.path.write_text(json.dumps(raw));ref=ArtifactRef(ref.path,hashlib.sha256(ref.path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='Finite'):
        screen_headroom([ref])


def test_mismatched_proxy_core_budgets_never_eliminate_each_other(tmp_path):
    from ralup.search import screen_headroom
    refs=[record(tmp_path,'proxy',settled=True,development_interval=[3.,3.1]),
          record(tmp_path,'core',model='core',settled=True,development_interval=[5.,5.1]),
          record(tmp_path,'short',budget=8192,settled=True,development_interval=[6.,6.1])]
    assert set(screen_headroom(refs)['advance'])=={'proxy','core','short'}


def test_headroom_template_requires_explicit_hash_bound_sidecar(tmp_path,monkeypatch):
    from ralup.search import run_headroom
    p=tmp_path/'proxy.json';p.write_text('{"profile":"proxy","synthetic":false}')
    with pytest.raises(ValueError,match='explicit|frozen'):
        run_headroom(p)
    sidecar={'suite':'headroom','purpose':'development-bootstrap',
        'template_sha256':'0'*64,'trials':[]}
    p.with_suffix('.headroom.json').write_text(json.dumps(sidecar))
    with pytest.raises(ValueError,match='hash'):
        run_headroom(p)


def test_search_warmstart_cost_report_is_directly_consumable(tmp_path):
    from ralup.search import warmstart_report
    from ralup.calibrate import import_warmstart
    from ralup.state import capture_snapshot,save_snapshot
    from ralup.models.transformer import ByteTransformer,ModelConfig
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.optim.base import MethodAdapter,default_parameter_groups
    from ralup.train import TrainingState
    from ralup.data.articles import Article
    from ralup.data.cursor import BatchCursor
    def state():
        model=ByteTransformer(ModelConfig(1,8,2));raw=b'partition cost contract'
        value=TrainingState(model,MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule()),
            cursor=BatchCursor.from_articles([Article(raw,[],hashlib.sha256(raw).hexdigest())],seed=17,run_target_budget=8),
            configuration={'source_partition':'target_train'})
        return value
    parent=state();parent.configuration['artifact_role']='warmstart'
    checkpoint=save_snapshot(capture_snapshot(parent),tmp_path/'state')
    ledger={key:getattr(parent.ledger,key) for key in ('stream','passes','elapsed','events')}
    report=warmstart_report(checkpoint,'target_train',{'ledger':ledger,'complete_seconds':7.,'parent':None})
    path=tmp_path/'warmstart.json';path.write_bytes(canonical_bytes(report))
    imported=import_warmstart(ArtifactRef(path,hashlib.sha256(path.read_bytes()).hexdigest()),state(),partition='target_train',continuation_targets=8)
    assert imported.configuration['warmstart_parent']['parent_cost']['project_seconds']==7.
    assert imported.ledger.passes==parent.ledger.passes


def test_fp32_snapshot_rejects_fake_initial_history_and_nonmapping_cache():
    from ralup.models.transformer import ByteTransformer,ModelConfig
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.optim.base import MethodAdapter,default_parameter_groups
    from ralup.train import TrainingState
    from ralup.state import capture_snapshot,restore_snapshot,IntegrityError
    model=ByteTransformer(ModelConfig(1,8,2))
    state=TrainingState(model,MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule()))
    state.histories={'loss_ema':1.,'loss_updates':1}
    with pytest.raises(IntegrityError,match='initial|FP32'):
        restore_snapshot(capture_snapshot(state),'cpu')
    state.histories={};model._contexts=[]
    with pytest.raises(IntegrityError,match='context|FP32'):
        restore_snapshot(capture_snapshot(state),'cpu')


def test_real_search_refuses_synthetic_payload_before_execution(tmp_path,monkeypatch):
    from ralup.search import register_trial,run_search_trial,source_identity
    monkeypatch.chdir(tmp_path)
    trial=register_trial({'family':'AdamW','seed':17,'budget':8192,'source_revision':'fixture',
        'source_identity':source_identity(),'purpose':'development-bootstrap','finite_forecast_seconds':10.,
        'resolved':{'synthetic':True}})
    with pytest.raises(ValueError,match='synthetic|real-data'):
        run_search_trial(trial)


def write_evidence(tmp_path, name, value):
    path=tmp_path/(name+'.json');path.write_bytes(canonical_bytes(value))
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def operation_config():
    return {'source_identity':{'fixture.py':'a'*64},'source_revision':'fixture','family':'SOAP',
        'seed':71,'budget':32768,'packing':32,'partition':'target_train','role':'continuation',
        'purpose':'development-bootstrap','finite_forecast_seconds':10.,
        'resolved':{'profile':'core','synthetic':False,'model':{'layers':4,'width':256,'heads':4},
            'runtime':{'device':'cpu','dtype':'float32'},'optimizer_probe':{'betas':[.9,.99],'eps':1e-8},
            'data':{'context_length':256,'logical_batch_bytes':8192,'target_block_bytes':128,'vocabulary_size':257,'boundary_token':256,
                'manifest':{'path':'fixture-corpus','sha256':'a'*64}}},
        'recipe':{'method':'soap','method_config':{'precondition_frequency':10},'quantizer':'row_dynamic_max',
            'grouping':'row','rotation':'identity','backward':'ste','schedule':'constant','learning_rate':.001,
            'normalization':'none','moment_coordinates':'original_W','clip_norm':1.,'decay':0.}}


def operation_contract_fixture(config):
    return {'model':config['resolved']['model'],'runtime':config['resolved']['runtime'],
        'optimizer_probe':config['resolved']['optimizer_probe'],'packing':config['packing'],
        'data_geometry':{k:config['resolved']['data'][k] for k in ('context_length','logical_batch_bytes','target_block_bytes','vocabulary_size','boundary_token')},
        'recipe':config['recipe']}


def admission_fixture(tmp_path, config):
    proof=write_evidence(tmp_path,'operation-proof-'+hashlib.sha256(canonical_bytes(operation_contract_fixture(config))).hexdigest(),{'status':'pass','scope':operation_contract_fixture(config),'source_identity':config['source_identity'],'maximum_steps':4})
    return {'status':'pass','source_identity':config['source_identity'],
        'scopes':[['core',config['recipe']['method'],config['recipe']['quantizer'],'identity','ste',config['packing']]],'maximum_steps':4,
        'operation_scopes':[{'contract':operation_contract_fixture(config),'maximum_steps':4,'evidence':proof}]}


@pytest.mark.parametrize('change',['model','refresh','clipping','schedule'])
def test_admission_rejects_unqualified_actual_operation_before_construction(tmp_path,monkeypatch,change):
    import copy
    import ralup.search as search
    monkeypatch.chdir(tmp_path);config=operation_config();config['source_identity']=search.source_identity()
    gate=admission_fixture(tmp_path,copy.deepcopy(config))
    if change=='model':config['resolved']['model']['width']=512
    elif change=='refresh':config['recipe']['method_config']['precondition_frequency']=1
    elif change=='clipping':config['recipe']['clip_norm']=2.
    else:config['recipe']['schedule']='cosine'
    config.update(output=str(tmp_path/'run'),admission=write_evidence(tmp_path,'admission',gate))
    def forbidden(config):raise AssertionError('unadmitted operation reached construction')
    monkeypatch.setattr(search,'create_trial_state',forbidden)
    with pytest.raises(ValueError,match='admission|operation'):
        search.run_search_trial(search.register_trial(config))
    assert not (tmp_path/'run').exists()


def bind_selection(tmp_path, rows):
    """Explicit tiny file-backed fixtures for the final analysis contract."""
    import copy
    protocol=write_evidence(tmp_path,'parent-protocol',{'schema':'ralup.parent_protocol.v1','partition':'target_train','description':'fixture same-seed parent protocol'})
    refs=[]
    for row in rows:
        config=operation_config();config.update(seed=row['seed'],budget=row['budget'],family=row['family'],recipe=copy.deepcopy(row['recipe']))
        config['parent']=write_evidence(tmp_path,f'parent-{row["seed"]}',{'partition':'target_train','seed':row['seed']})
        config['source_revision']=row.get('source_revision','fixture')
        row.update(source_revision=config['source_revision'],source_identity=config['source_identity'],data_sha256='a'*64,
            trial_config=write_evidence(tmp_path,row['trial_id']+'-config',config),parent_protocol=protocol,
            query_inputs=write_evidence(tmp_path,f'query-{row["seed"]}',{'partition':'development','seed':1707,'offsets':[{'document_id':'doc','offsets':[1,2]}]}))
        gate=admission_fixture(tmp_path,config)
        row['source_applicability']=write_evidence(tmp_path,row['trial_id']+'-admission',gate)
        # May intentionally encode failed/nonfinite rows for negative cases.
        path=tmp_path/(row['trial_id']+'-result.json');path.write_text(json.dumps(row,sort_keys=True))
        row['evidence']={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};refs.append(row['evidence'])
    parity=write_evidence(tmp_path,'parity',{'status':'pass','methods':sorted({r['recipe']['method'] for r in rows}),
        'source_identities':[rows[0]['source_identity']]})
    compute=write_evidence(tmp_path,'equal-search',{'status':'completed','result_refs':refs,
        'equal_initial_compute':True,'equal_development_information':True,
        'family_initial_compute':{r['family']:100. for r in rows}})
    analysis=write_evidence(tmp_path,'selection-analysis',{'schema':'ralup.backbone_selection_analysis.v1','status':'completed',
        'result_refs':refs,'parity_evidence':[parity],'equal_search_evidence':compute})
    return {'output':str(tmp_path/'selected.json'),'winner':rows[0]['trial_id'],'equal_initial_compute':True,
        'parity_evidence':[parity],'fresh_seeds':sorted({r['seed'] for r in rows}),'selection_analysis':analysis}


def final_rows(tmp_path, scores=(3.,3.)):
    return [json.loads(record(tmp_path,f'A-{seed}',seed=seed,model='core',full_budget=True,settled=True,
        lr_bracketed=True,evidence_kind='real_data',curve=[{'budget':32768,'development_bpb':score}],
        cost={'complete_seconds':10.},source_revision='fixture',data_sha256='a'*64).path.read_bytes())
        for seed,score in zip((71,72),scores)]


@pytest.mark.parametrize('bad',['failed','missing_source','missing_data','nan_score','negative_cost','short_endpoint','tampered_row','query_mismatch','parent_mismatch','unbound_analysis'])
def test_final_selection_rejects_unbound_or_incomparable_completed_evidence(tmp_path,bad):
    from ralup.search import select_backbone
    rows=final_rows(tmp_path)
    if bad=='failed':rows[0]['status']='failed'
    elif bad=='nan_score':rows[0]['curve'][-1]['development_bpb']=float('nan')
    elif bad=='negative_cost':rows[0]['cost']['complete_seconds']=-1.
    elif bad=='short_endpoint':rows[0]['curve'][-1]['budget']=8192
    rationale=bind_selection(tmp_path,rows)
    if bad=='missing_source':rows[0].pop('source_revision')
    elif bad=='missing_data':rows[0].pop('data_sha256')
    elif bad=='tampered_row':rows[0]['curve'][-1]['development_bpb']=1.
    elif bad in ('query_mismatch','parent_mismatch'):
        # Add a second competing recipe with a different matched-seed input/protocol.
        other=json.loads(json.dumps(rows[0]));other.pop('evidence');other['trial_id']='B-71';other['recipe']['learning_rate']=.002
        other2=json.loads(json.dumps(rows[1]));other2.pop('evidence');other2['trial_id']='B-72';other2['recipe']['learning_rate']=.002
        rows.extend([other,other2]);rationale=bind_selection(tmp_path,rows)
        key='query_inputs' if bad=='query_mismatch' else 'parent_protocol'
        original=json.loads(Path(rows[2][key]['path']).read_bytes())
        original['seed' if bad=='query_mismatch' else 'description']=999 if bad=='query_mismatch' else 'other protocol'
        rows[2][key]=write_evidence(tmp_path,'mismatched-'+key,original)
        saved={k:v for k,v in rows[2].items() if k!='evidence'}
        rows[2]['evidence']=write_evidence(tmp_path,'mismatched-row',saved)
        # Rebind analysis to actual modified rows, so the comparison must reject.
        analysis=json.loads(Path(rationale['selection_analysis']['path']).read_bytes());analysis['result_refs']=[r['evidence'] for r in rows]
        compute=json.loads(Path(analysis['equal_search_evidence']['path']).read_bytes());compute['result_refs']=analysis['result_refs']
        analysis['equal_search_evidence']=write_evidence(tmp_path,'rebound-compute',compute)
        rationale['selection_analysis']=write_evidence(tmp_path,'rebound-analysis',analysis)
    elif bad=='unbound_analysis':rationale.pop('selection_analysis')
    with pytest.raises(ValueError,match='evidence|completed|finite|endpoint|matched|analysis|source|cost|protocol'):
        select_backbone(rows,rationale)


def test_duplicate_seed_winner_inversion_and_duplicate_trial_ids_rejected(tmp_path):
    import copy
    from ralup.search import select_backbone
    rows=final_rows(tmp_path,(1.,5.))
    for row in copy.deepcopy(rows):
        row['trial_id']=row['trial_id'].replace('A','B');row['family']='other';row['recipe']['learning_rate']=.002
        row['curve'][-1]['development_bpb']=2.5;rows.append(row)
    rationale=bind_selection(tmp_path,rows);rationale['winner']='B-71'
    assert json.loads(select_backbone(rows,rationale).path.read_bytes())['b0']['evidence']['trial_id']=='B-71'
    duplicate=copy.deepcopy(rows[0]);duplicate.pop('evidence');duplicate['trial_id']='A-71-duplicate'
    rows.append(duplicate);rationale=bind_selection(tmp_path,rows);rationale['winner']='A-71';rationale['output']=str(tmp_path/'duplicate.json')
    with pytest.raises(ValueError,match='Duplicate.*seed'):
        select_backbone(rows,rationale)
    rows.pop();rows[2]['trial_id']=rows[0]['trial_id'];rationale=bind_selection(tmp_path,rows)
    with pytest.raises(ValueError,match='Duplicate.*trial'):
        select_backbone(rows,rationale)


def test_unknown_schedule_rejected_before_model_or_query():
    from ralup.search import create_trial_state
    with pytest.raises(ValueError,match='schedule'):
        create_trial_state({'recipe':{'method':'adamw','quantizer':'fp32','schedule':'cosin'}})


def test_operation_scope_cannot_borrow_unrelated_pass_proof(tmp_path):
    from ralup.search import _validate_admission
    config=operation_config();gate=admission_fixture(tmp_path,config)
    gate['operation_scopes'][0]['evidence']=write_evidence(tmp_path,'unrelated-proof',{'status':'pass','scope':{'unrelated':True}})
    with pytest.raises(ValueError,match='admission|operation'):
        _validate_admission(config,gate)


def test_scalar_lr_stress_needs_bound_delta_and_cannot_change_refresh(tmp_path):
    import copy
    from ralup.search import _validate_admission
    config=operation_config();gate=admission_fixture(tmp_path,copy.deepcopy(config))
    _validate_admission(config,gate)
    config['recipe']['learning_rate']=.01
    with pytest.raises(ValueError,match='admission|operation'):_validate_admission(config,gate)
    delta={'schema':'ralup.scalar_lr_applicability.v1','status':'pass','source_identity':config['source_identity'],
        'admitted_contract':gate['operation_scopes'][0]['contract'],'requested_contract':operation_contract_fixture(config),
        'numerical_evidence':gate['operation_scopes'][0]['evidence'],'risk_rationale':'Explicit finite scalar-rate stress; identical model, refresh, primitives and horizon; instability retained.'}
    config['operation_applicability']=write_evidence(tmp_path,'lr-delta',delta)
    _validate_admission(config,gate)
    config['recipe']['method_config']['precondition_frequency']=1
    delta['requested_contract']=operation_contract_fixture(config)
    config['operation_applicability']=write_evidence(tmp_path,'invalid-refresh-delta',delta)
    with pytest.raises(ValueError,match='admission|operation'):_validate_admission(config,gate)
