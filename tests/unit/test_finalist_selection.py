"""S26 allocation/selection fixtures are engineering evidence only."""
import copy
import json
from pathlib import Path
import pytest


def family(name, forecasts):
    return {'family':name,'status':'eligible','generic_improvements':['ordered_embedding'],
            'development_information':'same-panel', 'trials':[
                {'family':name,'seed':n,'budget':8192,'source_revision':'fixture',
                 'finite_forecast_seconds':seconds,'purpose':'development-bootstrap'}
                for n,seconds in enumerate(forecasts)]}


def test_equal_initial_seconds_allow_different_trial_counts(tmp_path,monkeypatch):
    import ralup.search as search
    monkeypatch.chdir(tmp_path)
    specs=search.allocate_search([family('fast',[5,5]),family('slow',[10])],20,[])
    assert [s.family for s in specs]==['fast','slow','fast']
    configs=[json.loads((Path('artifacts/search/configs')/(s.config_hash+'.json')).read_bytes()) for s in specs]
    assert {c['allocation']['initial_allowance_seconds'] for c in configs}=={10}
    assert all(c['allocation']['fixed_trial_count_fairness'] is False for c in configs)


def test_boundary_slow_and_blocked_contenders_are_retained(tmp_path,monkeypatch):
    import ralup.search as search
    monkeypatch.chdir(tmp_path)
    f=family('slow',[10]);f['stage']='extension'
    row={'family':'slow','status':'completed','trial_id':'slow-first','model':'proxy','partition':'target_train',
         'budget':8192,'development_interval':[2.9,3.3],'settled':False,'edge':True,'lr_bracketed':False,
         'curve':[{'budget':4096,'development_bpb':5.},{'budget':8192,'development_bpb':3.1}]}
    from ralup.search import _write,_ref
    f['extension_evidence']=_ref(_write(tmp_path/'curve.json',row))
    blocked=family('SOAP',[]);blocked.update(status='blocked',reason='actual operation scope unverified')
    specs=search.allocate_search([f,blocked],20,[row])
    config=json.loads((Path('artifacts/search/configs')/(specs[0].config_hash+'.json')).read_bytes())
    assert 'unsettled curve' in config['allocation']['extension_reasons']
    assert 'expand LR bracket' in config['allocation']['extension_reasons']
    assert config['allocation']['blocked_families']=={'SOAP':'actual operation scope unverified'}


def test_candidate_only_generic_improvement_and_unequal_information_reject(tmp_path,monkeypatch):
    from ralup.search import allocate_search
    monkeypatch.chdir(tmp_path)
    families=[family('candidate',[5]),family('learnedLR',[5])]
    families[0]['generic_improvements'].append('new-kernel')
    with pytest.raises(ValueError,match='generic'):allocate_search(families,10,[])
    families[0]['generic_improvements'].pop();families[0]['development_information']='extra-panel'
    with pytest.raises(ValueError,match='information'):allocate_search(families,10,[])


def test_full_budget_mean_bstar_can_differ_from_b0_and_ties_use_total_cost():
    from ralup.search import _rank_development
    rows=[]
    for method,scores,cost in [('B0',[3.,3.],1),('slow',[4.,1.],20),('tie',[2.,3.],10)]:
        for seed,score in zip((71,72),scores):
            rows.append({'trial_id':f'{method}-{seed}','family':method,'seed':seed,'budget':100,
                         'status':'completed','full_budget':True,'fresh_seed':True,'settled':True,
                         'lr_bracketed':True,'edge':False,'missing_mechanism':False,'model':'core',
                         'recipe':{'method':method},'curve':[{'budget':100,'development_bpb':score}],
                         'cost':{'complete_seconds':cost}})
    result=_rank_development(rows,[71,72])
    assert result['winner']['recipe']['method']=='tie'
    assert result['winner']['mean_bpb']==2.5
    assert result['winner']['end_to_end_seconds']==20
    rows[-1]['status']='failed'
    with pytest.raises(ValueError,match='failed|completed'): _rank_development(rows,[71,72])


def test_calibration_vectors_and_missing_measurements_cannot_be_finalists(tmp_path):
    from ralup.search import select_finalists
    with pytest.raises(ValueError,match='authority|prerequisite|B0'):
        select_finalists([],{'output':str(tmp_path/'final.json')})
    assert not (tmp_path/'final.json').exists()


def test_protected_scores_rejected_before_any_disposition_write(tmp_path):
    from ralup.search import select_finalists
    with pytest.raises(ValueError,match='protected'):
        select_finalists([{'test_bpb':1.}],{'output':str(tmp_path/'bad.json'),'disposition':'inconclusive'})
    assert not (tmp_path/'bad.json').exists()


@pytest.mark.local_evidence
def test_cli_requires_explicit_action_and_returns_gated_disposition(tmp_path,capsys):
    from ralup.__main__ import main
    root=Path(__file__).resolve().parents[2]
    b0=json.loads((root/'docs/harness/m07-closing-evidence.json').read_bytes())['disposition']
    meta=json.loads((root/'docs/harness/m11-evidence.json').read_bytes())
    controller=next(a['files']['disposition.json'] for a in meta['attempts'] if a['attempt']=='closure-v1')
    envelope={'action':'disposition','results':[], 'eligibility':{'disposition':'inconclusive',
        'b0_evidence':b0,'controller_evidence':controller,'output':str(tmp_path/'disposition.json')}}
    config=tmp_path/'config.json';config.write_text(json.dumps(envelope))
    assert main(['select-finalists','--config',str(config)])==0
    result=json.loads((tmp_path/'disposition.json').read_bytes())
    assert result['WP06']=='NEEDS_REVISION'
    assert result['B_star'] is None and result['candidate'] is None
    assert result['defeated_methods']==[]
    assert not result['uses_protected_scores'] and not result['completed_tournament']
    assert not result['measured_non_advantage'] and not result['global_unaffordability']
    assert result['confirmation_eligible'] is False
    config.write_text(json.dumps({'eligibility':envelope['eligibility']}))
    with pytest.raises(SystemExit):main(['select-finalists','--config',str(config)])


def test_disposition_rejects_unindexed_gate_labels(tmp_path):
    from ralup.search import select_finalists
    with pytest.raises(ValueError,match='indexed'):
        select_finalists([],{'disposition':'inconclusive','b0_evidence':{'status':'inconclusive'},
            'controller_evidence':{'status':'NEEDS_REVISION'},'output':str(tmp_path/'bad.json')})


def test_runner_charges_failed_trials_and_stops_at_family_allowance(tmp_path,monkeypatch):
    import ralup.search as search
    monkeypatch.chdir(tmp_path)
    clock=[0.];launched=[]
    # Only model work is replaced; the real allocator freezes configs and the
    # runner must persist the actual result reference and skip the next trial.
    def execute(spec):
        launched.append(spec);clock[0]+=7
        return search._write(tmp_path/f'{spec.seed}-result.json',{'trial_id':spec.config_hash,
            'family':spec.family,'status':'failed','cost':{'complete_seconds':7.},'curve':[],
            'failure':'finite-budget divergence'})
    from types import SimpleNamespace
    monkeypatch.setattr(search,'time',SimpleNamespace(perf_counter=lambda:clock[0]))
    monkeypatch.setattr(search,'run_search_trial',execute)
    config={'action':'search','families':[family('ordinary',[5,5])], 'allowance_seconds':10,
            'evidence':[],'output':str(tmp_path/'search.json')}
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    ref=search.run_finalist_tournament(path)
    result=json.loads(ref.path.read_bytes())
    assert len(launched)==1
    assert result['family_elapsed_seconds']=={'ordinary':7}
    assert len(result['results'])==1 and len(result['unrun_trials'])==1
    assert result['unrun_trials'][0]['reason']=='remaining family allowance below frozen forecast'
    assert result['defeated_methods']==[]


def test_extension_cannot_invent_overlap_by_looking_only_at_itself(tmp_path,monkeypatch):
    import ralup.search as search
    monkeypatch.chdir(tmp_path)
    base={'status':'completed','model':'proxy','partition':'target_train','budget':8192,
          'settled':True,'lr_bracketed':True,'edge':False,'curve':[]}
    leader={**base,'family':'leader','trial_id':'leader','development_interval':[1.,1.1]}
    loser={**base,'family':'loser','trial_id':'loser','development_interval':[4.,4.1]}
    f=family('loser',[5]);f.update(stage='extension',extension_evidence=search._ref(search._write(tmp_path/'loser.json',loser)))
    with pytest.raises(ValueError,match='unresolved'):
        search.allocate_search([f],10,[leader,loser])


def test_authenticated_bootstrap_rows_cannot_be_promoted_by_finalist_flags(tmp_path):
    from test_headroom_selection import final_rows,bind_selection
    from ralup.search import select_finalists
    rows=final_rows(tmp_path)
    for row in rows:row['fresh_seed']=True
    eligibility=bind_selection(tmp_path,rows)
    eligibility['controller_stages']={'meta-development':[],'development':[]}
    with pytest.raises(ValueError,match='prospective.*replication|replication.*authority'):
        select_finalists(rows,eligibility)


def test_exported_finalist_configs_are_immutable_and_content_addressed(tmp_path):
    from ralup.search import _export_finalist_configs
    value={'B_star':{'recipe':{'method':'ordinary'},'mean_bpb':2.},
           'candidate':{'family':'affine45'},'learned_lr':{'family':'learned_lr49'},'static':{'family':'static3'}}
    refs=_export_finalist_configs(value,tmp_path/'selection.json')
    assert set(refs)=={'B_star','candidate','learned_lr','static'}
    for key,ref in refs.items():
        assert Path(ref['path']).stem==ref['sha256']
        assert json.loads(Path(ref['path']).read_bytes())==value[key]
    assert _export_finalist_configs(value,tmp_path/'selection.json')==refs
    Path(refs['candidate']['path']).write_text('{}')
    with pytest.raises(ValueError,match='immutable'):_export_finalist_configs(value,tmp_path/'selection.json')


@pytest.mark.parametrize('name',['ewgs_fp4','trs_backbone_adapter','learned_fp4_scale','damping_fp4',
    'simple_matrix_mask','magma_matrix_mask','qar_fp4_adjacent','stochastic_shadow_forward',
    'stochastic_endpoint','metaquant_fp4_online_target_training'])
def test_search_routes_named_adaptations_through_actual_shared_train_step(name):
    import hashlib
    from ralup.search import _build_search_method
    from ralup.models.transformer import ByteTransformer,ModelConfig
    from ralup.optim.base import default_parameter_groups
    from ralup.train import TrainingState,train_step
    from ralup.data.articles import Article
    from ralup.data.cursor import BatchCursor
    from ralup.data.batches import next_logical_batch
    module={'ewgs_fp4':'ewgs','trs_backbone_adapter':'trs','learned_fp4_scale':'learned_scale',
        'damping_fp4':'noise','simple_matrix_mask':'noise','magma_matrix_mask':'noise',
        'qar_fp4_adjacent':'qar','stochastic_shadow_forward':'qar','stochastic_endpoint':'qar',
        'metaquant_fp4_online_target_training':'metaquant'}[name]
    import importlib
    config=copy.deepcopy(importlib.import_module('ralup.baselines.'+module).FIXTURE_CONFIG)
    if module=='noise':config['mode']={'damping_fp4':'damping','simple_matrix_mask':'simple','magma_matrix_mask':'magma'}[name]
    if name.startswith('stochastic_'):config['mode']='shadow_forward' if name.endswith('forward') else 'endpoint'
    if module=='metaquant':config['variant']='online_target_training'
    model=ByteTransformer(ModelConfig(1,8,2));groups=default_parameter_groups(model)
    recipe={'method':name,'method_config':config,'backbone_config':{'betas':[.9,.99],'eps':1e-8}}
    method=_build_search_method(recipe,dict(model.named_parameters()),groups)
    quantizer=method.recipe if module=='learned_scale' else config.get('scale_recipe','row_fixed_max')
    model.configure_quantization({'name':quantizer,'rotation':'identity','backward':'ewgs' if module=='ewgs' else 'ste'})
    state=TrainingState(model,method,microbatch=1)
    import torch
    state.rng_streams['quantizer']=torch.Generator(device='cpu').manual_seed(101)
    raw=b'eight123';cursor=BatchCursor.from_articles([Article(raw,[],hashlib.sha256(raw).hexdigest())],seed=1,run_target_budget=8)
    batch=next_logical_batch(cursor,8)
    train_step(state,batch);train_step(state,batch)
    assert state.schedule['step']==2 and state.ledger.stream['targets']==16
    assert len(state.ledger.events[-1]['loss_pass_events'])==1


def test_initialized_metaquant_cannot_enter_frozen_search_pool():
    from ralup.search import _build_search_method
    with pytest.raises(ValueError,match='META-trained'):
        _build_search_method({'method':'metaquant_fp4_frozen','method_config':{}},{},())


def replication_fixture(tmp_path, monkeypatch, damage=None, *, derived_parent=False):
    """Real typed checkpoints and byte-bound receipts; finite admission, not science.

    Only DATA_INDEX is redirected to a local finite corpus. Snapshot decoding,
    cursor/RNG validation, query-byte checks and cost/producer authentication are
    real. No native gate, learned controller or scientific finalist is produced.
    """
    import hashlib
    import random
    import torch
    from dataclasses import asdict
    from datetime import datetime, timedelta
    from test_batch_cursor import frozen_fixture
    from test_meta_selection import Evidence
    from ralup.config import canonical_bytes
    from ralup.records import ArtifactRef
    from ralup.search import source_identity, _ref, _write, warmstart_report
    from ralup.state import capture_snapshot, save_snapshot
    from ralup.models.transformer import ByteTransformer, ModelConfig
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.optim.base import MethodAdapter, default_parameter_groups
    from ralup.train import TrainingState, select_logical_batch, train_step
    from ralup.data.cursor import BatchCursor
    from ralup.data.blocks import make_block
    from ralup.meta import selection
    e=Evidence(tmp_path);source=source_identity();manifest=frozen_fixture(tmp_path)
    corpus=copy.deepcopy(manifest.payload)
    raw=b'finite development bytes';query_file=tmp_path/'development.bin';query_file.write_bytes(raw)
    doc={'id':'development','split':'validation','content_path':str(query_file),'content_start':0,
         'bytes':len(raw),'content_sha256':hashlib.sha256(raw).hexdigest()}
    corpus['audit']['protected_documents'].append(doc)
    data=e.write(corpus)
    cursor=BatchCursor.from_manifest(Path(data['path']),data['sha256'],partition='target-training',seed=10,run_target_budget=4)
    prefix=cursor.state['source']['hashes']['source_sha256']
    index=tmp_path/'finite-data-index.json';index.write_bytes(canonical_bytes({'manifest':data,'frozen_hashes':{'source_sha256':prefix}}))
    monkeypatch.setattr(selection,'DATA_INDEX',index)
    block=asdict(make_block(doc['content_sha256'],0,len(raw),b'',raw,document_id=prefix+':development'))
    query=e.write({'blocks':[{k:list(v) if isinstance(v,tuple) else v for k,v in block.items()}]})
    # Synthetic fixture contract times; these are not measured research costs.
    warm_start='2026-10-11T01:00:00+00:00'
    replication_start='2026-10-11T03:00:00+00:00'
    def receipt(inputs,outputs,start=warm_start,*,subject=None):
        started=datetime.fromisoformat(start);ended=started+timedelta(seconds=10)
        startup=e.write({'utc':start,'launcher_started_utc':start,'inputs':inputs,
                         'source':{k:{'sha256':v} for k,v in source.items()}})
        finish=e.write({'exit_code':0,'elapsed_seconds':10.,'completed_utc':ended.isoformat(),
                        'completed_objects':[{'path':r['path'],'sha256':r['sha256']} for r in outputs]})
        producer={'startup':startup,'finish':finish}
        if subject is not None:
            producer['enclosing_measurement']=e.write({
                'schema':'ralup.complete_component_interval.v1','subject':subject,
                'producer':{'startup':startup,'finish':finish},
                'boundary':'launcher_start_through_all_outputs_and_finish_publication',
                'wall_interval':{'clock':'UTC','start':started.timestamp(),'end':ended.timestamp()},
                'excluded_costs':['observer_and_receipt_administration']})
        return producer
    lineages={};parents={}
    for seed in (10,71,72):
        initialization=torch.Generator().manual_seed(seed)
        with torch.random.fork_rng():
            torch.set_rng_state(initialization.get_state());model=ByteTransformer(ModelConfig(1,8,2))
            initialization.set_state(torch.get_rng_state())
        stream_seed=lambda name:int.from_bytes(hashlib.sha256(f'ralup-rng-v1:{seed}:{name}'.encode()).digest()[:8],'little')
        config={'source_partition':'target_train','seed':seed,'pilot_target_budget':4,
                'artifact_role':'warmstart','resolved':{'data':{'manifest':data},'model':{'layers':1,'width':8,'heads':2}}}
        cursor=BatchCursor.from_manifest(Path(data['path']),data['sha256'],partition='target-training',seed=seed,run_target_budget=4)
        state=TrainingState(model,MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule()),
            cursor=cursor,configuration=config,rng_streams={'initialization':initialization,
                'quantizer':torch.Generator().manual_seed(stream_seed('quantization')),
                'meta':random.Random(stream_seed('meta')),'timing':random.Random(stream_seed('timing'))})
        if seed==72 and damage=='altered_origin_weights':
            with torch.no_grad():next(model.parameters()).add_(.125)
        if seed==72 and damage=='altered_origin_quantizer':state.rng_streams['quantizer'].manual_seed(123)
        initial=_ref(save_snapshot(capture_snapshot(state),tmp_path/f'initial-{seed}'))
        origin_config=e.write(config)
        train_step(state,select_logical_batch(state,4))
        checkpoint=_ref(save_snapshot(capture_snapshot(state),tmp_path/f'warm-{seed}'))
        work=e.write({'trial_id':f'warm-{seed}','family':'warm','status':'completed','curve':[],
            'executions':[{'event_id':f'warm-{seed}:loss','kind':'loss','targets':4,'stream_targets':4,
                           'classifications':['training'],'source_intervals':[[prefix,f'doc-{seed}',0,4]]}],
            'cost':{'wall_interval':{'clock':'UTC','start':datetime.fromisoformat(warm_start).timestamp(),
                                    'end':datetime.fromisoformat(warm_start).timestamp()+1}}})
        parents[seed]=_ref(_write(tmp_path/f'parent-{seed}'/'warmstart.json',warmstart_report(
            ArtifactRef(**checkpoint),'target_train',{'complete_seconds':1.,'execution_record':work,
                'ledger':{k:getattr(state.ledger,k) for k in ('stream','passes','elapsed','events')}})))
        producer=receipt([origin_config],[initial,checkpoint,parents[seed],work],subject=work)
        lineages[seed]={'seed':seed,'origin_config':origin_config,'initial_checkpoint':initial,
                        'parents':[{'artifact':parents[seed],'kind':'warmstart','producer':producer}]}
        if derived_parent:
            from ralup.calibrate import import_warmstart
            from ralup.warmstart import create_derived_warmstart
            previous=parents[seed];fresh_model=ByteTransformer(ModelConfig(1,8,2))
            fresh=TrainingState(fresh_model,MethodAdapter(dict(fresh_model.named_parameters()),default_parameter_groups(fresh_model),AdamWUpdateRule()),
                cursor=BatchCursor.from_manifest(Path(data['path']),data['sha256'],partition='target-training',seed=seed,run_target_budget=4),
                configuration={**config,'artifact_role':'continuation'})
            fresh=import_warmstart(ArtifactRef(**previous),fresh,partition='target_train',continuation_targets=4)
            trial=e.write({'seed':seed,'parent':previous,'source_identity':source})
            fresh.configuration['search_trial']=e.read(trial)
            train_step(fresh,select_logical_batch(fresh,4))
            derived_checkpoint=_ref(save_snapshot(capture_snapshot(fresh),tmp_path/f'derived-{seed}'))
            selection_proof=e.write({'status':'fixture','scope':'derived-parent admission only'})
            original_receipt=receipt([trial,previous],[derived_checkpoint,selection_proof])
            parents[seed]=_ref(create_derived_warmstart(ArtifactRef(**derived_checkpoint),tmp_path/f'derived-parent-{seed}.json',
                completed_attempts=[ArtifactRef(**original_receipt['finish'])],selection_evidence=[ArtifactRef(**selection_proof)]))
            lineages[seed]['parents'].append({'artifact':parents[seed],'kind':'derived_model_only','trial_config':trial,
                'producer':receipt([trial,previous],[parents[seed],derived_checkpoint])})
    if damage=='shared_parent':
        parents[72]=parents[71]
    elif damage=='repackaged_parent':
        # A new report hash/path cannot make the same checkpoint a fresh unit.
        parents[72]=e.change(parents[71],extra_label='seed72')
        lineages[72]={**lineages[71],'seed':72}
        lineages[72]['parents']=[{'artifact':parents[72],'kind':'warmstart',
            'producer':receipt([lineages[71]['origin_config']],[parents[72],lineages[71]['initial_checkpoint'],e.read(parents[71])['checkpoint']])}]
    elif damage=='origin_label':
        # Distinct seed/config metadata still imports the original RNG trajectory.
        lineages[72]={**lineages[71],'seed':72,'origin_config':e.change(lineages[71]['origin_config'],seed=72)}
        parents[72]=parents[71]
    rows=[];configs=[]
    def result(seed,method,parent,*,start=replication_start):
        config={'seed':seed,'budget':8,'source_identity':source,'parent':parent,
                'resolved':{'data':{'manifest':data}},'recipe':{'method':method}}
        config_ref=e.write(config);key=hashlib.sha256(canonical_bytes(config)).hexdigest()
        row={'trial_id':key,'family':method,'seed':seed,'budget':8,'status':'completed','source_identity':source,
             'trial_config':config_ref,'query_inputs':query,'recipe':config['recipe'],'model':'core',
             'curve':[{'budget':8,'development_bpb':2.}],
             'executions':[{'event_id':key+':loss','kind':'loss','targets':8,'stream_targets':8,
                            'classifications':['training'],'source_intervals':[[prefix,'doc',0,8]]}],
             'cost':{'parent':parent,'complete_seconds':10.,'wall_interval':{'clock':'UTC',
                 'start':datetime.fromisoformat(start).timestamp(),'end':datetime.fromisoformat(start).timestamp()+10}}}
        reference=e.write(row)
        return {**row,'evidence':reference},config_ref
    tuning_seed=71 if damage=='retained_tuning_seed' else 10
    tuning,tuning_config=result(tuning_seed,'AdamW',parents[tuning_seed],start=warm_start)
    tuning_producer=receipt([tuning_config],[tuning['evidence']],subject=tuning['evidence'])
    for method in ('AdamW','Muon'):
        for seed in (71,72):
            parent=parents[seed]
            if damage=='unmatched_methods' and method=='Muon' and seed==71:parent=parents[72]
            row,config=result(seed,method,parent);rows.append(row);configs.append(config)
    inventory=['AdamW','Muon','Muon+','SOAP','Madam','EWGS','TRS','QuEST','QAR',
               'stochastic_rounding','learned_scale','damping_masking','MetaQuant']
    proof=e.write({'scope':'finite engineering authority fixture, no scientific eligibility'})
    design={'schema':'ralup.finalist_replication_design.v1','synthetic':False,'source_identity':source,
            'created_utc':'2026-10-11T02:00:00+00:00','fresh_seeds':[71,72],'tuning_seeds':[10],
            'trial_configs':configs,'full_budget':8,
            'seed_lineages':list(lineages.values()),
            'tuning_runs':[{'result':tuning['evidence'],'config':tuning_config,'producer':tuning_producer}],
            'family_status':{name:{'status':'included' if name in ('AdamW','Muon') else 'unresolved',
                'reason':'finite fixture only','evidence':[proof]} for name in inventory}}
    design_ref=e.write(design)
    tuning_refs=[tuning['evidence']]
    if damage=='omitted_tuning_result':
        hidden,_=result(71,'AdamW',parents[71],start=warm_start);tuning_refs.append(hidden['evidence'])
    assessment=e.write({'replication_design':design_ref,'status':'completed','result_refs':tuning_refs})
    completion=e.write({'schema':'ralup.finalist_replication_completion.v1','status':'completed','design':design_ref,
        'result_refs':[row['evidence'] for row in rows],
        'producers':[receipt([design_ref],[row['evidence']],start=replication_start,subject=row['evidence']) for row in rows],
        'assessments':{hashlib.sha256(canonical_bytes(row['recipe'])).hexdigest():{'settled':True,'lr_bracketed':True,
            'edge':False,'missing_mechanism':False,'tuning_evidence':assessment} for row in rows}})
    analysis=e.write({'replication_design':design_ref,'replication_completion':completion})
    return rows,{'selection_analysis':analysis,'fresh_seeds':[71,72]}


def test_complete_replication_inventory_accepts_distinct_matched_training_units(tmp_path,monkeypatch):
    from ralup.search import _replication_authority
    rows,eligibility=replication_fixture(tmp_path,monkeypatch)
    assessed,_=_replication_authority(rows,eligibility)
    assert len(assessed)==4 and all(row['fresh_seed'] for row in assessed)
    # Each child includes its disjoint ten-second warm and replication regions.
    assert [row['cost']['complete_seconds'] for row in assessed]==[20.]*4


def test_typed_derived_seed_lineage_preserves_separate_unknown_parent_cost_gate(tmp_path,monkeypatch):
    from ralup.search import _replication_authority,_seed_parent_lineage,_read
    from ralup.records import ArtifactRef
    from ralup.meta import selection
    rows,eligibility=replication_fixture(tmp_path,monkeypatch,derived_parent=True)
    analysis=_read(ArtifactRef(**eligibility['selection_analysis']))
    design=_read(ArtifactRef(**analysis['replication_design']))
    manifest=json.loads(selection.DATA_INDEX.read_bytes())['manifest']
    units={record['seed']:_seed_parent_lineage(record,design['source_identity'],manifest)
           for record in design['seed_lineages']}
    assert all(units[row['seed']]['parent']==row['cost']['parent'] for row in rows)
    assert len({unit['initial_weights'] for unit in units.values()})==3
    # The existing typed derived schema has aggregate historical cost. Successful
    # lineage validation cannot invent missing physical ancestor IDs/costs.
    with pytest.raises(ValueError,match='Complete physical execution and parent costs'):
        _replication_authority(rows,eligibility)


@pytest.mark.parametrize('damage',['retained_tuning_seed','omitted_tuning_result','shared_parent',
                                  'repackaged_parent','origin_label','unmatched_methods',
                                  'altered_origin_weights','altered_origin_quantizer'])
def test_replication_rejects_reused_or_unaccounted_training_units(tmp_path,monkeypatch,damage):
    from ralup.search import _replication_authority
    rows,eligibility=replication_fixture(tmp_path,monkeypatch,damage)
    with pytest.raises(ValueError,match='seed|lineage|parent|tuning|origin'):
        _replication_authority(rows,eligibility)
