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
