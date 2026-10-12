"""Data-only engineering fixtures never imply selected scientific policies."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from ralup.config import canonical_bytes
from ralup.records import ArtifactRef, read_artifact, write_immutable_json
from ralup.search import source_identity


def put(root, name, value):
    return write_immutable_json(root/name, value)


def ref(value):
    return {'path': str(value.path), 'sha256': value.sha256}


def parent(root, *, nested_warm=False):
    inputs={key:ref(put(root,key+'.json',{'identity':key,'scope':'synthetic-engineering'}))
            for key in ('stream','model','budget','scorer','warm_start','continuation','normalizer','learning')}
    warm={'run_id':'warm','status':'completed','parents':[],
          'events':[{'event_id':'warm-pass','kind':'loss','targets':4,'stream_targets':4,
                     'classifications':['ordinary','backward'],'source_intervals':[['corpus','warm-doc',0,4]]}],
          'wall_intervals':None,'unknown_ancestor_cost':True}
    if nested_warm:
        warm['parents']=[{'run_id':'warm-ancestor','status':'completed','parents':[],
            'events':[{'event_id':'warm-ancestor-pass','kind':'loss','targets':4,'stream_targets':4,
                'classifications':['ordinary','backward'],'source_intervals':[['corpus','ancestor-doc',0,4]]}],
            'wall_intervals':None,'unknown_ancestor_cost':True}]
    inputs['warm_start']=ref(put(root,'warm-authority.json',{'scope':'synthetic-engineering','cost':warm}))
    value={'schema':'ralup.attribution_parent.v1','scope':'synthetic-engineering',
        'source_identity':source_identity(),'source_revision':'engineering-current',
        'inputs':inputs,'seeds':[11,12,13,14],'budget':64,
        'feature_contract':{'features':14,'history':True,'ema':True,'mask_after_scan':True,'normalizer_fit':'META-only'},
        'history_groups':[{'names':['a','b'],'shape':[2,2]}],
        'history_permutation':{'predeclared':True,'groups':[['a','b']],'indices':[[1,0]],'retrain':True},
        'assisted_fraction':.25,'budget_augmentation':{'extra_data_targets':64,'extra_compute_targets':64},
        'selected_after_protected':False}
    return put(root,'parent.json',value)


def api():
    from ralup.analysis import attribution
    return attribution


def test_attribution_interface_is_present():
    module=api()
    assert callable(module.make_ablation_trials)
    assert callable(module.analyze_attribution)
    assert callable(module.check_extension_eligibility)


@pytest.mark.parametrize('contrast',[
    'learned_lr_wider','learned_lr_shared','static','initialized','additive_only','multiplicative_only',
    'assistance_only','alpha_zero','history_removed','cell_removed','history_permuted',
    'history_scramble_inference','norm_matched','magnitude_preconditioned_additive','madam',
    'fixed_scale','dynamic_scale','learned_scale','identity_rotation','hadamard_rotation',
    'random_assistance','rounding','masking','damping','extra_data','extra_compute'])
def test_prioritized_trials_match_and_declare_real_obligations(tmp_path,contrast):
    module=api();p=parent(tmp_path)
    trials=module.make_ablation_trials(p,contrast,'explore',True)
    assert len(trials)==8
    for candidate,control in zip(trials[::2],trials[1::2]):
        assert candidate.seed==control.seed
        assert candidate.stream_hash==control.stream_hash
        assert candidate.pairing_contract==control.pairing_contract
        assert candidate.budget==control.budget
        assert control.claim_status=='exploratory'
        assert control.comparison_kind=='retuned_competition'
        assert control.retuning_status=='required'
        contract=json.loads(control.attribution_contract)
        assert contract['backbone_lr_tuning']=='independent'
        if contrast not in ('extra_data','extra_compute'):
            assert control.feature_cost_retained
            assert contract['compute_information']=={'features':14,'history':True,'ema':True,'mask_after_scan':True,'normalizer_fit':'META-only'}
        if contrast in ('history_removed','cell_removed','history_permuted','alpha_zero'):
            assert control.retraining_status=='required'
        if contrast=='norm_matched': assert control.is_privileged_diagnostic
        if contrast=='history_scramble_inference':assert control.retraining_status=='not_retrained'
        if contrast=='learned_lr_wider':assert contract['variant_settings']['h_range']==[.25,32]
        if contrast=='learned_lr_shared':assert contract['variant_settings']['h_range']==[.25,4]


def test_intervention_does_not_claim_retuned_or_retrained_inference_scramble(tmp_path):
    module=api();p=parent(tmp_path)
    c=module.make_ablation_trials(p,'history_scramble_inference','explore',False)[1]
    assert c.comparison_kind=='original_setting_intervention'
    assert c.retuning_status=='not_retuned'
    assert c.retraining_status=='not_retrained'
    c=module.make_ablation_trials(p,'alpha_zero','explore',False)[1]
    assert c.retraining_status=='required'


@pytest.mark.parametrize('mutation', ['unknown_parent','stale_source','feature_skipped','mismatched_shape','invalid_permutation','target_normalizer'])
def test_parent_and_history_are_authenticated_before_trial_build(tmp_path,mutation):
    module=api();p=parent(tmp_path);value=read_artifact(p)
    if mutation=='unknown_parent':value['inputs']['warm_start']['sha256']='0'*64
    if mutation=='stale_source':value['source_identity']={}
    if mutation=='feature_skipped':value['feature_contract']['history']=False
    if mutation=='mismatched_shape':value['history_groups'].append({'names':['c'],'shape':[3,2]});value['history_permutation']['groups']=[['a','c']]
    if mutation=='invalid_permutation':value['history_permutation']['indices']=[[1,1]]
    if mutation=='target_normalizer':value['feature_contract']['normalizer_fit']='target'
    bad=put(tmp_path,'bad-parent.json',value)
    with pytest.raises(ValueError):module.make_ablation_trials(bad,'history_permuted','explore',True)


def test_post_test_parent_remains_exploratory_and_science_is_not_fixture(tmp_path):
    module=api();p=parent(tmp_path);v=read_artifact(p);v['selected_after_protected']=True
    selected=put(tmp_path,'selected.json',v)
    assert all(t.claim_status=='exploratory' for t in module.make_ablation_trials(selected,'static','explore',True))
    with pytest.raises(ValueError):module.make_ablation_trials(selected,'static','confirm',True)
    v['scope']='scientific-development';science=put(tmp_path,'science.json',v)
    with pytest.raises(ValueError,match='producer'):module.make_ablation_trials(science,'static','explore',True)


def runs(root, contrast='static', deltas=None, *, failed=None, retune=True, phase='explore', fresh=True, predeclared=True, post_test=False, nested_warm=False):
    module=api();p=parent(root,nested_warm=nested_warm);trials=module.make_ablation_trials(p,contrast,phase,retune)
    design={'schema':'ralup.attribution_family.v1','scope':'synthetic-engineering','phase':phase,
        'contrasts':[contrast],'seeds':[11,12,13,14],'alpha':.05,'multiplicity':'bonferroni',
        'predeclared':predeclared,'fresh_design':fresh,'selection_before_outcomes':not post_test,
        'retune':retune,'parent':ref(p),'regression_covariates':['entropy','length'],'design_id':'fresh-engineering-family'}
    d=put(root,'design.json',design);results=[];deltas=deltas or [-.2,-.21,-.19,-.22]
    for i,t in enumerate(trials):
        contract=json.loads(t.attribution_contract);seed=t.seed;is_control=i%2==1
        receipt={'schema':'ralup.attribution_execution.v1','scope':'synthetic-engineering',
            'trial_hash':t.config_hash,'design_sha256':d.sha256,'parent_sha256':p.sha256,
            'training_status':'retrained' if contract['retraining_status']=='required' else 'not_retrained',
            'tuning_status':'independently_retuned' if retune else 'not_retuned','backbone_lr_tuned':retune,
            'feature_scans':4,'features_per_scan':14,'history_updates':4,'ema_updates':4,'meta_feature_scans':4,
            'operational_telemetry':True,'complete_target_training_seed':True,'learning_design_sha256':read_artifact(p)['inputs']['learning']['sha256']}
        execution=put(root,f'execution-{i}.json',receipt)
        warm=read_artifact(ArtifactRef(**read_artifact(p)['inputs']['warm_start']))['cost']
        cost={'run_id':f'run-{i}','status':'diverged' if failed==i else 'completed','parents':[warm],
              'events':[{'event_id':f'pass-{i}','kind':'loss','targets':64,'stream_targets':64,
                         'classifications':['backward','ordinary'],'source_intervals':[['corpus','doc',0,64]]}],
              'wall_intervals':None,'unknown_ancestor_cost':True,'unknown_partial_work':failed==i}
        if contract['extra_targets']:
            event=cost['events'][0];event['targets']+=contract['extra_targets'];event['source_intervals'][0][3]+=contract['extra_targets']
            if contract['extra_targets_kind']=='extra_data_targets':event['stream_targets']+=contract['extra_targets']
        receipt['cost_sha256']=hashlib.sha256(canonical_bytes(cost)).hexdigest()
        execution=put(root,f'cost-execution-{i}.json',receipt)
        score=None if failed==i else 1.+(deltas[i//2] if not is_control else 0)
        value={'schema':'ralup.attribution_run.v1','scope':'synthetic-engineering',
               'trial':contract,'execution':ref(execution),'design_sha256':d.sha256,
               'status':cost['status'],'development_bpb':score,'cost':cost,
               'articles':[{'document':'doc-a','development_bpb':score,'entropy':1.,'length':64.},
                           {'document':'doc-b','development_bpb':score,'entropy':2.,'length':128.}]}
        results.append(put(root,f'run-{i}.json',value))
    return results,{'declaration':ref(d)}


@pytest.mark.parametrize('deltas,outcome',[([-.2,-.21,-.19,-.22],'favorable'),([.2,.21,.19,.22],'unfavorable'),([-.01,.01,-.02,.02],'inconclusive'),([0,0,0,0],'inconclusive')])
def test_analysis_reports_paired_direction_and_cost_unknowns(tmp_path,deltas,outcome):
    r,d=runs(tmp_path,deltas=deltas);report=api().analyze_attribution(r,d)
    assert report['contrasts']['static']['outcome']==outcome
    assert report['claim_status']=='exploratory'
    assert report['scientific_mechanism_established'] is False
    assert report['project_cost']['known_total_targets']==516
    assert report['project_cost']['known_unique_targets']==68
    assert report['project_cost']['seconds'] is None
    assert report['project_cost']['common_engineering_seconds'] is None
    assert report['regressions']['role']=='descriptive exploratory; seed remains independent unit'


@pytest.mark.parametrize('tamper',['stream','scorer','parent','missing_retune','missing_retrain','skipped_feature','fake_inference','design','duplicate_seed','status_score'])
def test_analysis_rejects_mismatched_pair_or_unearned_learning_labels(tmp_path,tamper):
    r,d=runs(tmp_path,contrast='history_removed');v=read_artifact(r[1]);receipt=read_artifact(ArtifactRef(**v['execution']))
    if tamper in ('stream','scorer'):v['trial']['pairing'][tamper]='0'*64
    if tamper=='parent':v['trial']['parent']['sha256']='0'*64
    if tamper=='missing_retune':receipt['tuning_status']='not_retuned'
    if tamper=='missing_retrain':receipt['training_status']='not_retrained'
    if tamper=='skipped_feature':receipt['feature_scans']=0
    if tamper=='fake_inference':v['trial']['retraining_status']='not_retrained'
    if tamper=='design':v['design_sha256']='0'*64
    if tamper=='duplicate_seed':v['trial']['seed']=11;r[3]=put(tmp_path,'duplicate.json',v)
    if tamper=='status_score':v['status']='diverged'
    v['execution']=ref(put(tmp_path,'tampered-execution.json',receipt))
    r[1]=put(tmp_path,'tampered-run.json',v)
    with pytest.raises(ValueError):api().analyze_attribution(r,d)


def test_divergence_and_missing_seed_deny_favorable_inference(tmp_path):
    r,d=runs(tmp_path,failed=3);report=api().analyze_attribution(r,d)
    assert report['contrasts']['static']['outcome']=='failed'
    assert report['contrasts']['static']['planned_n']==4
    assert report['contrasts']['static']['conditional_interval'] is not None
    report=api().analyze_attribution(r[:-1],d)
    assert report['contrasts']['static']['outcome']=='incomplete'


def test_confirmatory_family_requires_predeclaration_multiplicity_and_freshness(tmp_path):
    r,d=runs(tmp_path);decl=read_artifact(ArtifactRef(**d['declaration']))
    for key,value in [('multiplicity','none'),('fresh_design',False),('predeclared',False)]:
        v=copy.deepcopy(decl);v['phase']='confirm';v[key]=value
        bad=put(tmp_path,key+'.json',v)
        with pytest.raises(ValueError):api().analyze_attribution(r,{'declaration':ref(bad)})


def extension(root,name='larger_shape',outcome='favorable'):
    observation=put(root,'observation.json',{'schema':'ralup.extension_observation.v1','scope':'synthetic-engineering',
        'outcome':outcome,'supported_local_evidence':True,'diagnosis':None,'selected_after_protected':False})
    contract={'schema':'ralup.extension_contract.v1','extension':name,'scope':'synthetic-engineering',
        'source_identity':source_identity(),'observation_sha256':observation.sha256,'numerical_contract':{'representation':'E2M1-full-plus-minus-6'},
        'provenance':{'corpus':'synthetic-current'},'activation_gradient_contract':{'backward':'STE'},
        'packing_contract':{'layout':'explicit-engineering'},'learning_contract':{'finalized':True,'own_tuning':True},
        'budget':{'includes_full_inherited_cost':True,'measured_seconds':10.,'measured_bytes':100,'reserved_seconds':20.,'reserved_bytes':200},
        'branch_contract':{'parameters':19300000,'meta_seeds':[101,102],'reset':'fresh_all_states','carry':'complete_current_state',
                           'representation':'E2M1-full-plus-minus-6','block_size':None,'corpus':'new-independent-synthetic'}}
    c=put(root,'extension.json',contract)
    return {'observation':ref(observation),'contract':ref(c)}


@pytest.mark.parametrize('name',['larger_shape','independent_meta_clusters','from_scratch','carry_state','fp8','mxfp4','w4a4','new_corpus'])
def test_extensions_require_own_exact_contracts(tmp_path,name):
    module=api()
    assert module.check_extension_eligibility({},name).status=='blocked'
    e=extension(tmp_path,name);c=read_artifact(ArtifactRef(**e['contract']))
    if name=='fp8':c['branch_contract']['representation']='E4M3-FP8'
    if name=='mxfp4':c['branch_contract'].update(representation='E2M1-E8M0-MXFP4',block_size=32)
    if name=='w4a4':c['branch_contract']['representation']='W4A4'
    if name in ('fp8','mxfp4','w4a4'):c['numerical_contract']['representation']=c['branch_contract']['representation']
    e['contract']=ref(put(tmp_path,'exact-extension.json',c))
    gate=module.check_extension_eligibility(e,name)
    assert gate.status=='pass'
    assert gate.evidence


@pytest.mark.parametrize('mutation',['negative','null','budget','learning','source','post_test','shape','corpus','representation','diagnosis'])
def test_extension_negative_and_unknown_paths(tmp_path,mutation):
    e=extension(tmp_path);o=read_artifact(ArtifactRef(**e['observation']));c=read_artifact(ArtifactRef(**e['contract']))
    expected='blocked'
    if mutation=='negative':o['outcome']='unfavorable';expected='negative'
    if mutation=='null':o['outcome']='inconclusive';expected='inconclusive'
    if mutation=='budget':c['budget']['measured_seconds']=None
    if mutation=='learning':c['learning_contract']['finalized']=False
    if mutation=='source':c['source_identity']={}
    if mutation=='post_test':o['selected_after_protected']=True
    if mutation=='shape':c['branch_contract']['parameters']=1
    if mutation=='corpus':e=extension(tmp_path/'nested','new_corpus');c=read_artifact(ArtifactRef(**e['contract']));c['branch_contract']['corpus']='WikiText-2'
    if mutation=='representation':e=extension(tmp_path/'nested','fp8');c=read_artifact(ArtifactRef(**e['contract']))
    if mutation=='diagnosis':o.update(outcome='inconclusive',diagnosis={'hypothesis':'transfer missing','falsification':'fresh full shape','predeclared':True})
    e['observation']=ref(put(tmp_path,'new-observation.json',o));c['observation_sha256']=e['observation']['sha256']
    e['contract']=ref(put(tmp_path,'new-contract.json',c))
    name=c['extension'];gate=api().check_extension_eligibility(e,name)
    assert gate.status==('pass' if mutation=='diagnosis' else expected)


def test_magnitude_control_is_first_order_normalized_z_and_distinct_from_exponential():
    from ralup.policy.actions import magnitude_preconditioned_additive, multiplicative_proposal
    u=torch.tensor([[-2.,0.,4.]],dtype=torch.float32);d=torch.tensor([[.02,.03,-.04]],dtype=torch.float32)
    a=torch.sign(u)*d/u.square().mean().sqrt();z=u*a;c=d.square().mean().sqrt()/z.square().mean().sqrt()
    result=magnitude_preconditioned_additive(u,d)
    assert torch.allclose(result,c*z,atol=1e-8)
    assert not torch.equal(result,multiplicative_proposal(u,d))
    assert torch.equal(magnitude_preconditioned_additive(torch.zeros_like(u),d),torch.zeros_like(u))


def test_registry_and_config_catalog_are_complete():
    from ralup.baselines.registry import attribution_registry
    entries=attribution_registry()
    assert entries['magnitude_preconditioned_additive']['geometry']=='first_order_c_times_z'
    assert entries['magnitude_preconditioned_additive']['is_madam'] is False
    root=Path(__file__).resolve().parents[2]
    contrasts=[]
    for name in ('actions','information','scales','noise','budgets'):
        c=json.loads((root/'configs/ablations'/f'{name}.json').read_bytes())
        assert c['schema']=='ralup.ablation_catalog.v1'
        contrasts.extend(c['contrasts'])
    assert set(contrasts)==set(entries)


def test_cli_engineering_artifacts_are_analyzed_and_real_prerequisite_is_explicit(tmp_path,capsys):
    from ralup.__main__ import main
    r,d=runs(tmp_path)
    cfg=put(tmp_path,'cli.json',{'schema':'ralup.attribution_suite.v1','scope':'synthetic-engineering',
        'runs':[ref(x) for x in r],'declared_family':d,'output':str(tmp_path/'report.json')})
    assert main(['run','--phase','explore','--suite','attribution','--config',str(cfg.path)])==0
    assert json.loads(capsys.readouterr().out)['claim_status']=='exploratory'
    bad=put(tmp_path,'no-parent-cli.json',{'schema':'ralup.attribution_suite.v1','scope':'scientific-development','parent':None})
    with pytest.raises(SystemExit) as error:main(['run','--phase','explore','--suite','attribution','--config',str(bad.path)])
    assert error.value.code==2
    assert 'authentic' in capsys.readouterr().err.lower()


@pytest.mark.parametrize('contrast',['history_removed','alpha_zero','history_scramble_inference','magnitude_preconditioned_additive','norm_matched','extra_compute'])
@pytest.mark.parametrize('retune',[False,True])
def test_learning_obligations_and_interventions_execute_as_declared(tmp_path,contrast,retune):
    r,d=runs(tmp_path,contrast=contrast,retune=retune)
    result=api().analyze_attribution(r,d)
    row=result['contrasts'][contrast]
    assert row['comparison_kind']==('retuned_competition' if retune else 'original_setting_intervention')
    assert result['deployment_cost']['mean_warm_inclusive']['known_total_targets']==(100 if contrast=='extra_compute' else 68)
    if contrast in ('norm_matched','history_scramble_inference'):
        assert row['mechanism_claim_eligible'] is False


def test_fresh_engineering_confirmation_and_posttest_choice(tmp_path):
    r,d=runs(tmp_path,phase='confirm')
    report=api().analyze_attribution(r,d)
    assert report['claim_status']=='confirmatory-engineering'
    assert report['scientific_mechanism_established'] is False
    r,d=runs(tmp_path/'posttest',post_test=True)
    report=api().analyze_attribution(r,d)
    assert report['claim_status']=='exploratory'
    assert report['fresh_design_required'] is True


@pytest.mark.parametrize('mutation',['unbound_cost','missing_warm','different_warm','skipped_telemetry'])
def test_cost_receipt_authentication_and_retained_warm_scan_telemetry(tmp_path,mutation):
    r,d=runs(tmp_path);v=read_artifact(r[1]);receipt=read_artifact(ArtifactRef(**v['execution']))
    if mutation=='unbound_cost':v['cost']['events'][0]['targets']=1
    if mutation=='missing_warm':v['cost']['parents']=[]
    if mutation=='different_warm':v['cost']['parents'][0]['run_id']='other'
    if mutation=='skipped_telemetry':receipt['operational_telemetry']=False
    if mutation!='unbound_cost':receipt['cost_sha256']=hashlib.sha256(canonical_bytes(v['cost'])).hexdigest()
    v['execution']=ref(put(tmp_path,'bad-cost-execution.json',receipt));r[1]=put(tmp_path,'bad-cost-run.json',v)
    with pytest.raises(ValueError):api().analyze_attribution(r,d)


def test_family_multiplicity_corrects_every_planned_contrast_and_freshness(tmp_path):
    r,d=runs(tmp_path);decl=read_artifact(ArtifactRef(**d['declaration']))
    decl['contrasts']=['static','history_removed'];new=put(tmp_path,'two-family.json',decl)
    rewritten=[]
    for i,item in enumerate(r):
        value=read_artifact(item);value['design_sha256']=new.sha256
        execution=read_artifact(ArtifactRef(**value['execution']));execution['design_sha256']=new.sha256
        value['execution']=ref(put(tmp_path,f'family-exec-{i}.json',execution))
        rewritten.append(put(tmp_path,f'family-run-{i}.json',value))
    report=api().analyze_attribution(rewritten,{'declaration':ref(new)})
    assert report['multiplicity']['interval_confidence']==.975
    assert report['contrasts']['history_removed']['outcome']=='incomplete'
    assert len(report['contrasts']['static']['interval'])==2


@pytest.mark.parametrize('mutation',['same_normalizer_target','semantic_cost_change','extension_unbound_observation','inherited_footprint'])
def test_strict_identity_and_new_branch_budget_boundary(tmp_path,mutation):
    module=api()
    if mutation=='same_normalizer_target':
        p=parent(tmp_path);v=read_artifact(p)
        v['inputs']['normalizer']=ref(put(tmp_path,'target-normalizer.json',{'scope':'scientific-protected','identity':'normalizer'}))
        bad=put(tmp_path,'cross-scope-parent.json',v)
        with pytest.raises(ValueError):module.make_ablation_trials(bad,'static','explore',True)
    elif mutation=='semantic_cost_change':
        r,d=runs(tmp_path);v=read_artifact(r[1]);v['cost']['events'][0]['classifications']=['query']
        r[1]=put(tmp_path,'semantic-cost.json',v)
        with pytest.raises(ValueError,match='cost receipt'):module.analyze_attribution(r,d)
    else:
        e=extension(tmp_path);c=read_artifact(ArtifactRef(**e['contract']))
        if mutation=='extension_unbound_observation':c['observation_sha256']='0'*64
        else:c['budget']['includes_full_inherited_cost']=False
        e['contract']=ref(put(tmp_path,'bad-boundary.json',c))
        assert module.check_extension_eligibility(e,'larger_shape').status=='blocked'


def test_finite_divergence_keeps_observed_unequal_feature_work(tmp_path):
    r,d=runs(tmp_path,failed=3);v=read_artifact(r[3]);execution=read_artifact(ArtifactRef(**v['execution']))
    execution.update(feature_scans=1,history_updates=1,ema_updates=1,complete_target_training_seed=False)
    v['execution']=ref(put(tmp_path,'early-divergence-execution.json',execution));r[3]=put(tmp_path,'early-divergence.json',v)
    report=api().analyze_attribution(r,d)
    assert report['contrasts']['static']['outcome']=='failed'
    assert report['contrasts']['static']['failed_seeds']==[12]


def test_canonical_trial_types_cannot_masquerade_as_exact_identity(tmp_path):
    r,d=runs(tmp_path);v=read_artifact(r[1]);v['trial']['seed']=float(v['trial']['seed'])
    r[1]=put(tmp_path,'float-seed.json',v)
    with pytest.raises(ValueError):api().analyze_attribution(r,d)


def test_regressions_keep_distinct_mechanism_names(tmp_path):
    r,d=runs(tmp_path);result=api().analyze_attribution(r,d)
    assert all(row['contrast']=='static' for row in result['regressions']['per_article'])
    assert 'static' in result['regressions']['by_contrast']
    assert result['regressions']['by_contrast']['static']['entropy']['n_seeds']==4


def test_augmented_ordinary_control_must_charge_actual_additional_exposure(tmp_path):
    r,d=runs(tmp_path,contrast='extra_compute');v=read_artifact(r[1]);event=v['cost']['events'][0]
    event['targets']=64;event['source_intervals'][0][3]=64
    execution=read_artifact(ArtifactRef(**v['execution']));execution['cost_sha256']=hashlib.sha256(canonical_bytes(v['cost'])).hexdigest()
    v['execution']=ref(put(tmp_path,'missing-extra-execution.json',execution));r[1]=put(tmp_path,'missing-extra.json',v)
    with pytest.raises(ValueError,match='augmentation'):api().analyze_attribution(r,d)


@pytest.mark.parametrize('case',['duplicate_stream','duplicate_extra_compute','warm_event_reused','nested_warm_event_reused'])
def test_completed_budget_rejects_repeated_or_ancestral_physical_passes(tmp_path,case):
    contrast='extra_compute' if case=='duplicate_extra_compute' else 'static'
    r,d=runs(tmp_path,contrast=contrast,nested_warm=case=='nested_warm_event_reused')
    value=read_artifact(r[1]);cost=value['cost'];ordinary=cost['events'][0]
    ordinary.update(targets=64,stream_targets=64,source_intervals=[['corpus','doc',0,64]])
    if case=='duplicate_stream':
        ordinary.update(targets=32,stream_targets=32,source_intervals=[['corpus','doc',0,32]])
        cost['events']=[ordinary,copy.deepcopy(ordinary)]
    elif case=='duplicate_extra_compute':
        query={'event_id':'extra-query','kind':'loss','targets':32,'stream_targets':0,
               'classifications':['query','backward','additional_loss'],'source_intervals':[['corpus','query-doc',0,32]]}
        cost['events']=[ordinary,query,copy.deepcopy(query)]
    else:
        warm=cost['parents'][0]
        ancestor=warm['parents'][0] if case=='nested_warm_event_reused' else warm
        reused=copy.deepcopy(ancestor['events'][0])
        ordinary.update(targets=60,stream_targets=60,source_intervals=[['corpus','doc',0,60]])
        cost['events']=[ordinary,reused]
    receipt=read_artifact(ArtifactRef(**value['execution']))
    receipt['cost_sha256']=hashlib.sha256(canonical_bytes(cost)).hexdigest()
    value['execution']=ref(put(tmp_path,'physical-cost-execution.json',receipt))
    r[1]=put(tmp_path,'physical-cost-run.json',value)
    with pytest.raises(ValueError,match='physical'):api().analyze_attribution(r,d)


@pytest.mark.parametrize('contrast',['static','extra_compute'])
def test_completed_budget_accepts_distinct_authenticated_physical_passes(tmp_path,contrast):
    r,d=runs(tmp_path,contrast=contrast);value=read_artifact(r[1]);cost=value['cost']
    cost['events']=[{'event_id':f'distinct-{i}','kind':'loss','targets':32,'stream_targets':32,
                    'classifications':['ordinary','backward'],'source_intervals':[['corpus','doc',32*i,32*i+32]]}
                   for i in range(2)]
    if contrast=='extra_compute':
        cost['events'].append({'event_id':'distinct-query','kind':'loss','targets':64,'stream_targets':0,
                              'classifications':['query','additional_loss','backward'],
                              'source_intervals':[['corpus','query-doc',0,64]]})
    receipt=read_artifact(ArtifactRef(**value['execution']));receipt['cost_sha256']=hashlib.sha256(canonical_bytes(cost)).hexdigest()
    value['execution']=ref(put(tmp_path,'distinct-execution.json',receipt));r[1]=put(tmp_path,'distinct-run.json',value)
    result=api().analyze_attribution(r,d)
    assert result['contrasts'][contrast]['complete_n']==4
    assert result['project_cost']['raw_classifications']['distinct-0']==['ordinary','backward']


def test_common_panel_regressions_preserve_article_variation_and_seed_slopes(tmp_path):
    r,d=runs(tmp_path)
    for i,reference in enumerate(r):
        value=read_artifact(reference);seed_index=i//2
        deltas=[-.1*(seed_index+1),-.2*(seed_index+1)]
        for j,article in enumerate(value['articles']):
            article['development_bpb']=1.+(deltas[j] if i%2==0 else 0.)
        r[i]=put(tmp_path,f'common-panel-{i}.json',value)
    result=api().analyze_attribution(r,d)
    entropy=result['regressions']['by_contrast']['static']['entropy']
    length=result['regressions']['by_contrast']['static']['length']
    assert entropy['slope']==pytest.approx(-.25)
    assert entropy['n_seeds']==entropy['n_estimable_seeds']==4
    assert [row['slope'] for row in entropy['per_seed']]==pytest.approx([-.1,-.2,-.3,-.4])
    assert length['slope']==pytest.approx(-.25/64)
    assert result['regressions']['method']=='within_seed_article_ols_intercept_then_equal_seed_mean'
    assert result['regressions']['causal_inference'] is False


@pytest.mark.parametrize('case',['one_article','constant_covariate'])
def test_insufficient_within_seed_variation_is_explicitly_null(tmp_path,case):
    r,d=runs(tmp_path)
    for i,reference in enumerate(r):
        value=read_artifact(reference)
        if case=='one_article':value['articles']=value['articles'][:1]
        else:
            for article in value['articles']:article.update(entropy=1.,length=64.)
        r[i]=put(tmp_path,f'insufficient-article-{i}.json',value)
    entropy=api().analyze_attribution(r,d)['regressions']['by_contrast']['static']['entropy']
    assert entropy['slope'] is None
    assert entropy['n_seeds']==4 and entropy['n_estimable_seeds']==0
    assert all(row['slope'] is None for row in entropy['per_seed'])
