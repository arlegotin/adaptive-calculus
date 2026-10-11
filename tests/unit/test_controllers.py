import importlib.util
import math
import pytest
import torch
from ralup.policy.features import FeatureVector


def api():
    assert importlib.util.find_spec('ralup.policy.controllers') is not None, 'M09 controller module required'
    from ralup.policy import controllers
    return controllers


def test_exact_capacities_initialization_and_lr_range():
    c=api();f=FeatureVector(torch.arange(14).float())
    assert c.count_parameters(c.Affine45())==45
    assert c.count_parameters(c.LearnedLR49())==49
    assert c.count_parameters(c.Static3())==3
    assert c.Affine45()(f)==c.PolicyAction(0.,1.,1.)
    assert c.Static3()(f)==c.PolicyAction(0.,1.,1.)
    lr=c.LearnedLR49();assert lr(f)==1 and lr.max_multiplier==32
    with torch.no_grad():lr.output.bias.fill_(100)
    assert lr(f)==32
    assert c.LearnedLR49(max_multiplier=4).max_multiplier==4


def test_bounds_information_removal_and_registered_state():
    c=api();f=FeatureVector(torch.ones(14))
    a=c.Affine45()
    with torch.no_grad():a.linear.bias.copy_(torch.tensor([10.,100.,-10.]))
    assert a(f)==c.PolicyAction(1.,4.,.125)
    state=c.freeze_controller(a,family='no_history')
    assert c.evaluate_controller(state,f)==a(f)
    with torch.no_grad():a.linear.weight.zero_();a.linear.bias.copy_(torch.tensor([0.,0.,1.]));a.linear.weight[0,8]=1
    assert c.evaluate_controller(c.freeze_controller(a),f).alpha==1
    assert c.evaluate_controller(c.freeze_controller(a,family='no_history'),f).alpha==0
    for family,want in [('additive_only',c.PolicyAction(0.,4.,1.)),('multiplicative_only',c.PolicyAction(1.,4.,1.)),('assistance_only',c.PolicyAction(0.,1.,.125))]:
        with torch.no_grad():a.linear.bias.copy_(torch.tensor([1.,100.,-10.]))
        assert c.evaluate_controller(c.freeze_controller(a,family=family),f)==want
    with pytest.raises(ValueError):c.freeze_controller(lambda _:None)
    with pytest.raises(ValueError):c.ControllerState('pickle',torch.zeros(45))
    with pytest.raises(ValueError):c.ControllerState('affine45',torch.full((45,),float('nan')))
    with pytest.raises(ValueError):c.PolicyAction(0.,float('nan'),1.)


def test_all_fourteen_features_reach_both_networks_and_coverage_never_input():
    c=api();a=c.Affine45();lr=c.LearnedLR49()
    with torch.no_grad():
        a.linear.bias.copy_(torch.tensor([.5,0.,1.]));a.linear.weight.fill_(.01)
        lr.hidden.weight.fill_(.01);lr.hidden.bias.zero_();lr.output.weight.fill_(1.);lr.output.bias.zero_()
    zero=FeatureVector(torch.zeros(14),{'matrix_id':'forbidden','query_loss':999})
    for i in range(14):
        v=torch.zeros(14);v[i]=1;f=FeatureVector(v)
        assert a(f)!=a(zero) and lr(f)!=lr(zero)
    assert a(zero)==a(FeatureVector(torch.zeros(14)))
    assert lr(zero)==lr(FeatureVector(torch.zeros(14)))
    # Output/source range differs explicitly; shared-range sensitivity exists.
    with torch.no_grad():lr.output.bias.fill_(100)
    assert lr(zero)==32
    assert c.evaluate_controller(c.freeze_controller(lr,family='learned_lr49_shared_range'),zero)==c.PolicyAction(0.,4.,1.)


@pytest.mark.parametrize('family',['affine45','initialized','additive_only','multiplicative_only','assistance_only','no_history','no_cells','norm_matched','learned_lr49','learned_lr49_shared_range','static3'])
def test_every_artifact_family_plain_json_roundtrip_and_tamper_rejection(tmp_path,family):
    c=api()
    from ralup.records import ArtifactRef,sha256_file
    refs=[]
    for name in ('normalizer','B0','data','training'):
        p=tmp_path/(name+'.json');p.write_text('{}');refs.append(ArtifactRef(p,sha256_file(p)))
    model=c.LearnedLR49() if family.startswith('learned_lr') else c.Static3() if family=='static3' else c.Affine45()
    state=c.freeze_controller(model,family=family)
    artifact=c.ControllerArtifact(family,state.parameters,*refs)
    import json
    public=json.loads(json.dumps(artifact.to_json(),allow_nan=False))
    restored=c.ControllerArtifact.from_json(public)
    assert torch.equal(restored.parameters,artifact.parameters) and restored.family==family
    assert public['privileged_candidate_rms']==(family=='norm_matched')
    assert public['h_range']==[.25,32 if family=='learned_lr49' else 4]
    with pytest.raises(ValueError):c.ControllerArtifact.from_json({**public,'pickle':'evil'})
    public['privileged_candidate_rms']=not public['privileged_candidate_rms']
    with pytest.raises(ValueError):c.ControllerArtifact.from_json(public)
    refs[2].path.write_text('{"changed":true}')
    with pytest.raises(ValueError):artifact.to_json()


def test_freeze_rejects_added_executable_owner_and_wrong_shape():
    c=api();a=c.Affine45();a.parameters=lambda: iter([])
    with pytest.raises(ValueError):c.freeze_controller(a)
    a=c.Affine45();a.linear.weight=torch.nn.Parameter(torch.zeros(3,13))
    with pytest.raises(ValueError):c.freeze_controller(a)
    a=c.Affine45();a.linear.weight.register_hook(lambda x:x)
    with pytest.raises(ValueError):c.freeze_controller(a)
