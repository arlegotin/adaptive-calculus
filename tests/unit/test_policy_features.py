"""Independent scalar S19 fixtures; deliberate wrong denominators must fail."""
import json
import math
from pathlib import Path
import pytest
import torch
from ralup.quantization.context import prepare_context
from ralup.records import ArtifactRef, sha256_file


def context(u, step=0, scale=1.):
    return prepare_context(u, {'name':'row_fixed_max','previous_scales':torch.full((u.shape[0],1),scale)}, step, None)


def test_all_fourteen_features_hand_computed_without_actions():
    from ralup.policy.features import collect_features
    from ralup.policy.history import MatrixHistory, LossObservation
    u=torch.tensor([[0.,.4,1.,7.]])
    d=torch.tensor([[0.,.1,.2,-.1]])
    g=torch.ones_like(u); m=torch.tensor([[1.,0.,0.,0.]])
    h=MatrixHistory(transition_fraction=.25,reversal_fraction=.5,disagreement_fraction=.75,
                    updates=3,reversal_valid=True,realized_valid=True)
    f=collect_features(u,d,g,m,h,context(u),LossObservation(2.,4.),.3)
    # rms(U)=sqrt(50.16/4); rms(d)=sqrt(.06/4); cos(g,m)=1/2.
    want=[math.log10(math.sqrt(12.54)),0.,math.log10(math.sqrt(.015/12.54)),.5,
          1.,(.1/.35/2+.2/.25/2)/2,0.,.25,.25,.5,.75,.05,.3,-2.]
    torch.testing.assert_close(f.values,torch.tensor(want),atol=1e-6,rtol=1e-6)
    assert f.values.shape==(14,) and not f.values.requires_grad
    assert f.coverage['nonzero_moves']==3 and f.coverage['positive_distance']==2


def test_zero_populations_history_and_first_loss_return_finite_zeros():
    from ralup.policy.features import collect_features
    from ralup.policy.history import MatrixHistory, LossObservation
    u=torch.zeros(2,4)
    f=collect_features(u,u,u,u,MatrixHistory(),context(u),LossObservation(None,9.),2.)
    torch.testing.assert_close(f.values,torch.tensor([-30.,-30.,-30.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,-1.]))
    assert f.coverage['nonzero_moves']==f.coverage['positive_distance']==0
    assert f.coverage['history_valid'] is False


def test_boundary_fraction_positive_distance_and_saturation():
    from ralup.policy.features import collect_features
    from ralup.policy.history import MatrixHistory, LossObservation
    u=torch.tensor([[.74,6.,-6.,.75]])
    d=torch.tensor([[.01,1.,-1.,-.01]])
    f=collect_features(u,d,d,d,MatrixHistory(),context(u),LossObservation(0.,100.),-.1)
    assert f.values[5].item()==pytest.approx(.5,abs=2e-6)
    assert f.values[6].item()==1 and f.values[7].item()==0
    assert f.values[11].item()==1 and f.values[12].item()==0
    assert f.coverage['positive_distance']==1  # midpoint distance0 excluded; endpoints outward excluded


def test_collect_is_prospective_and_does_not_mutate_loss_history():
    from ralup.policy.features import collect_features
    from ralup.policy.history import MatrixHistory, LossObservation
    u=torch.ones(1,2); h=MatrixHistory()
    for _ in range(2):
        f=collect_features(u,u,u,u,h,context(u),LossObservation(4.,0.),7/100)
        assert f.values[11].item()==pytest.approx(-.05)
        assert f.values[12].item()==pytest.approx(.07)
    assert h.updates==0


def test_frozen_normalizer_rejects_target_and_protected_before_loading_rows(tmp_path):
    from ralup.policy.normalizer import fit_normalizer, ProvenanceError
    for partition in ['target_train','development','meta_development','protected_test']:
        p=tmp_path/(partition+'.json');p.write_text(json.dumps({'partition':partition}))
        with pytest.raises(ProvenanceError):fit_normalizer([ArtifactRef(p,sha256_file(p))])


def test_normalizer_floor_clip_and_immutable_fit_provenance():
    from ralup.policy.normalizer import FrozenNormalizer
    n=FrozenNormalizer(torch.zeros(14),torch.ones(14), 'authenticated-meta-reference')
    v=torch.arange(14,dtype=torch.float32)
    result,info=n.normalize(v)
    assert result.tolist()==[0.,1.,2.,3.,4.,5.,5.,5.,5.,5.,5.,5.,5.,5.]
    assert info['clipped']==8 and info['total']==14
    assert n.mean.tolist()==[0.]*14 and n.std.tolist()==[1.]*14


def test_soap_feature_four_must_inverse_new_moment_basis():
    from ralup.optim.soap import SoapUpdateRule
    from ralup.optim.base import SchedulePoint
    from ralup.policy.features import collect_features,proposal_first_moment
    from ralup.policy.history import MatrixHistory,LossObservation
    rule=SoapUpdateRule({'precondition_frequency':1})
    w=torch.tensor([[-.3,0.,1.1],[.01,-2.,.7]])
    g=torch.tensor([[.2,-.7,.4],[.3,.9,-.1]])
    p=rule.propose(w,g,{},SchedulePoint(.03,0.,0.))
    p=rule.propose(w,g,p.new_optimizer_state,SchedulePoint(.03,0.,.1))
    m,_,source=proposal_first_moment(rule,p,g,MatrixHistory())
    torch.testing.assert_close(m,.05*g,atol=1e-7,rtol=1e-5)
    f=collect_features(w,p.gradient_displacement,g,m,MatrixHistory(),context(w),LossObservation(None,1.),.1)
    wrong=collect_features(w,p.gradient_displacement,g,p.new_optimizer_state['exp_avg'],MatrixHistory(),context(w),LossObservation(None,1.),.1)
    assert f.values[3].item()==pytest.approx(1.,abs=2e-6)
    assert abs(wrong.values[3].item()-1.)>.1  # wrong basis fails the hand cosine contract
    assert source=='adapter_inverse_basis_original_W'


def test_s19_common_gradient_ema_is_prospective_for_no_moment_backbone():
    from ralup.optim.base import Proposal
    from ralup.policy.features import proposal_first_moment
    from ralup.policy.history import MatrixHistory
    g=torch.tensor([[2.,4.]])
    p=Proposal(g,g,{})
    h=MatrixHistory(common_gradient_ema=torch.tensor([[1.,3.]]))
    m,pending,_=proposal_first_moment(object(),p,g,h)
    torch.testing.assert_close(m,torch.tensor([[1.05,3.05]]))
    torch.testing.assert_close(h.common_gradient_ema,torch.tensor([[1.,3.]]))
    assert pending is m


def normalization_fixture(tmp_path):
    def write(name,payload):
        p=tmp_path/name;p.write_text(json.dumps(payload));return {'path':str(p),'sha256':sha256_file(p)}
    corpus=write('corpus.json',{'synthetic':False,'partitions':{'meta-train':{},'target-training':{}}})
    parent=write('parent.json',{'source_partition':'meta_train','cursor':{'source':{'kind':'indexed','partition':'meta-train','sha256':corpus['sha256']}}})
    backbone=write('backbone.json',{'schema':'ralup.exploratory_backbone.v1','status':'EXPLORATORY','parents':{'meta_train':{'artifact':parent}}})
    producer=write('startup.json',{'source':{'src/example.py':{'sha256':'1'*64}}})
    payload={'schema':'ralup.feature_trajectory.v1','partition':'meta_train','role':'reference','synthetic':False,
             'source_identity':{'src/example.py':'1'*64},'data_manifest':corpus,'producer':producer,'backbone':backbone,
             'parent':parent,'continuation_horizon':10,'rows':[
                 {'step':0,'matrix':'x','values':[0.]*14},
                 {'step':1,'matrix':'x','values':[2.]*12+[.1,0.]}]}
    ref=write('trajectory.json',payload)
    return payload,ArtifactRef(**ref)


def test_meta_reference_population_fit_and_constant_std_floor(tmp_path):
    from ralup.policy.normalizer import fit_normalizer
    _,ref=normalization_fixture(tmp_path);n=fit_normalizer([ref])
    torch.testing.assert_close(n.mean,torch.tensor([1.]*12+[.05,0.]))
    torch.testing.assert_close(n.std,torch.tensor([1.]*12+[.05,1e-6]))
    assert ref.sha256 in n.source_manifest and str(ref.path) in n.source_manifest


@pytest.mark.parametrize('damage',['duplicate','parent_mislabel','data_mislabel','producer','progress'])
def test_fit_rejects_relabelled_or_duplicate_reference_evidence(tmp_path,damage):
    from ralup.policy.normalizer import fit_normalizer,ProvenanceError
    d,ref=normalization_fixture(tmp_path)
    if damage=='duplicate':
        with pytest.raises(ProvenanceError):fit_normalizer([ref,ref])
        return
    if damage=='producer':d['source_identity']['src/example.py']='2'*64
    elif damage=='progress':d['rows'][1]['values'][12]=1.
    else:
        key='parent' if damage=='parent_mislabel' else 'data_manifest'
        p=Path(d[key]['path']);value=json.loads(p.read_bytes())
        if key=='parent':value['cursor']['source']['partition']='target-training'
        else:value['synthetic']=True
        p.write_text(json.dumps(value));d[key]['sha256']=sha256_file(p)
    ref.path.write_text(json.dumps(d));ref=ArtifactRef(ref.path,sha256_file(ref.path))
    with pytest.raises(ProvenanceError):fit_normalizer([ref])


def test_transition_telemetry_distinguishes_actual_before_decay_and_base_additive():
    from ralup.policy.features import collect_features
    from ralup.policy.history import MatrixHistory,LossObservation
    before=torch.tensor([[1.]]);u0=torch.tensor([[.6]]);d=torch.tensor([[.7]])
    f=collect_features(u0,d,d,d,MatrixHistory(),context(before),LossObservation(None,1.),0.)
    assert f.coverage['decay_reference_transitions']==1
    assert f.coverage['base_additive_transitions']==1
    assert f.coverage['decay_endpoint_displacement']==.5
    assert f.coverage['base_additive_endpoint_displacement']==1.
