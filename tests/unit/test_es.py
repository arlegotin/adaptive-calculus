"""S21 estimator scaling/sign and finite fail-closed inputs."""
import importlib.util
import pytest
import torch


def api():
    assert importlib.util.find_spec('ralup.meta') is not None, 'M11 meta package required'
    from ralup.meta.es import estimate_es
    return estimate_es


def test_quadratic_complete_basis_matches_analytic_gradient():
    estimate_es=api()
    phi=torch.tensor([.2,-.4,.9],dtype=torch.float64)
    h=torch.tensor([[2.,.2,0.],[.2,3.,.3],[0.,.3,4.]],dtype=torch.float64)
    b=torch.tensor([1.,-2.,.4],dtype=torch.float64)
    e=torch.cat([torch.eye(3),-torch.eye(3)]).double()*3**.5
    f=lambda x:.5*torch.einsum('bi,ij,bj->b',x,h,x)+x@b
    got=estimate_es(phi,e,f(phi+.05*e),f(phi-.05*e),.05)
    torch.testing.assert_close(got,h@phi+b,rtol=1e-12,atol=1e-12)
    assert not got.requires_grad


@pytest.mark.parametrize('sigma',[0.,-1.,float('nan')])
def test_invalid_sigma_rejected(sigma):
    with pytest.raises(ValueError):api()(torch.zeros(3),torch.ones(2,3),torch.ones(2),torch.zeros(2),sigma)


def test_outer_adam_uses_descent_and_no_decay():
    api()
    from ralup.meta.es import OuterAdam
    opt=OuterAdam(torch.tensor([2.,-3.]))
    opt.step(torch.tensor([1.,0.]))
    torch.testing.assert_close(opt.phi,torch.tensor([1.99,-3.]))
    resumed=OuterAdam.from_state_dict(opt.state_dict())
    opt.step(torch.tensor([-.5,2.]));resumed.step(torch.tensor([-.5,2.]))
    assert torch.equal(opt.phi,resumed.phi)


def test_outer_adam_rejects_overflow_without_corrupting_checkpoint():
    from ralup.meta.es import OuterAdam
    opt=OuterAdam(torch.tensor([1.]))
    before=opt.state_dict()
    with pytest.raises(ValueError,match='finite'):
        opt.step(torch.tensor([3e38]))
    assert opt.state_dict()==before


def test_selection_requires_two_disjoint_stages_and_full_continuations(tmp_path):
    api()
    from ralup.meta import selection
    assert hasattr(selection,'select_controller'),'M11 two-stage selection required'
    with pytest.raises(ValueError,match='stages'):selection.select_controller([],[])
    with pytest.raises(ValueError,match='partition'):
        selection.select_controller([{'partition':'test'}],[{'partition':'development'}])


def test_meta_cli_rejects_unresolved_scientific_template(capsys):
    from ralup.__main__ import main
    with pytest.raises(SystemExit):main(['meta-train','--config','configs/meta.json'])
    assert 'unresolved' in capsys.readouterr().err


def test_real_query_authentication_uses_source_namespaced_document_ids(tmp_path):
    from types import SimpleNamespace
    from dataclasses import replace
    from ralup.meta import tasks
    from ralup.data.blocks import make_block
    from ralup.data.cursor import BatchCursor
    assert hasattr(tasks,'authenticate_queries'),'Explicit authenticated-query boundary required'
    raw=b'authentic query bytes';path=tmp_path/'content';path.write_bytes(raw)
    source='a'*64;article=__import__('hashlib').sha256(raw).hexdigest()
    doc={'id':'train:17','content_sha256':article,'content_path':str(path),'content_start':0,'bytes':len(raw)}
    cursor=SimpleNamespace(_source={'kind':'indexed','partition':'meta-train','hashes':{'source_sha256':source}},_documents=[doc],
      _stat_key=BatchCursor._stat_key,_content_stats={str(path):BatchCursor._stat_key(path.stat())})
    block=make_block(article,0,len(raw),b'',raw,document_id=source+':train:17')
    tasks.authenticate_queries(cursor,[block])
    with pytest.raises(ValueError,match='source'):tasks.authenticate_queries(cursor,[replace(block,labels=tuple([1]*len(block.labels)))])
    with pytest.raises(ValueError,match='documents'):tasks.authenticate_queries(cursor,[replace(block,document_id='train:17')])


def test_two_stage_selection_uses_meta_nomination_then_full_official_loss(tmp_path):
    from ralup.meta.selection import select_controller
    from ralup.policy.controllers import ControllerArtifact,freeze_controller,Affine45,Static3
    from ralup.records import ArtifactRef,sha256_file
    p=tmp_path/'provenance.json';p.write_text('{}');ref=ArtifactRef(p,sha256_file(p))
    def artifact(controller):
        state=freeze_controller(controller)
        return ControllerArtifact(state.family,state.parameters,ref,ref,ref,ref)
    a=artifact(Affine45());other=Affine45()
    with torch.no_grad():other.linear.bias[1]=.1
    b=artifact(other);c=artifact(Static3())
    def row(artifact,partition,loss):
        return dict(partition=partition,controller=artifact,seed=19,terminal_loss=loss,elapsed_seconds=4.,horizon=128,full_horizon=128,input_sha256='same-endpoint',opportunity_sha256='equal-budget')
    meta=[row(a,'meta-development',2.),row(b,'meta-development',1.),row(c,'meta-development',1.5)]
    official=[row(b,'development',1.4),row(c,'development',1.2)]
    assert select_controller(meta,official) is c
    with pytest.raises(ValueError,match='nominees'):select_controller(meta,[row(a,'development',.1),row(c,'development',1.2)])
    wrong=[dict(official[0],horizon=16),official[1]]
    with pytest.raises(ValueError,match='Full continuation'):select_controller(meta,wrong)
    with pytest.raises(ValueError,match='Unequal'):select_controller(meta,[dict(official[0],seed=20),official[1]])
