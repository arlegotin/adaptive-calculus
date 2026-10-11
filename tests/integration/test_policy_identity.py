import importlib.util
import json
import pytest
import torch
from ralup.config import canonical_bytes
from ralup.records import ArtifactRef,sha256_file
from ralup.policy.normalizer import FrozenNormalizer
from ralup.train import train_step,select_logical_batch,TrainingFailure
from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash,IntegrityError
from ralup.optim.base import SchedulePoint
from test_resume import fixture


def normalizer(tmp_path):
    n=FrozenNormalizer(torch.arange(14).float(),torch.ones(14),'meta_train_fixture')
    p=tmp_path/'normalizer.json'
    if not p.exists():p.write_bytes(canonical_bytes({'mean':n.mean.tolist(),'std':n.std.tolist(),'source_manifest':n.source_manifest}))
    return n,ArtifactRef(p,sha256_file(p))


def policy(tmp_path,controller=None,*,rotation='identity',rule=None,family=None):
    assert importlib.util.find_spec('ralup.policy.adapter') is not None,'M09 policy adapter required'
    from ralup.policy.adapter import PolicyAdapter
    from ralup.policy.controllers import Affine45
    s=fixture('cpu','row_dynamic_max');s.model.configure_quantization({'name':'row_dynamic_max','rotation':rotation,'backward':'ste'})
    n,ref=normalizer(tmp_path)
    s.method=PolicyAdapter(rule or s.method.rule,controller or Affine45(),n,normalizer_ref=ref,family=family).bind(dict(s.model.named_parameters()),s.method.groups)
    s.optimizer_state=s.method.optimizer_states
    s.configuration['policy_features']={'enabled':True,'full_horizon':16,'trace_cadence':1}
    s.histories['feature_normalizer']=n
    return s


def test_initialized_full_trajectory_equals_authoritative_base_and_parameter_owners(tmp_path):
    from ralup.train import _equal,_rng_state,_restore_rng
    from ralup.optim.base import MethodAdapter
    from ralup.optim.muon import MuonUpdateRule
    from ralup.optim.matrix_adapter import MatrixMethodAdapter
    # Same instrumentation/normalizer on the reference; ordinary raw default is untouched.
    for rule in (None,MuonUpdateRule(dict(MuonUpdateRule.defaults))):
        base=fixture('cpu','row_dynamic_max');base.model.configure_quantization({'name':'row_dynamic_max','rotation':'identity','backward':'ste'})
        if rule is not None:base.method=MatrixMethodAdapter(dict(base.model.named_parameters()),base.method.groups,rule);base.optimizer_state=base.method.optimizer_states
        n,ref=normalizer(tmp_path);base.configuration['policy_features']={'enabled':True,'full_horizon':16,'trace_cadence':1};base.histories['feature_normalizer']=n
        s=policy(tmp_path,rule=rule)
        identities={name:id(p) for name,p in s.model.named_parameters()}
        for i in range(4):
            point=SchedulePoint(.013,.7,i/16);base.schedule['point']=s.schedule['point']=point
            rng=_rng_state();ra=train_step(base,select_logical_batch(base,5));_restore_rng(rng);rb=train_step(s,select_logical_batch(s,5))
            assert _equal(base.model.state_dict(),s.model.state_dict())
            assert _equal(base.optimizer_state,s.optimizer_state)
            assert _equal(base.histories,s.histories) and base.cursor.state==s.cursor.state
            assert _equal(base.quantizer_state,s.quantizer_state) and _equal(base.schedule,s.schedule)
            assert ra.work['feature_work']['matrix_scans']==rb.work['feature_work']['matrix_scans']==6
            assert identities=={name:id(p) for name,p in s.model.named_parameters()}
        restored=restore_snapshot(capture_snapshot(s),'cpu')
        assert exact_cpu_state_hash(restored)==exact_cpu_state_hash(s)
        rng=_rng_state();train_step(s,select_logical_batch(s,5));_restore_rng(rng);train_step(restored,select_logical_batch(restored,5))
        assert exact_cpu_state_hash(restored)==exact_cpu_state_hash(s)


@pytest.mark.parametrize('family',['affine45','learned_lr49','learned_lr49_shared_range','static3','additive_only','multiplicative_only','assistance_only','no_history','no_cells','norm_matched','initialized'])
def test_registered_families_require_full_features_selected_attached_normalizer(tmp_path,family):
    from ralup.policy.controllers import Affine45,LearnedLR49,Static3
    controller=LearnedLR49() if family.startswith('learned_lr') else Static3() if family=='static3' else Affine45()
    s=policy(tmp_path,controller,family=family)
    s.configuration.clear()
    with pytest.raises(TrainingFailure,match='feature'):train_step(s,select_logical_batch(s,5))
    s=policy(tmp_path,controller,family=family);s.histories.clear()
    with pytest.raises(TrainingFailure,match='normalizer'):train_step(s,select_logical_batch(s,5))
    s=policy(tmp_path,controller,family=family);s.histories['feature_normalizer'].mean.add_(1)
    with pytest.raises((IntegrityError,ValueError),match='normalizer|deployment'):capture_snapshot(s)


def test_changed_actions_original_moments_decay_and_exact_resume(tmp_path):
    from ralup.policy.controllers import Affine45
    from ralup.policy.actions import compose_action
    from ralup.quantization.rotation import to_storage,from_storage
    a=Affine45()
    with torch.no_grad():a.linear.bias.copy_(torch.tensor([.7,.2,.5]))
    s=policy(tmp_path,a,rotation='full_hadamard');ids={n:id(p) for n,p in s.model.named_parameters()}
    for _ in range(2):train_step(s,select_logical_batch(s,5))
    restored=restore_snapshot(capture_snapshot(s),'cpu')
    train_step(s,select_logical_batch(s,5));train_step(restored,select_logical_batch(restored,5))
    assert exact_cpu_state_hash(s)==exact_cpu_state_hash(restored)
    assert ids=={n:id(p) for n,p in s.model.named_parameters()}
    for info in s.method.last_actions.values():
        assert info['accepted_bound_excess_max']<=1e-6
    # With zero gradients, h cannot amplify decay. Cold Adam moments are zero.
    s=policy(tmp_path,a,rotation='identity');s.model.prepare_quantized_views(0,None)
    before={n:p.clone() for n,p in s.model.named_parameters()}
    for p in s.model.parameters():p.grad=torch.zeros_like(p)
    s._work={'training_loss':1.};s.method.finish_step(s,SchedulePoint(.1,.2,0.))
    for name in s.method.groups[0].parameter_names:assert torch.equal(s.method.parameters[name],before[name]*.98)
    assert all(torch.count_nonzero(v['exp_avg'])==0 for v in s.optimizer_state.values())


def test_mutated_deployment_or_executable_controller_rejected(tmp_path):
    from ralup.policy.controllers import Affine45,freeze_controller
    s=policy(tmp_path);s.method.controller_state.parameters[0]=1
    with pytest.raises((IntegrityError,ValueError),match='deployment'):capture_snapshot(s)
    s=policy(tmp_path);s.method.normalizer_ref.path.write_text('{}')
    with pytest.raises(TrainingFailure,match='provenance|normalizer'):train_step(s,select_logical_batch(s,5))
    a=Affine45();a.linear.register_forward_hook(lambda *args:None)
    with pytest.raises(ValueError,match='hooks'):freeze_controller(a)


def test_standardization_is_consumed_and_source_controller_stays_frozen(tmp_path):
    from ralup.policy.controllers import Affine45,FeatureVector,evaluate_controller
    a=Affine45()
    with torch.no_grad():a.linear.bias.copy_(torch.tensor([.5,0.,1.]));a.linear.weight[0,0]=.1
    s=policy(tmp_path,a)
    with torch.no_grad():a.linear.bias.fill_(999)
    result=train_step(s,select_logical_batch(s,5))
    for name,feature in result.telemetry['features'].items():
        expected=evaluate_controller(s.method.controller_state,FeatureVector(torch.tensor(feature['normalized'])))
        assert s.method.last_actions[name]['alpha']==expected.alpha
    assert s.method.controller_state.parameters[-3:].tolist()==[.5,0.,1.]


@pytest.mark.parametrize('mutant',['reorder','amplify_decay','replace_parameter'])
def test_identity_decay_owner_checks_detect_deliberately_bad_policy(tmp_path,monkeypatch,mutant):
    from ralup.policy.adapter import PolicyAdapter
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.optim.base import Proposal
    # Deliberate wrong arithmetic is applied only to the policy's matrix proposals.
    s=policy(tmp_path);base=fixture('cpu','row_dynamic_max');base.model.configure_quantization({'name':'row_dynamic_max','rotation':'identity','backward':'ste'})
    original=PolicyAdapter.finish_step
    def bad(self,state,schedule):
        result=original(self,state,schedule)
        if mutant=='replace_parameter':
            name=next(iter(self.parameters));parent,_,leaf=name.rpartition('.')
            module=state.model.get_submodule(parent)
            module._parameters[leaf]=torch.nn.Parameter(self.parameters[name].clone())
        elif mutant=='amplify_decay':
            with torch.no_grad():
                for n in self.groups[0].parameter_names:self.parameters[n].mul_(.98)
        return result
    if mutant=='reorder':
        original_commit=s.method.commit_proposals
        def bad_commit(proposals):
            # Commit telemetry W0+dW instead of the authoritative addcdiv operands.
            result=original_commit(proposals)
            with torch.no_grad():
                for n,p in proposals.items():self_parameter=s.method.parameters[n];self_parameter.copy_(p.reference_after_decay+p.gradient_displacement)
            return result
        monkeypatch.setattr(s.method,'commit_proposals',bad_commit)
    else:monkeypatch.setattr(PolicyAdapter,'finish_step',bad)
    train_step(base,select_logical_batch(base,5))
    if mutant=='replace_parameter':
        with pytest.raises(TrainingFailure,match='Parameter'):train_step(s,select_logical_batch(s,5))
    else:
        train_step(s,select_logical_batch(s,5))
        assert any(not torch.equal(p,dict(base.model.named_parameters())[n]) for n,p in s.model.named_parameters()),'Identity detector must reject mutant'


def test_controller_cli_keeps_missing_B0_closed_and_requires_indexed_protocol(tmp_path,capsys):
    from ralup.__main__ import main
    p=tmp_path/'B0.json';p.write_text('{"status":"pass"}')
    d=tmp_path/'design.json';d.write_text('{}')
    with pytest.raises(SystemExit):main(['test-policy','--suite','controllers','--backbone',str(p),'--purpose','exploratory-development-bootstrap','--design',str(d),'--device','cpu','--output',str(tmp_path/'result.json')])
    assert 'typed exploratory' in capsys.readouterr().err
    assert not (tmp_path/'result.json').exists()


def test_identity_projection_matches_shared_state_and_rejects_changed_actions(tmp_path):
    from ralup.policy.checks import identity_backbone_trajectory_hash
    from ralup.train import _rng_state,_restore_rng
    s=policy(tmp_path);base=fixture('cpu','row_dynamic_max');base.model.configure_quantization({'name':'row_dynamic_max','rotation':'identity','backward':'ste'})
    base.configuration=dict(s.configuration);base.histories['feature_normalizer']=s.histories['feature_normalizer']
    for i in range(2):
        r=_rng_state();train_step(base,select_logical_batch(base,5));_restore_rng(r);train_step(s,select_logical_batch(s,5))
        assert identity_backbone_trajectory_hash(base)==identity_backbone_trajectory_hash(s)
    name=next(iter(s.method.last_actions));s.method.last_actions[name]['h']=1.1
    with pytest.raises((ValueError,IntegrityError)):identity_backbone_trajectory_hash(s)
