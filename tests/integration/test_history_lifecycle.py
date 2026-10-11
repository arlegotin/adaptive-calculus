"""Logical-step history, pending refresh and safe replay consumers."""
import pytest
import torch
from ralup.quantization.context import prepare_context


def ctx(u,step,scale=1.):
    return prepare_context(u,{'name':'row_fixed_max','previous_scales':torch.full((u.shape[0],1),scale)},step,None)


def test_history_reversal_only_matching_scales_and_single_commit():
    from ralup.policy.history import MatrixHistory, commit_history, resolve_realized_history
    a=torch.tensor([[.5,1.]]);b=torch.tensor([[1.,1.5]])
    h=commit_history(MatrixHistory(),a,b,ctx(a,0))
    assert h.transition_fraction==1 and h.pending and h.previous_codes.dtype==torch.uint8
    with pytest.raises(ValueError,match='commit|step'):commit_history(h,a,b,ctx(a,0))
    h=resolve_realized_history(h,ctx(b,1));h=commit_history(h,b,a,ctx(b,1))
    h=resolve_realized_history(h,ctx(a,2));h=commit_history(h,a,b,ctx(a,2))
    assert h.reversal_fraction==1 and h.reversal_coverage==1
    h=resolve_realized_history(h,ctx(b,3,2.));h=commit_history(h,b,a,ctx(b,3,2.))
    assert h.reversal_fraction==h.reversal_coverage==0
    assert h.numerical_displacement>0  # endpoints move even without comparable ordinals


def test_refresh_compares_same_post_step_weights_and_pending_survives_snapshot(tmp_path):
    from ralup.policy.history import MatrixHistory, commit_history, resolve_realized_history
    a=torch.tensor([[.5,1.]]);b=torch.tensor([[1.,2.]])
    h=commit_history(MatrixHistory(),a,b,ctx(a,0))
    h=resolve_realized_history(h,ctx(b,1,2.))
    assert h.disagreement_fraction==1 and h.realized_valid and not h.pending
    assert h.refresh_endpoint_displacement==0  # different codes, equal numerical values
    with pytest.raises(ValueError):resolve_realized_history(commit_history(MatrixHistory(),a,b,ctx(a,0)),ctx(b,2))


def test_feature_scans_use_new_original_w_moment_and_full_schedule_once(tmp_path):
    from test_resume import fixture
    from ralup.train import train_step, select_logical_batch
    from ralup.state import capture_snapshot,restore_snapshot,exact_cpu_state_hash
    s=fixture('cpu','row_fixed_max')
    s.configuration['policy_features']={'enabled':True,'full_horizon':100,'trace_cadence':1}
    s.schedule['step']=0
    before={n:id(p) for n,p in s.model.named_parameters()}
    result=train_step(s,select_logical_batch(s,5))
    assert s.histories['loss_updates']==1
    assert len(result.telemetry['features'])==6
    for f in result.telemetry['features'].values():
        assert f['values'][3]==pytest.approx(1.,abs=1e-6)
        assert f['values'][11]==f['values'][12]==0
    restored=restore_snapshot(capture_snapshot(s),'cpu')
    assert exact_cpu_state_hash(restored)==exact_cpu_state_hash(s)
    a=train_step(s,select_logical_batch(s,5));b=train_step(restored,select_logical_batch(restored,5))
    assert exact_cpu_state_hash(restored)==exact_cpu_state_hash(s)
    assert before=={n:id(p) for n,p in s.model.named_parameters()}
    assert all(f['values'][12]==pytest.approx(.01) for f in a.telemetry['features'].values())
    assert s.histories['loss_updates']==2


def test_evaluation_leaves_pending_history_and_feature_state_unchanged():
    from test_resume import fixture
    from ralup.train import train_step,select_logical_batch
    from ralup.state import exact_cpu_state_hash
    from ralup.evaluate import score_blocks
    s=fixture('cpu','row_fixed_max');s.configuration['policy_features']={'enabled':True,'full_horizon':10,'trace_cadence':1}
    batch=select_logical_batch(s,5);train_step(s,batch);before=exact_cpu_state_hash(s)
    score_blocks(s.model,batch.block_slices,evaluation_context={'partition':'fixture'})
    assert before==exact_cpu_state_hash(s)


def test_repeated_microbatch_history_commit_rolls_back_entire_update():
    from test_resume import fixture
    from ralup.train import train_step,select_logical_batch,TrainingFailure
    from ralup.policy.history import MatrixHistory,commit_history
    s=fixture('cpu','row_fixed_max');before={n:p.clone() for n,p in s.model.named_parameters()}
    ordinary=s.method.backward_microbatch
    def illegal(state,microbatch):
        result=ordinary(state,microbatch)
        for name,c in state.model._contexts.items():
            w=state.method.parameters[name]
            state.histories[name]=commit_history(state.histories.get(name,MatrixHistory()),w,w,c)
        return result
    s.method.backward_microbatch=illegal
    with pytest.raises(TrainingFailure,match='commit|history'):train_step(s,select_logical_batch(s,5))
    assert s.histories=={} and s.schedule['step']==0
    assert all(torch.equal(p,before[n]) for n,p in s.model.named_parameters())
    assert s.ledger.events[-1]['status']=='failed'


@pytest.mark.parametrize('field,value',[('pending',False),('updates',99),('last_step',99),('reversal_coverage',2.),('two_steps_ago_scales',None)])
def test_h07_rejects_malformed_or_omitted_history_fields_before_restore(field,value):
    from test_resume import fixture
    from ralup.train import train_step,select_logical_batch
    from ralup.state import capture_snapshot,IntegrityError
    from dataclasses import replace
    s=fixture('cpu','row_fixed_max')
    for _ in range(3):train_step(s,select_logical_batch(s,5))
    name=next(iter(s.model._contexts));s.histories[name]=replace(s.histories[name],**{field:value})
    with pytest.raises(IntegrityError):capture_snapshot(s)


def test_policy_cli_requires_typed_exploratory_purpose_and_frozen_actual_design(tmp_path,capsys):
    from ralup.__main__ import main
    import json
    p=tmp_path/'B0.json';p.write_text(json.dumps({'schema':'b0','status':'pass'}))
    design=tmp_path/'design.json';design.write_text('{}')
    with pytest.raises(SystemExit):
        main(['test-policy','--suite','features','--backbone',str(p),'--purpose','exploratory-development-bootstrap','--design',str(design),'--output',str(tmp_path/'result.json')])
    assert 'typed exploratory' in capsys.readouterr().err
    assert not (tmp_path/'result.json').exists()


def test_next_step_feature_ten_observes_completed_A_B_A_and_refresh_does_not_overwrite():
    from ralup.policy.history import MatrixHistory,LossObservation,commit_history,resolve_realized_history
    from ralup.policy.features import collect_features
    a=torch.tensor([[.5,1.]]);b=torch.tensor([[1.,1.5]])
    h=commit_history(MatrixHistory(),a,b,ctx(a,0))
    h=resolve_realized_history(h,ctx(b,1));h=commit_history(h,b,a,ctx(b,1))
    # Exactly two completed transitions suffice. Current refresh changes codes
    # but must never reinterpret/erase the previous cached-grid reversal.
    h=resolve_realized_history(h,ctx(a,2,2.))
    f=collect_features(a,torch.zeros_like(a),a,a,h,ctx(a,2,2.),LossObservation(1.,1.),.2)
    assert f.values[9].item()==1 and h.reversal_coverage==1
    assert h.updates==2 and h.realized_valid
    h=commit_history(h,a,b,ctx(a,2,2.))
    assert h.reversal_fraction==h.reversal_coverage==0
