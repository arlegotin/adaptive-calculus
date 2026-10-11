"""Only synthetic protected fixtures; actual official-test bytes are never opened."""
import json
from pathlib import Path

import pytest
from ralup import evaluate


@pytest.mark.parametrize('denial',['phase','analysis','missing','source','runtime','policy','data','config','endpoint'])
def test_denial_precedes_any_payload_read(e03_factory,monkeypatch,denial):
    inputs,ref,inventory,artifact=e03_factory()
    payload=Path(inputs['dataset']['payload']['path']); reads=[]
    original=Path.read_bytes
    def observed(path):
        if path == payload: reads.append(path)
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',observed)
    phase='confirm'
    if denial=='phase': phase='explore'
    elif denial=='missing': inventory.pop()
    elif denial=='source': monkeypatch.setattr(evaluate,'confirmation_source_identity',lambda: {'changed':'0'*64})
    elif denial=='runtime': monkeypatch.setattr(evaluate,'confirmation_runtime_identity',lambda: {'changed':True})
    elif denial in ('analysis','policy','data','config'):
        role={'data':'data','config':'C-config'}.get(denial,denial)
        (ref.path.parent/(role+'.json')).write_bytes(b'changed')
    elif denial=='endpoint':
        row=inventory[-1]; record=json.loads(Path(row['terminal']['path']).read_bytes())
        record['endpoint']['target_bytes']=1; row['terminal']=artifact('wrong-endpoint.json',record)
    with pytest.raises((evaluate.ProtectedAccessError,evaluate.IntegrityError,evaluate.IncompleteConfirmation)):
        evaluate.open_protected_targets(phase=phase,manifest_ref=ref,inventory=inventory)
    assert reads == []


def test_single_synthetic_open_and_append_only_result(e03_factory):
    inputs,ref,inventory,artifact=e03_factory()
    blocks=list(evaluate.open_protected_targets(phase='confirm',manifest_ref=ref,inventory=inventory))
    assert sum(sum(block.score_mask) for block in blocks)==33
    before=evaluate.confirmation_access_events(ref)
    assert [row['kind'] for row in before]==['frozen','opened']
    with pytest.raises(evaluate.ProtectedAccessError,match='already'):
        evaluate.open_protected_targets(phase='confirm',manifest_ref=ref,inventory=inventory)
    result=artifact('scores.json',{'synthetic':True,'manifest_sha256':ref.sha256,'bpb':8.0})
    evaluate.record_protected_result(ref,result)
    assert evaluate.existing_protected_result(ref).sha256==result['sha256']
    with pytest.raises(evaluate.ProtectedAccessError): evaluate.record_protected_result(ref,result)
    after=evaluate.confirmation_access_events(ref)
    assert after[:2]==before
    Path(result['path']).write_bytes(b'changed result')
    with pytest.raises(evaluate.IntegrityError): evaluate.existing_protected_result(ref)


def test_post_inspection_amendment_marks_claim_exploratory(e03_factory):
    _,ref,inventory,_=e03_factory()
    evaluate.open_protected_targets(phase='confirm',manifest_ref=ref,inventory=inventory)
    event=evaluate.record_confirmation_amendment(ref,{'claim':'revised claim'})
    assert event['claim_status']=='exploratory'
    with pytest.raises(evaluate.ProtectedAccessError): evaluate.open_protected_targets(phase='confirm',manifest_ref=ref,inventory=inventory)


def test_audit_tampering_and_missing_history_deny(e03_factory):
    _,ref,inventory,_=e03_factory()
    path=ref.path.with_suffix('.access.jsonl')
    path.write_bytes(b'{}\n')
    with pytest.raises(evaluate.IntegrityError): evaluate.open_protected_targets(phase='confirm',manifest_ref=ref,inventory=inventory)


def test_training_scoring_context_still_rejects_protected_partition():
    with pytest.raises(ValueError,match='protected'):
        evaluate.evaluation_contexts(None,{'partition':'test'})


def test_payload_hash_failure_retains_failed_single_open(e03_factory):
    inputs,ref,inventory,_=e03_factory()
    Path(inputs['dataset']['payload']['path']).write_bytes(b'changed protected fixture')
    with pytest.raises(evaluate.IntegrityError): evaluate.open_protected_targets(phase='confirm',manifest_ref=ref,inventory=inventory)
    assert [e['kind'] for e in evaluate.confirmation_access_events(ref)]==['frozen','opened','payload_failed']
    with pytest.raises(evaluate.ProtectedAccessError): evaluate.open_protected_targets(phase='confirm',manifest_ref=ref,inventory=inventory)


def test_preinspection_amendment_requires_new_freeze(e03_factory):
    _,ref,inventory,_=e03_factory()
    event=evaluate.record_confirmation_amendment(ref,{'endpoint':'new exploratory endpoint'})
    assert event['claim_status']=='requires_new_freeze'
    with pytest.raises(evaluate.ProtectedAccessError): evaluate.open_protected_targets(phase='confirm',manifest_ref=ref,inventory=inventory)


def test_result_without_open_denies(e03_factory):
    _,ref,_,artifact=e03_factory()
    result=artifact('unopened-result.json',{'manifest_sha256':ref.sha256,'synthetic':True})
    with pytest.raises(evaluate.ProtectedAccessError): evaluate.record_protected_result(ref,result)
    assert evaluate.existing_protected_result(ref) is None


def test_amendment_classifies_inspection_that_occurs_between_read_and_append(e03_factory,monkeypatch):
    from ralup.records import ConfirmationAudit
    _,ref,inventory,_=e03_factory()
    original=ConfirmationAudit.events
    entered=False
    def interleaved_open(audit):
        nonlocal entered
        rows=original(audit)
        if not entered:
            entered=True
            # Actual gateway reserves/opens the synthetic fixture after this read.
            evaluate.open_protected_targets(phase='confirm',manifest_ref=ref,inventory=inventory)
        return rows
    monkeypatch.setattr(ConfirmationAudit,'events',interleaved_open)
    event=evaluate.record_confirmation_amendment(ref,{'claim':'changed after concurrent inspection'})
    assert event['claim_status']=='exploratory'
    assert event['fresh_independent_confirmation_required'] is True
