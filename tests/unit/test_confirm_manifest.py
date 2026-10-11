"""Mutations here must deny immutable contracts, not merely change text."""
import copy
import hashlib
import json

import pytest

from ralup import evaluate
from ralup.config import canonical_bytes
from ralup.records import ArtifactRef


def test_required_frozen_confirmation_interfaces():
    assert callable(getattr(evaluate, 'freeze_confirmation', None))
    assert callable(getattr(evaluate, 'validate_confirmation', None))


def test_canonical_immutable_manifest(e03_factory):
    inputs, ref, inventory, _ = e03_factory()
    manifest = evaluate.validate_confirmation(ref, inventory)
    assert manifest.sha256 == ref.sha256 == hashlib.sha256(ref.path.read_bytes()).hexdigest()
    assert canonical_bytes(manifest.payload) == ref.path.read_bytes()
    with pytest.raises(TypeError): manifest.payload['endpoint']['target_bytes'] = 1
    assert manifest.payload['seeds'] == (21,22)
    with pytest.raises(FileExistsError): evaluate.freeze_confirmation(inputs)


@pytest.mark.parametrize('role', ['model','quantizer','backbone','policy','normalizer','b_star','learned_lr','analysis','data','C-config'])
def test_changed_dependency_denies(e03_factory, role):
    inputs, ref, inventory, _ = e03_factory()
    (ref.path.parent/(role+'.json')).write_bytes(b'changed bytes')
    with pytest.raises(evaluate.IntegrityError): evaluate.validate_confirmation(ref, inventory)


def test_changed_manifest_hash_denies(e03_factory):
    _, ref, inventory, _ = e03_factory()
    with pytest.raises(evaluate.IntegrityError):
        evaluate.validate_confirmation(ArtifactRef(ref.path, '0'*64), inventory)


@pytest.mark.parametrize('change', ['missing','duplicate','extra','endpoint','status','checkpoint','receipt','config','lineage'])
def test_complete_authenticated_terminal_inventory(e03_factory, change):
    _, ref, inventory, artifact = e03_factory()
    if change == 'missing': inventory.pop()
    elif change == 'duplicate': inventory.append(inventory[0])
    elif change == 'extra': inventory.append({'seed':999,'arm':'C','terminal':inventory[0]['terminal']})
    else:
        row = inventory[0]
        record=json.loads(open(row['terminal']['path']).read())
        if change == 'endpoint': record['endpoint']['target_bytes']=63
        elif change == 'status': record['status']='running'
        elif change == 'checkpoint': record['checkpoint']=None
        elif change == 'config': record['config_sha256']='0'*64
        elif change == 'lineage': record['lineage']['unit']='independent-synthetic-22'
        elif change == 'receipt': record['seed']=22
        row['terminal']=artifact('altered-terminal.json', record)
    with pytest.raises((evaluate.IntegrityError, evaluate.IncompleteConfirmation)):
        evaluate.validate_confirmation(ref, inventory)


def test_terminal_divergence_preserved_without_checkpoint(e03_factory):
    _, ref, inventory, artifact=e03_factory()
    row=inventory[-1]; record=json.loads(open(row['terminal']['path']).read())
    record.update(status='diverged',checkpoint=None,failure={'kind':'nonfinite','disposition':'substantive_failure'})
    row['terminal']=artifact('divergence.json',record)
    result=evaluate.validate_confirmation(ref,inventory)
    assert result.sha256 == ref.sha256
    assert record['checkpoint'] is None
    record['checkpoint']=artifact('invented-checkpoint.json', {'synthetic':True})
    row['terminal']=artifact('invented-divergence.json',record)
    with pytest.raises(evaluate.IntegrityError): evaluate.validate_confirmation(ref,inventory)


@pytest.mark.parametrize('change', ['nan','seed','n','timing','budget','rule','failure','science'])
def test_bad_freeze_inputs_never_publish(e03_factory, change):
    inputs, _, _, _=e03_factory(); inputs=copy.deepcopy(inputs)
    inputs['output']=str(__import__('pathlib').Path(inputs['output']).with_name('bad.json'))
    if change=='nan': inputs['endpoint']['target_bytes']=float('nan')
    elif change=='seed': inputs['seeds']=[21,21]
    elif change=='n': inputs['n']=3
    elif change=='timing': inputs['timing_order'][0]['arms']=['C','C','LR']
    elif change=='budget': inputs['endpoint']['target_bytes']=0
    elif change=='rule': inputs['bonferroni']['interval_confidence']=.95
    elif change=='failure': inputs['failure_protocol']['replacement']=True
    elif change=='science': inputs.update(scope='scientific-confirmation',schema='ralup.confirmation.v1',confirmation_eligible=True)
    with pytest.raises((ValueError,TypeError)): evaluate.freeze_confirmation(inputs)
    assert not __import__('pathlib').Path(inputs['output']).exists()


def test_caller_engineering_design_cannot_freeze_science(tmp_path):
    design=tmp_path/'engineering.json'
    design.write_bytes(canonical_bytes({'status':'engineering_fixed_design','engineering_n':9,
        'locked_n':9,'confirmation_eligible':True,'purpose':'synthetic-engineering'}))
    with pytest.raises(evaluate.IncompleteConfirmation,match='scientific'):
        evaluate.freeze_confirmation({'design':{'path':str(design),'sha256':hashlib.sha256(design.read_bytes()).hexdigest()},
                                      'output':str(tmp_path/'confirm_manifest.json')} )
    assert not (tmp_path/'confirm_manifest.json').exists()


@pytest.mark.parametrize('defect',['wrong-seed','wrong-arm','not-final','reused'])
def test_checkpoint_must_authenticate_exact_final_arm(e03_factory,defect):
    inputs,ref,inventory,artifact=e03_factory()
    row=inventory[0]; terminal=json.loads(open(row['terminal']['path']).read())
    checkpoint={'schema':'ralup.synthetic_final_checkpoint.v1','synthetic':True,'seed':21,'arm':'C',
        'endpoint':inputs['endpoint'],'config_sha256':inputs['resolved_configs']['C']['sha256'],
        'source_identity':inputs['source_identity'],'runtime_identity':inputs['runtime_identity'],'final':True}
    if defect=='wrong-seed': checkpoint['seed']=22
    elif defect=='wrong-arm': checkpoint['arm']='LR'
    elif defect=='not-final': checkpoint['final']=False
    if defect=='reused': terminal['checkpoint']=json.loads(open(inventory[-1]['terminal']['path']).read())['checkpoint']
    else: terminal['checkpoint']=artifact('wrong-final.json',checkpoint)
    row['terminal']=artifact('wrong-final-terminal.json',terminal)
    with pytest.raises(evaluate.IntegrityError): evaluate.validate_confirmation(ref,inventory)


def test_freeze_cli_refuses_forged_usable_scientific_design(tmp_path,capsys):
    from ralup.__main__ import main
    path=tmp_path/'design.json'
    path.write_bytes(canonical_bytes({'purpose':'scientific-development','status':'fixed_design',
        'locked_n':4,'confirmation_eligible':True}))
    with pytest.raises(SystemExit) as error:
        main(['plan-confirmation','--freeze','--design',str(path),'--output',str(tmp_path/'manifest.json')])
    assert error.value.code==2
    assert 'indexed original' in capsys.readouterr().err
    assert not (tmp_path/'manifest.json').exists()


def test_freeze_cli_names_actual_missing_authentic_evidence(tmp_path,capsys):
    from ralup.__main__ import main
    path='/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/e02-20261012/plan-confirmation-v1/confirmation-plan.json'
    with pytest.raises(SystemExit) as error:
        main(['plan-confirmation','--freeze','--design',path,'--output',str(tmp_path/'manifest.json')])
    assert error.value.code==2
    message=capsys.readouterr().err
    for evidence in ('WP04','B0','WP06','C','learned-LR','B*','four independent','locked n','checkpoint'):
        assert evidence in message
    assert not (tmp_path/'manifest.json').exists()
