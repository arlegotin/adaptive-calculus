"""Breaks caught: untrusted rule state, omitted factors/bounds, uncharged CPU transport."""
import copy,json,os,subprocess,sys,time
from pathlib import Path
import pytest
import torch
from ralup.optim.base import default_parameter_groups,SchedulePoint

NAMES=['soap_pinned_fp32','madam_clean_fp32']

def fixture(name,device='cpu'):
    from test_resume import fixture as ordinary
    from ralup.baselines.registry import build_matrix_method
    state=ordinary(device,'row_dynamic_max')
    state.method=build_matrix_method(name,dict(state.model.named_parameters()),default_parameter_groups(state.model),{'precondition_frequency':1} if name.startswith('soap') else {})
    state.optimizer_state=state.method.optimizer_states
    return state

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('device',['cpu','mps'])
def test_complete_rule_state_snapshot_duplicate_and_fresh_process_continuation(tmp_path,name,device):
    if device=='mps':
        if not torch.backends.mps.is_available():pytest.skip('Actual MPS unavailable')
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    from test_resume import advance
    from ralup.state import capture_snapshot,save_snapshot,load_snapshot,restore_snapshot,exact_cpu_state_hash
    dest=Path(os.environ.get('M02_ATTEMPT_DIR',tmp_path))/f'{name}-{device}'
    full=fixture(name,device);advance(full,4);expected=exact_cpu_state_hash(full)
    part=fixture(name,device);advance(part,2)
    inventory={}
    if name.startswith('soap'):
        for n in part.method.groups[0].parameter_names:inventory[n]=part.method.rule.state_inventory(part.optimizer_state[n])
        assert sum(row['factor_bytes'] for row in inventory.values())>0
        assert all(row['factorization_count']==4 for row in inventory.values())
        assert bool(sum(row['transfer_bytes'] for row in inventory.values()))==(device=='mps')
        work=part.ledger.events[-1]['optimizer_work']
        assert work['factorization_count']==12 and work['qr_calls']==6
        assert part.ledger.passes['backward']==10
    snap=capture_snapshot(part)
    forecast=4*(sum(t.numel()*t.element_size() for t in snap.tensors.values())+sum(map(len,snap.producer_files.values()))+len(json.dumps(snap.metadata)))
    assert forecast<8*1024**2
    ref=save_snapshot(snap,dest/'snapshots')
    restore_start=time.perf_counter();resumed=restore_snapshot(load_snapshot(ref),device);restore_seconds=time.perf_counter()-restore_start
    if name.startswith('soap'):
        for n in part.method.groups[0].parameter_names:
            for original,restored in zip(part.optimizer_state[n]['Q'],resumed.optimizer_state[n]['Q']):
                if isinstance(original,torch.Tensor):
                    assert original.stride()==restored.stride() and torch.equal(original,restored)
    past=resumed.ledger.elapsed['seconds'];advance(resumed,2)
    for label,value in [('full',full),('resumed',resumed)]:
        owned=capture_snapshot(value)
        from safetensors.torch import save_file
        dest.mkdir(parents=True,exist_ok=True)
        save_file(owned.tensors,str(dest/(label+'.safetensors')))
        (dest/(label+'.metadata.json')).write_text(json.dumps(owned.metadata,indent=2))
    assert resumed.ledger.elapsed['seconds']>past and exact_cpu_state_hash(resumed)==expected
    script='''import sys
sys.path.insert(0,'tests/integration')
from test_resume import advance
from ralup.state import restore_snapshot,load_snapshot,exact_cpu_state_hash
from ralup.records import ArtifactRef
s=restore_snapshot(load_snapshot(ArtifactRef(sys.argv[1],sys.argv[2])),sys.argv[3]);advance(s,2);print(exact_cpu_state_hash(s))
'''
    result=subprocess.run([sys.executable,'-c',script,str(ref.path),ref.sha256,device],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stderr=='' and result.stdout.strip()==expected
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'inventory.json').write_text(json.dumps({'restore_seconds':restore_seconds,'basis_restore_copy_bytes':sum(row['basis_bytes'] for row in inventory.values()),'snapshot_forecast_bytes':forecast,'snapshot_tensor_bytes':sum(t.numel()*t.element_size() for t in snap.tensors.values()),'inventory':inventory,'checkpoint':{'path':str(ref.path),'sha256':ref.sha256},'cost':{'stream':part.ledger.stream,'passes':part.ledger.passes,'elapsed':part.ledger.elapsed,'events':part.ledger.events},'resume_hash':expected},indent=2))

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('mutation',['unknown_rule','missing','geometry','metadata','layout'])
def test_snapshot_rejects_untrusted_or_incomplete_state_before_transfer(name,mutation):
    from test_resume import advance
    from ralup.state import capture_snapshot,_validated,_Encoder,IntegrityError
    state=fixture(name);advance(state,2);snap=capture_snapshot(state);payload=_validated(snap)
    n=state.method.groups[0].parameter_names[0];values=payload['method']['optimizer_states'][n]
    if mutation=='unknown_rule':payload['rule']['schema']='untrusted.source.Constructor'
    elif name.startswith('soap'):
        if mutation=='missing':values['Q'].pop()
        if mutation=='geometry':values['GG'][0]=torch.ones(1)
        if mutation=='metadata':values['precondition_frequency']=999
        if mutation=='layout':values['basis_strides'][0]=(values['Q'][0].shape[0],1)
    else:
        if mutation=='missing':values.pop('bound')
        if mutation=='geometry':values['bound']=torch.ones(1)
        if mutation=='metadata':values['step']=-1
        if mutation=='layout':values['basis_strides']=[(1,1)]
    encoder=_Encoder();snap.metadata=encoder.encode(payload);snap.tensors=encoder.tensors
    with pytest.raises(IntegrityError):_validated(snap)


def test_cpu_factorization_failure_cost_survives_transaction_rollback(monkeypatch):
    from test_resume import advance
    from ralup.train import TrainingFailure
    state=fixture('soap_pinned_fp32');before={n:p.detach().clone() for n,p in state.model.named_parameters()}
    def unsupported(*a,**k):raise RuntimeError('injected unavailable eig operation')
    monkeypatch.setattr(torch.linalg,'eigh',unsupported)
    with pytest.raises(TrainingFailure):advance(state,1)
    assert state.failed and state.ledger.events[-1]['status']=='failed'
    work=state.ledger.events[-1]['optimizer_work']
    assert work['eig_calls']==1 and work['factorization_count']==2 and work['factorization_seconds']>=0
    assert state.ledger.passes['backward']==5
    for n,p in state.model.named_parameters():assert torch.equal(p,before[n])


def test_registry_and_cli_distinguish_permitted_source_and_clean_equation_research(tmp_path,capsys):
    from ralup.__main__ import main
    from ralup.baselines.registry import default_registry
    from ralup.models.transformer import ByteTransformer,ModelConfig
    registry=default_registry(default_parameter_groups(ByteTransformer(ModelConfig(1,8,2))))
    soap=registry.require('soap_pinned_fp32',adaptation_label='pinned_SOAP_CPU_factorization_original_W')
    madam=registry.require('madam_clean_fp32',adaptation_label='clean_Madam_sign_restricted_original_W')
    assert 'a1e553530f' in soap.source_revision and soap.license_status=='permitted'
    assert madam.update_rule_settings['source_copy_status']=='blocked'
    assert madam.update_rule_settings['equation_implementation_status']=='independent_local_research'
    assert registry.claim_eligibility(madam.name)['blocked_is_defeated_baseline'] is False
    output=Path(os.environ.get('M02_ATTEMPT_DIR',tmp_path))/'cli-soap-madam.json'
    assert main(['verify-baselines','--suite','soap-madam','--output',str(output)])==0
    report=json.loads(output.read_bytes());capsys.readouterr()
    assert report['status']=='pass' and len(report['variants'])==2
    assert report['scientific_gate'] is None
