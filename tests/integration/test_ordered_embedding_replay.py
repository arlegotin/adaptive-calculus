"""N01 authenticated legacy parent and shared scientific-owner replay."""
import hashlib
import os
import pytest
import torch


def test_native_parent_model_only_forward_contract_and_strict_resume(tmp_path,monkeypatch):
    import ralup.state as snapshots
    from ralup.calibrate import import_warmstart
    from test_warmstart_isolation import fixture
    from ralup.records import ArtifactRef
    parent = fixture()
    refs,files = snapshots._archive_producer()
    key='src/ralup/models/transformer.py'
    files[key] = files[key].replace(b'\nfrom .embedding import OrderedEmbedding',b'').replace(b'OrderedEmbedding(257,',b'nn.Embedding(257,').replace(b'OrderedEmbedding(256,',b'nn.Embedding(256,')
    refs[key]=ArtifactRef(refs[key].path,hashlib.sha256(files[key]).hexdigest())
    del files['src/ralup/models/embedding.py']; del refs['src/ralup/models/embedding.py']
    with monkeypatch.context() as patch:
        patch.setattr(snapshots,'_archive_producer',lambda:(refs,files))
        ref=snapshots.save_snapshot(snapshots.capture_snapshot(parent),tmp_path/'native')
    with pytest.raises(snapshots.IntegrityError,match='incompatible'):
        snapshots.restore_snapshot(snapshots.load_snapshot(ref),'cpu')
    fresh=import_warmstart(ref,fixture(),partition='meta_train',continuation_targets=7)
    inputs=torch.tensor([[256,0,1]]); positions=torch.tensor([[0,1,2]])
    native=fixture().model
    native.embedding=torch.nn.Embedding.from_pretrained(parent.model.embedding.weight.detach().clone(),freeze=False)
    native.position=torch.nn.Embedding.from_pretrained(parent.model.position.weight.detach().clone(),freeze=False)
    assert torch.equal(native(inputs,positions,contexts={}),fresh.model(inputs,positions,contexts={}))
    assert fresh.cursor.state == {**parent.cursor.state,'run_target_budget':7}
    assert fresh.ledger.passes == parent.ledger.passes
    assert torch.equal(fresh.rng_streams['quantizer'].get_state(),parent.rng_streams['quantizer'].get_state())
    # A producer with any additional forward change is rejected.
    files[key] += b'\n# undeclared model change\n'
    refs[key]=ArtifactRef(refs[key].path,hashlib.sha256(files[key]).hexdigest())
    with monkeypatch.context() as patch:
        patch.setattr(snapshots,'_archive_producer',lambda:(refs,files))
        bad=snapshots.save_snapshot(snapshots.capture_snapshot(parent),tmp_path/'bad')
    with pytest.raises(snapshots.IntegrityError,match='producer'):
        import_warmstart(bad,fixture(),partition='meta_train',continuation_targets=7)


@pytest.mark.parametrize('recipe',['row_fixed_max','row_dynamic_max','quest_fp4_row_reference'])
def test_cpu_duplicate_and_fresh_interpreter_strict_resume(tmp_path,recipe):
    import subprocess,sys
    from test_resume import fixture,advance
    from ralup.state import capture_snapshot,save_snapshot,exact_cpu_state_hash
    first=fixture('cpu',recipe);advance(first,4)
    expected=exact_cpu_state_hash(first)
    duplicate=fixture('cpu',recipe);advance(duplicate,4)
    assert expected==exact_cpu_state_hash(duplicate)
    interrupted=fixture('cpu',recipe);advance(interrupted,2)
    ref=save_snapshot(capture_snapshot(interrupted),tmp_path/'resume')
    code='''import sys
sys.path.insert(0,"tests/integration")
from test_resume import advance
from ralup.state import restore_snapshot,load_snapshot,exact_cpu_state_hash
from ralup.records import ArtifactRef
state=restore_snapshot(load_snapshot(ArtifactRef(sys.argv[1],sys.argv[2])),"cpu")
advance(state,2)
print(exact_cpu_state_hash(state))
'''
    result=subprocess.run([sys.executable,'-c',code,str(ref.path),ref.sha256],capture_output=True,text=True,check=True)
    assert result.stdout.strip()==expected
    assert result.stderr==''
