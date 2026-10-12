"""Numerically executed synthetic byte fixtures; official test is never read."""
import json
from pathlib import Path

import pytest


def test_runner_shared_warm_frozen_order_costs_and_duplicate_score(e04_factory):
    from ralup.runner import run_confirmation,score_confirmation,terminal_inventory
    from ralup.analysis.confirm import PairedResults,analyze_confirmation
    from ralup.records import read_artifact
    ref,manifest=e04_factory(n=4)
    refs=run_confirmation(manifest)
    assert len(refs)==12
    terminals=[read_artifact(r) for r in refs]
    assert [(r['seed'],r['arm']) for r in terminals]==[(r['seed'],a) for r in manifest.payload['timing_order'] for a in r['arms']]
    for seed in manifest.payload['seeds']:
        costs=[read_artifact(type(ref)(**r['costs'])) for r in terminals if r['seed']==seed]
        assert len({r['initial_state_sha256'] for r in costs})==1
        assert len({r['parent']['sha256'] for r in costs})==1
        assert all(r['stream_targets']==manifest.payload['endpoint']['target_bytes'] for r in costs)
    inventory=terminal_inventory(refs)
    first=score_confirmation(manifest,inventory)
    second=score_confirmation(manifest,inventory)
    assert first==second
    result=read_artifact(first)
    assert result['scored_targets']>0
    assert result['project_costs']['physical_events'] < sum(r['deployment_costs']['physical_events'] for r in result['outcomes'])
    analysis=analyze_confirmation(PairedResults.from_artifact(first),manifest)
    assert analysis['n']==4
    assert len(analysis['quality_cost_curves'])==12
    assert analysis['scientific_confirmation_established'] is False
    assert run_confirmation(manifest)==refs


def test_interruption_resume_preserves_originals_and_exact_state(e04_factory,monkeypatch):
    from ralup import runner
    from ralup.records import read_artifact
    ref,manifest=e04_factory(n=2,steps=3)
    original=runner._advance
    calls=0
    def interrupt(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==3: raise KeyboardInterrupt('synthetic interruption')
        return original(*args,**kwargs)
    monkeypatch.setattr(runner,'_advance',interrupt)
    with pytest.raises(KeyboardInterrupt): runner.run_confirmation(manifest)
    root=ref.path.parent/'runs'
    before={str(p):p.read_bytes() for p in root.rglob('*.json')}
    monkeypatch.setattr(runner,'_advance',original)
    refs=runner.run_confirmation(manifest)
    assert all(Path(p).read_bytes()==value for p,value in before.items())
    assert len(refs)==6
    assert list(root.rglob('interrupted-*.json'))
    states=[]
    for terminal_ref in refs:
        terminal=read_artifact(terminal_ref)
        costs=read_artifact(type(ref)(**terminal['costs']))
        state=read_artifact(type(ref)(**costs['final_state']))['state']
        assert state['step']==3
        assert state['cursor']==manifest.payload['endpoint']['target_bytes']
        states.append(state)
    assert any(read_artifact(type(ref)(**read_artifact(r)['costs']))['unknown_partial_work'] for r in refs)
    _,clean_manifest=e04_factory(n=2,steps=3)
    clean=[]
    for r in runner.run_confirmation(clean_manifest):
        cost=read_artifact(type(ref)(**read_artifact(r)['costs']))
        clean.append(read_artifact(type(ref)(**cost['final_state']))['state'])
    for resumed,uninterrupted in zip(states,clean):
        for key in ('weights','momentum','step','cursor','data_offset','named_streams','warm_updates'):
            assert resumed[key]==uninterrupted[key]


def test_actual_source_and_runtime_changes_deny_resume(e04_factory,monkeypatch):
    from ralup import runner,evaluate
    from ralup.records import IntegrityError
    _,manifest=e04_factory(n=2)
    runner.run_confirmation(manifest)
    monkeypatch.setattr(evaluate,'confirmation_source_identity',lambda:{'changed':'0'*64})
    with pytest.raises(IntegrityError,match='source'):runner.run_confirmation(manifest)


def test_training_unique_bytes_do_not_inflate_by_repeated_stream_exposures(e04_factory):
    from ralup import runner
    from ralup.analysis.cost import _totals
    from ralup.records import read_artifact,ArtifactRef
    _,manifest=e04_factory(n=2,steps=3)
    refs=runner.run_confirmation(manifest)
    cost=read_artifact(ArtifactRef(**read_artifact(refs[0])['costs']))
    totals=_totals([cost['run']])
    assert totals['known_total_targets']==224
    assert totals['unique_targets']==64


def test_actual_export_and_complete_mean_deployment_costs(e04_factory):
    from ralup import runner
    from ralup.records import read_artifact,ArtifactRef
    from ralup.analysis.confirm import PairedResults,analyze_confirmation
    _,manifest=e04_factory(n=4)
    refs=runner.run_confirmation(manifest)
    cost=read_artifact(ArtifactRef(**read_artifact(refs[0])['costs']))
    export=ArtifactRef(**cost['export'])
    assert cost['export_bytes']==export.path.stat().st_size
    assert len(read_artifact(export)['quantized_logits'])==256
    report=analyze_confirmation(PairedResults.from_artifact(runner.score_confirmation(manifest,runner.terminal_inventory(refs))),manifest)
    assert report['mean_deployment_costs']['C']['total_targets']==197
    assert report['mean_deployment_costs']['C']['n_planned']==4
    assert report['mean_deployment_costs']['C']['offline_search_meta_costs'] is None


def test_interrupted_scoring_retains_completed_blocks_and_cannot_retry(e04_factory,monkeypatch):
    from ralup import runner,evaluate
    from ralup.records import read_artifact,ProtectedAccessError
    ref,manifest=e04_factory(n=2)
    inventory=runner.terminal_inventory(runner.run_confirmation(manifest))
    original=evaluate.score_blocks
    def interrupted(*args,**kwargs):
        callback=kwargs.get('progress_callback')
        assert callback is not None
        def stop(progress):
            callback(progress)
            if progress['phase']=='block_completed':raise KeyboardInterrupt('after actual completed scoring block')
        return original(*args,**{**kwargs,'progress_callback':stop})
    monkeypatch.setattr(evaluate,'score_blocks',interrupted)
    with pytest.raises(KeyboardInterrupt):runner.score_confirmation(manifest,inventory)
    failure=next((ref.path.parent/'runs').glob('scoring-failed-*.json'))
    retained=json.loads(failure.read_bytes())
    assert retained['completed_partial_targets']==37
    assert retained['unknown_partial_work'] is True
    assert not (ref.path.parent/'runs'/'paired-results.json').exists()
    with pytest.raises(ProtectedAccessError):runner.score_confirmation(manifest,inventory)


def test_runner_invalidates_all_units_after_implementation_failure(e04_factory,monkeypatch):
    from ralup import runner,evaluate
    from ralup.records import ProtectedAccessError
    ref,manifest=e04_factory(n=2)
    original=runner._advance;calls=0
    def broken(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==3:raise RuntimeError('synthetic implementation bug')
        return original(*args,**kwargs)
    monkeypatch.setattr(runner,'_advance',broken)
    with pytest.raises(RuntimeError):runner.run_confirmation(manifest)
    amendments=[r for r in evaluate.confirmation_access_events(ref) if r['kind']=='amendment']
    assert amendments[0]['amendment']['affected_seeds']==[100,101]
    assert set(amendments[0]['amendment']['affected_arms'])=={'C','Bstar','LR'}
    monkeypatch.setattr(runner,'_advance',original)
    with pytest.raises(ProtectedAccessError):runner.run_confirmation(manifest)


def test_synthetic_cli_complete_analysis_and_idempotent_readback(e04_factory,capsys):
    from ralup.__main__ import main
    ref,_=e04_factory(n=2)
    assert main(['confirm','--manifest',str(ref.path)])==0
    score=json.loads(capsys.readouterr().out)
    assert score['scientific_confirmation_established'] is False
    assert main(['analyze-confirmation','--manifest',str(ref.path)])==0
    analysis=json.loads(capsys.readouterr().out)
    assert analysis['n']==2 and analysis['scientific_confirmation_established'] is False
    assert analysis['H1_H2_supported']==all(analysis['contrasts'][arm]['interval'][1]<0 for arm in ('Bstar','LR'))
    before=(ref.path.parent/'runs'/'confirmation-analysis.json').read_bytes()
    assert main(['analyze-confirmation','--manifest',str(ref.path)])==0
    assert (ref.path.parent/'runs'/'confirmation-analysis.json').read_bytes()==before


def test_missing_manifest_cli_authenticates_actual_e02_disposition(monkeypatch,tmp_path,capsys):
    from ralup.__main__ import main
    from ralup import evaluate
    from ralup.records import IncompleteConfirmation
    observed=[]
    def refusal(reference):
        observed.append(reference)
        raise IncompleteConfirmation('authenticated producer disposition probe')
    monkeypatch.setattr(evaluate,'_refuse_design',refusal)
    with pytest.raises(SystemExit) as error:
        main(['confirm','--manifest',str(tmp_path/'missing.json')])
    assert error.value.code==2
    assert len(observed)==1 and observed[0].path.name=='confirmation-plan.json'
    assert 'authenticated producer disposition probe' in capsys.readouterr().err


def test_forged_terminal_state_denied_before_protected_payload(e04_factory):
    from ralup import runner
    from ralup.records import read_artifact,ArtifactRef,write_immutable_json,IntegrityError
    _,manifest=e04_factory(n=2)
    refs=runner.run_confirmation(manifest)
    terminal=read_artifact(refs[0]);cost=read_artifact(ArtifactRef(**terminal['costs']))
    state=read_artifact(ArtifactRef(**cost['final_state']))
    state['config_sha256']='0'*64
    fake=write_immutable_json(refs[0].path.parent/'forged-state.json',state)
    cost['final_state']={'path':str(fake.path),'sha256':fake.sha256}
    fake_cost=write_immutable_json(refs[0].path.parent/'forged-cost.json',cost)
    terminal['costs']={'path':str(fake_cost.path),'sha256':fake_cost.sha256}
    fake_terminal=write_immutable_json(refs[0].path.parent/'forged-terminal.json',terminal)
    refs[0]=fake_terminal
    with pytest.raises(IntegrityError):runner.score_confirmation(manifest,runner.terminal_inventory(refs))


def test_divergence_has_no_checkpoint_replacement_or_all_run_claim(e04_factory):
    from ralup.runner import run_confirmation,score_confirmation,terminal_inventory
    from ralup.analysis.confirm import PairedResults,analyze_confirmation
    from ralup.records import read_artifact
    _,manifest=e04_factory(n=4,diverge=(100,'C'))
    refs=run_confirmation(manifest)
    failed=[read_artifact(r) for r in refs if read_artifact(r)['status']=='diverged']
    assert len(failed)==1 and failed[0]['checkpoint'] is None
    assert len(run_confirmation(manifest))==12
    result=score_confirmation(manifest,terminal_inventory(refs))
    report=analyze_confirmation(PairedResults.from_artifact(result),manifest)
    assert report['n']==4 and report['status']=='substantive_failure'
    assert not report['all_run_bpb_claim_established']


def test_invalidation_retains_all_affected_arms_and_denies_scoring(e04_factory):
    from ralup.runner import run_confirmation,score_confirmation,terminal_inventory,invalidate_confirmation
    from ralup.records import ProtectedAccessError,read_artifact
    _,manifest=e04_factory(n=2)
    refs=run_confirmation(manifest)
    originals={r.path:r.path.read_bytes() for r in refs}
    invalidation=invalidate_confirmation(manifest,refs,'synthetic optimizer bug',affected_seeds=[100])
    assert len(read_artifact(invalidation)['affected_terminals'])==3
    assert all(path.read_bytes()==content for path,content in originals.items())
    with pytest.raises(ProtectedAccessError): score_confirmation(manifest,terminal_inventory(refs))
    with pytest.raises(ProtectedAccessError): run_confirmation(manifest)


def test_scoring_refuses_incomplete_inventory_before_payload(e04_factory,monkeypatch):
    from ralup.runner import run_confirmation,score_confirmation,terminal_inventory
    from ralup.records import IncompleteConfirmation
    ref,manifest=e04_factory(n=2)
    refs=run_confirmation(manifest)
    protected=Path(manifest.payload['dataset']['payload']['path']); reads=[]
    original=Path.read_bytes
    def read(path):
        if path==protected: reads.append(path)
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    with pytest.raises(IncompleteConfirmation): score_confirmation(manifest,terminal_inventory(refs)[:-1])
    assert reads==[]


@pytest.mark.parametrize('command',['confirm','analyze-confirmation'])
def test_actual_cli_refuses_absent_authorities(command,tmp_path,capsys):
    from ralup.__main__ import main
    with pytest.raises(SystemExit) as error:
        main([command,'--manifest',str(tmp_path/'confirm_manifest.json')])
    assert error.value.code==2
    message=capsys.readouterr().err
    for missing in ('WP04','B0','WP06','B*','C/LR','paired pilot','locked n'):
        assert missing in message


@pytest.mark.parametrize('kind',['success','null','inconclusive'])
def test_executed_synthetic_success_null_inconclusive(e04_factory,kind):
    from ralup.runner import run_confirmation,score_confirmation,terminal_inventory
    from ralup.analysis.confirm import PairedResults,analyze_confirmation
    _,manifest=e04_factory(kind=kind)
    result=score_confirmation(manifest,terminal_inventory(run_confirmation(manifest)))
    report=analyze_confirmation(PairedResults.from_artifact(result),manifest)
    assert report['status']=={'success':'supported','null':'negative','inconclusive':'inconclusive'}[kind]
    assert report['scientific_confirmation_established'] is False
