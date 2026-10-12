"""E04 review regressions: completed warm recovery and actual publication costs."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('boundary',['warm-state.json','warm-cost.json','partial-state'])
def test_completed_warm_persistence_interruption_never_reexecutes(e04_factory,monkeypatch,boundary):
    from ralup import runner
    from ralup.records import ArtifactRef,IntegrityError,read_artifact
    from ralup.analysis.cost import _totals
    ref,manifest=e04_factory(n=2,steps=1)
    put=runner._put;advance=runner._advance;warm_calls=[];interrupted=False
    def counted(state,config,count):
        if state['warm_updates']==0:warm_calls.append(state['seed'])
        return advance(state,config,count)
    def stop(path,value):
        nonlocal interrupted
        if not interrupted and path.name==(boundary if boundary!='partial-state' else 'warm-state.json'):
            interrupted=True
            if boundary=='partial-state':path.write_bytes(b'{')
            else:put(path,value)
            raise KeyboardInterrupt('completed warm persistence interruption')
        return put(path,value)
    monkeypatch.setattr(runner,'_advance',counted);monkeypatch.setattr(runner,'_put',stop)
    with pytest.raises(KeyboardInterrupt):runner.run_confirmation(manifest)
    root=ref.path.parent/'runs';originals={p:p.read_bytes() for p in root.rglob('*.json')}
    retained=list((root/'seed-100').glob('interrupted-*.json'))
    assert len(retained)==1
    evidence=json.loads(retained[0].read_bytes())
    assert evidence['completed_warm']['state']['warm_updates']==1
    assert len(evidence['run']['events'])==1
    monkeypatch.setattr(runner,'_put',put)
    if boundary=='partial-state':
        with pytest.raises(IntegrityError):runner.run_confirmation(manifest)
    else:
        refs=runner.run_confirmation(manifest)
        cost=read_artifact(ArtifactRef(**read_artifact(refs[0])['costs']))
        assert _totals([cost['run']])['known_total_targets']==96
        assert len([e for run in _walk(cost['run']) for e in run['events'] if 'warm_start' in e['classifications']])>=1
    assert warm_calls.count(100)==1
    assert all(path.read_bytes()==value for path,value in originals.items())


def _walk(run):
    yield run
    for parent in run['parents']:yield from _walk(parent)


def test_slow_required_publication_enters_attributed_union_without_complete_claim(e04_factory,monkeypatch):
    from ralup import runner,evaluate
    from ralup.analysis.confirm import PairedResults,analyze_confirmation
    _,manifest=e04_factory(n=2,steps=1)
    clock=[1000.];put=runner._put;write=runner.write_immutable_json;audit=evaluate.record_protected_result
    monkeypatch.setattr(runner,'time',SimpleNamespace(time=lambda:clock[0]))
    def slow_put(path,value):
        clock[0]+={'warm-cost.json':5.,'costs.json':7.,'terminal.json':11.,'terminal-publication.json':13.}.get(path.name,0.)
        return put(path,value)
    def slow_write(path,value):
        clock[0]+={'paired-results.json':17.,'scoring-publication.json':23.}.get(Path(path).name,0.)
        return write(path,value)
    def slow_audit(*args,**kwargs):
        clock[0]+=19.
        return audit(*args,**kwargs)
    monkeypatch.setattr(runner,'_put',slow_put);monkeypatch.setattr(runner,'write_immutable_json',slow_write)
    monkeypatch.setattr(evaluate,'record_protected_result',slow_audit)
    reference=runner.score_confirmation(manifest,runner.terminal_inventory(runner.run_confirmation(manifest)))
    result=PairedResults.from_artifact(reference)
    assert result.project_costs['known_interval_seconds']==154.
    assert result.project_costs['seconds'] is None
    for row in result.outcomes:
        assert row['deployment_costs']['known_interval_seconds']==59.
        assert row['deployment_costs']['seconds'] is None
    report=analyze_confirmation(result,manifest)
    assert all(not cost['complete_full_budget_mean_available'] for cost in report['mean_deployment_costs'].values())
    assert all(cost['metric_complete']['total_targets'] for cost in report['mean_deployment_costs'].values())
    assert runner.score_confirmation(manifest,runner.terminal_inventory(runner.run_confirmation(manifest)))==reference


@pytest.mark.parametrize('missing',['total_targets','unique_targets','seconds'])
def test_completed_status_with_unknown_metric_cannot_claim_complete_cost(e04_factory,missing):
    from ralup.analysis.confirm import PairedResults,analyze_confirmation
    _,manifest=e04_factory(n=2)
    costs={'total_targets':133,'unique_targets':101,'stream_targets':96,'seconds':1.,'physical_events':3,
           'known_total_targets':133,'known_interval_seconds':1.}
    costs[missing]=None
    rows=[{'seed':seed,'arm':arm,'unit':f'fresh-{seed}','status':'completed','bpb':7.,'deployment_costs':costs}
          for seed in manifest.payload['seeds'] for arm in manifest.payload['arms']]
    report=analyze_confirmation(PairedResults(manifest.sha256,rows),manifest)
    assert all(not cost['complete_full_budget_mean_available'] for cost in report['mean_deployment_costs'].values())


def test_finite_extreme_logits_saturate_before_division():
    from ralup.runner import _quantize
    assert _quantize([1e308,-1e308,.2])==pytest.approx([.7,-.8,.2])
    for invalid in (float('nan'),float('inf'),float('-inf')):
        with pytest.raises(ValueError,match='finite'):_quantize([invalid])
