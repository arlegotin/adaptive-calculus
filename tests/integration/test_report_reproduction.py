"""Engineering controls: report completeness never upgrades scientific authority."""
import copy
import importlib
import json
from pathlib import Path

import pytest

from ralup.config import canonical_bytes
from ralup.records import ArtifactRef, sha256_file


def api():
    assert importlib.util.find_spec('ralup.report'), 'E07 report implementation missing'
    return importlib.import_module('ralup.report')


def put(path, value):
    path.write_bytes(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    return ArtifactRef(path, sha256_file(path))


def fixture(tmp_path):
    source=put(tmp_path/'source.json', {'original':True})
    rows=[]
    for i,status in enumerate(['completed','diverged']):
        rows.append({'run_id':str(i),'status':status,'seed':i,'provenance':'simulated',
            'source':{'path':str(source.path),'sha256':source.sha256},
            'development_bpb':1.2 if i==0 else None,
            'cost':{'stream_targets':10,'loss_targets':14,'gradient_only_targets':0,'hvp_targets':0,
                'warm_start_seconds':1.,'deployment_seconds':3.,'tuning_seconds':2.,'project_seconds':None}})
    trials=put(tmp_path/'trials.json',rows)
    return {'schema':'ralup.research_evidence.v1','scope':'synthetic-engineering',
        'registry':['0','1'],'trials':{'path':str(trials.path),'sha256':trials.sha256},
        'sources':[{'path':str(source.path),'sha256':source.sha256}],
        'scientific_status':{'wp04':'inconclusive','wp06':'NEEDS_REVISION','confirmation':None},
        'sections':{k:[] for k in api().SECTIONS},'precision':[], 'claim':{'outcome':'inconclusive'}}


def test_raw_tables_repeat_exactly_and_failures_remain(tmp_path):
    a=api(); evidence=put(tmp_path/'evidence.json',fixture(tmp_path))
    first=a.build_report([evidence],tmp_path/'first');second=a.build_report([evidence],tmp_path/'second')
    assert first.path.read_bytes()==second.path.read_bytes()
    one=json.loads(first.path.read_bytes())
    assert one['tables']['failures']==['1']
    assert one['disposition']['claim_status']=='development_only'


@pytest.mark.parametrize('corruption',['drop_failure','source_hash','missing_cost','stream_only'])
def test_incomplete_evidence_rejected(tmp_path,corruption):
    a=api();e=fixture(tmp_path);p=Path(e['trials']['path']);rows=json.loads(p.read_bytes())
    if corruption=='drop_failure':rows.pop()
    elif corruption=='source_hash':rows[0]['source']['sha256']='0'*64
    elif corruption=='missing_cost':del rows[0]['cost']['warm_start_seconds']
    else:rows[0]['cost']={'stream_targets':10}
    ref=put(p,rows);e['trials']['sha256']=ref.sha256
    with pytest.raises(ValueError,match='incomplete evidence'):
        a.build_report([put(tmp_path/'evidence.json',e)],tmp_path/'report')


@pytest.mark.parametrize('outcome,decision',[('negative','local_non_advantage'),('inconclusive','inconclusive'),('blocked','blocked'),('positive','positive_quality_but_slower')])
def test_distinct_dispositions(outcome,decision):
    r=api().research_disposition({'outcome':outcome,'quality_comparable':True,
        'baseline_cost':10,'candidate_cost':12,'meta_cost':50,'method_tuning_cost':20})
    assert r['decision']==decision
    assert r['efficiency_improvement'] is False
    assert r['break_even_reuses'] is None


def test_break_even_requires_comparable_quality_and_full_tuning():
    a=api();e={'outcome':'positive','quality_comparable':True,'baseline_cost':10,'candidate_cost':8,'meta_cost':50,'method_tuning_cost':20}
    r=a.research_disposition(e)
    assert r['break_even_reuses']==25 and r['full_project_break_even_reuses']==35
    assert a.research_disposition({**e,'quality_comparable':False})['break_even_reuses'] is None
    with pytest.raises(ValueError,match='independent meta-run clusters'):
        a.research_disposition({**e,'meta_recipe_claim':True,'independent_meta_run_clusters':1})


def test_reproduce_checks_immutable_evidence_before_output(tmp_path):
    a=api();e=put(tmp_path/'evidence.json',fixture(tmp_path))
    result=a.reproduce(e,{'destination':str(tmp_path/'reproduced')})
    assert result.status=='pass' # report reconstruction only, never a scientific pass
    e.path.write_bytes(b'{}')
    assert a.reproduce(e,{'destination':str(tmp_path/'bad')}).status=='blocked'


@pytest.mark.parametrize('candidate,decision',[(12,'positive_quality_but_slower'),(None,'positive_quality_efficiency_unresolved'),(8,'positive_quality_and_lower_deployment_cost')])
def test_positive_quality_report_preserves_unknown_project_cost(tmp_path,candidate,decision):
    a=api();e=fixture(tmp_path)
    e['claim']={'outcome':'positive','quality_comparable':True,'baseline_cost':10,
        'candidate_cost':candidate,'meta_cost':None,'method_tuning_cost':None}
    result=a.build_report([put(tmp_path/'evidence.json',e)],tmp_path/'report')
    report=json.loads(result.path.read_bytes())
    assert report['disposition']['decision']==decision
    assert report['disposition']['efficiency_improvement'] is (candidate==8)
    assert report['disposition']['break_even_reuses'] is None
    assert report['disposition']['full_project_break_even_reuses'] is None
    assert report['tables']['per_seed'][0]['cost']['project_seconds'] is None


@pytest.mark.parametrize('corruption',['altered','missing'])
def test_primary_ledgers_authenticated_without_reinterpreting_mapping(tmp_path,corruption):
    a=api();e=fixture(tmp_path)
    ledger=tmp_path/'baseline-sources.md';ledger.write_text('retained primary source ledger\n')
    ledger_ref={'path':str(ledger),'sha256':sha256_file(ledger)}
    mapping=put(tmp_path/'mapping.json',{'historical_original_path':'/absent/historical/identity.md','operational':ledger_ref})
    e['sources'].append({'path':str(mapping.path),'sha256':mapping.sha256})
    e['sections']['sources_and_deviations']={'primary_ledgers':[ledger_ref]}
    root=put(tmp_path/'evidence.json',e)
    assert a.reproduce(root,{'destination':str(tmp_path/'good')}).status=='pass'
    if corruption=='altered':ledger.write_text('changed bytes\n')
    else:ledger.unlink()
    assert sha256_file(root.path)==root.sha256 and sha256_file(mapping.path)==mapping.sha256
    with pytest.raises(ValueError,match='incomplete evidence'):
        a.build_report([root],tmp_path/'bad-report')
    assert a.reproduce(root,{'destination':str(tmp_path/'bad-reproduce')}).status=='blocked'
