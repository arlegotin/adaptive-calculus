import json
from pathlib import Path

import pytest


def test_memory_views_do_not_double_count_and_reserve_os():
    from ralup.calibrate import memory_limits, memory_report
    limits = memory_limits(36 * 2**30, 28 * 2**30)
    assert limits['os_reserve_bytes'] == 9 * 2**30
    assert limits['resident_cap_bytes'] == 27 * 2**30
    report = memory_report([{'rss_bytes': 100, 'metal_bytes': 80, 'driver_bytes': 90}])
    assert report['sampled_maxima'] == {'rss_bytes': 100, 'metal_bytes': 80, 'driver_bytes': 90}
    assert report['unique_total_bytes'] is None
    assert report['driver_peak_bytes'] is None


def test_forecast_keeps_queries_resets_io_failures_and_ancestry():
    from ralup.calibrate import forecast_cost
    measured = {'step_seconds': [2., 4.], 'reset_seconds': [1., 3.],
                'query_seconds': [5., 6.], 'checkpoint_seconds': [7., 8.],
                'snapshot_bytes': 100, 'project_seconds': 40., 'parent_seconds': 10.}
    result = forecast_cost(measured, {'steps': 4, 'queries': 2, 'resets': 2, 'checkpoints': 2, 'retained': 3})
    assert result['future_seconds_interval'] == [34., 50.]
    assert result['prior_project_seconds'] == 50.
    assert result['retention_bytes'] == 300
    assert result['soap'] == 'remeasure actual state in M02/M07'


def test_uncharged_queries_reject():
    from ralup.calibrate import audit_queries
    with pytest.raises(ValueError, match='uncharged'):
        audit_queries({'query': 2}, [{'targets': 3}])
    audit_queries({'query': 3}, [{'targets': 3}])


def test_duration_doublings_never_select_scientific_budget():
    from ralup.calibrate import duration_jobs
    result = duration_jobs(8192, 3)
    assert [j['target_budget'] for j in result] == [8192, 16384, 32768]
    assert all(j['b0'] is None and j['selection'] == 'pending M07 development' for j in result)


def test_empirical_fit_requires_two_distinct_measured_sizes():
    from ralup.calibrate import fit_memory
    fit = fit_memory([{'microbatch': 1, 'rss_bytes': 100}, {'microbatch': 2, 'rss_bytes': 140}])
    assert fit['rss_bytes']['slope_bytes_per_block'] == 40
    with pytest.raises(ValueError, match='two'):
        fit_memory([{'microbatch': 1, 'rss_bytes': 100}])


def test_central_unique_union_does_not_count_disposable_reset_arms_twice():
    from ralup.calibrate import reconcile_cost_scope
    first = [{'document_id': 'source:article-a', 'offsets': [0, 1, 2]},
             {'document_id': 'source:article-b', 'offsets': [4, 5]}]
    second = [{'document_id': 'source:article-a', 'offsets': [1, 2, 3]}]
    raw = {'cost': {'stream': {'targets': 13, 'unique': 13, 'updates': 3}, 'passes': {'query': 2}},
           'arms': [{'attempts': [{'target_offsets': first}, {'target_offsets': second}]}],
           'queries': [{'targets': 2, 'target_offsets': [{'document_id': 'source:article-b', 'offsets': [5, 6]}]}],
           'cost_warmup': [{'target_offsets': first}]}
    raw['cost']['warmup_attempts'] = raw.pop('cost_warmup')
    fixed = reconcile_cost_scope(raw)
    assert raw['cost']['stream']['unique'] == 13
    assert fixed['cost']['stream']['unique'] == 7
    assert fixed['cost']['trajectory_unique_sum'] == 13
    assert fixed['global_coverage']['intervals'] == {'source:article-a': [[0, 4]], 'source:article-b': [[4, 7]]}
    assert len(fixed['global_coverage']['sha256']) == 64
    assert fixed['cost']['passes'] == raw['cost']['passes']
