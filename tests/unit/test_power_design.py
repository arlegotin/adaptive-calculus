"""Engineering fixtures only; no trained arm or measured scientific pilot."""
import importlib
import json
import math

import numpy as np
import pytest
from scipy.stats import chi2, nct, norm, t


def api():
    # A missing API is an assertion failure in the initial RED run.
    spec = importlib.util.find_spec('ralup.analysis.power')
    assert spec is not None, 'E02 power design API is missing'
    return importlib.import_module('ralup.analysis.power')


@pytest.mark.parametrize('coverage', [.8, .95, .99])
def test_sd_upper_uses_lower_chi_square_quantile(coverage):
    values = np.array([-.02, .03, .01, -.04])
    expected = math.sqrt(3 * np.var(values, ddof=1) / chi2.ppf(1-coverage, 3))
    assert api().paired_sd_upper(values, coverage) == pytest.approx(expected)


@pytest.mark.parametrize('values', [[], [1.], [1., 2., 3.], [0., 0., 0., 0.],
                                     [2., 2., 2., 2.], [0., 1., 2., float('nan')]])
def test_degenerate_or_incomplete_pilot_is_inadequate(values):
    p = api()
    with pytest.raises(p.InadequatePilot):
        p.paired_sd_upper(np.array(values), .95)


@pytest.mark.parametrize('sd,effect', [(.02, .01), (.03, .004), (.01, .3)])
def test_confirmation_n_is_minimum_integer_for_upper_bound_below_zero(sd, effect):
    p = api(); n = p.confirmation_n(sd, effect)
    def power(k):
        return nct.sf(t.ppf(.9875, k-1), k-1, effect * math.sqrt(k) / sd)
    assert n >= 2 and power(n) >= .95
    assert n == 2 or power(n-1) < .95
    assert p.confirmation_n(sd, effect/2) >= n
    assert p.confirmation_n(sd*2, effect) >= n


def test_normal_start_uses_9875_not_975():
    p = api()
    expected = ((norm.ppf(.9875) + norm.ppf(.95)) * .03/.01)**2
    wrong = ((norm.ppf(.975) + norm.ppf(.95)) * .03/.01)**2
    assert p.normal_n_approximation(.03, .01) == pytest.approx(expected)
    assert abs(p.normal_n_approximation(.03, .01) - wrong) > 10


@pytest.mark.parametrize('sd,effect', [(0, .1), (-1, .1), (1, 0), (1, -1), (float('inf'), 1)])
def test_invalid_design_parameters_cannot_produce_tiny_n(sd, effect):
    with pytest.raises(ValueError):
        api().confirmation_n(sd, effect)


def test_two_contrast_curve_uses_larger_n_and_keeps_cost_unknown():
    p = api()
    curve = p.precision_curve({'vs_baseline': .03, 'vs_learned_lr': .06}, [.01, .02],
                              {'paired_seconds': None, 'paired_bytes': 100},
                              {'seconds': 10000, 'bytes': 100000})
    assert curve[0]['n'] == max(curve[0]['n_by_contrast'].values())
    assert curve[0]['n'] > curve[1]['n']
    assert curve[0]['forecast_seconds'] is None
    assert curve[0]['resource_status'] == 'unknown'
    assert curve[0]['forecast_bytes'] == curve[0]['n'] * 100
    assert curve[0]['ci_half_width_bpb']['vs_learned_lr'] == pytest.approx(
        t.ppf(.9875, curve[0]['n']-1)*.06/math.sqrt(curve[0]['n']))


def test_tail_and_stability_diagnostics_do_not_certify_normality():
    p = api()
    report = p.pilot_diagnostics(np.array([0., 0., 0., 1.]), .95)
    assert report['review_required']
    assert report['heavy_tail_flags']
    assert report['normality_certified'] is False
    assert report['leave_one_out_sd_min'] == 0
    assert len(report['ordered_deltas']) == 4


def save(tmp_path, name, payload):
    from ralup.config import canonical_bytes
    from ralup.records import ArtifactRef, sha256_file
    path = tmp_path/name
    path.write_bytes(canonical_bytes(payload))
    return ArtifactRef(path, sha256_file(path))


def fixture_plan(tmp_path, *, effects=None):
    effects = effects or [(-.05, -.06), (-.03, -.02), (-.02, -.04), (-.04, -.08)]
    refs = [save(tmp_path, f'pilot-{i}.json', {
        'schema': 'ralup.engineering_paired_pilot.v1', 'synthetic': True,
        'seed': i+20, 'status': 'completed', 'development_bpb': {
            'candidate': 1., 'baseline': 1.-a, 'learned_lr': 1.-b}})
            for i, (a, b) in enumerate(effects)]
    assumptions = {'purpose': 'synthetic-engineering', 'output': str(tmp_path/'plan.json'),
        'pilot_seeds': [20, 21, 22, 23], 'tuning_seeds': [10], 'confirmation_seeds': [100, 101],
        'sd_upper_coverage': .95, 'resources': {'seconds': 100000000, 'bytes': 100000000},
        'target_delta_bpb': .02, 'decision_rationale': 'Synthetic precision exercise only.',
        'tail_review': 'Synthetic ordered differences reviewed; normal assumption is conditional.',
        'bootstrap_seed': 8128}
    costs = {'paired_seconds': 10., 'paired_bytes': 100.,
        'scope': 'Synthetic complete paired deployment including attributable common warm starts; project dedup separate.'}
    return refs, costs, assumptions


def test_engineering_fixed_design_never_becomes_scientific_authority(tmp_path):
    p = api(); refs, costs, assumptions = fixture_plan(tmp_path)
    assert hasattr(p, 'plan_confirmation'), 'E02 planning API missing'
    ref = p.plan_confirmation(refs, costs, [.01, .02], assumptions)
    result = json.loads(ref.path.read_bytes())
    assert result['status'] == 'engineering_fixed_design'
    assert result['confirmation_eligible'] is False
    assert result['locked_n'] is None
    row = result['curves'][1]
    assert result['engineering_n'] == row['n'] == max(row['n_by_contrast'].values())
    assert len(result['pilot_runs']) == 4
    assert result['sensitivities']['bootstrap']['resampling_unit'] == 'paired complete training seed'
    assert result['failure_protocol']['replace_divergent_seed'] is False
    before = ref.path.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        p.plan_confirmation(refs, costs, [.01, .02], assumptions)
    assert ref.path.read_bytes() == before


@pytest.mark.parametrize('case', ['overlap', 'duplicate', 'omitted', 'tampered', 'science_label'])
def test_pilot_inventory_cannot_drop_reuse_or_relabel_units(tmp_path, case):
    p = api(); refs, costs, assumptions = fixture_plan(tmp_path)
    assert hasattr(p, 'plan_confirmation'), 'E02 planning API missing'
    if case == 'overlap': assumptions['confirmation_seeds'] = [20]
    if case == 'duplicate': refs[1] = refs[0]
    if case == 'omitted': refs.pop()
    if case == 'tampered': refs[0].path.write_text('{}')
    if case == 'science_label': assumptions['purpose'] = 'scientific-development'
    with pytest.raises(ValueError):
        p.plan_confirmation(refs, costs, [.02], assumptions)


@pytest.mark.parametrize('case,expected', [('zero', 'inadequate_pilot'), ('divergence', 'inadequate_pilot'),
    ('unstable', 'inadequate_pilot'), ('unknown_cost', 'insufficient_resources'),
    ('cost_exceeds', 'insufficient_resources'), ('missing_review', 'inadequate_pilot')])
def test_reportable_inadequate_precision_preserves_all_attempts(tmp_path, case, expected):
    p = api(); refs, costs, assumptions = fixture_plan(tmp_path)
    assert hasattr(p, 'plan_confirmation'), 'E02 planning API missing'
    if case == 'zero':
        refs, costs, assumptions = fixture_plan(tmp_path, effects=[(-.02, -.02)]*4)
    if case == 'unstable':
        refs, costs, assumptions = fixture_plan(tmp_path, effects=[(-.02, -.02)]*3+[(-1., -1.)])
    if case == 'divergence':
        value = json.loads(refs[0].path.read_bytes()); value['status'] = 'diverged'; value['development_bpb']['candidate'] = None
        refs[0] = save(tmp_path, 'diverged.json', value)
    if case == 'unknown_cost': costs['paired_seconds'] = None
    if case == 'cost_exceeds': assumptions['resources']['seconds'] = 1.
    if case == 'missing_review': assumptions.pop('tail_review')
    result = json.loads(p.plan_confirmation(refs, costs, [.02], assumptions).path.read_bytes())
    assert result['status'] == expected
    assert result['locked_n'] is None and result['engineering_n'] is None
    assert len(result['pilot_runs']) == 4
    assert result['global_unaffordability'] is False


def actual_selection():
    from pathlib import Path
    from ralup.records import ArtifactRef
    index = json.loads(Path('docs/harness/e01-evidence.json').read_bytes())
    attempt = next(a for a in index['attempts'] if a['attempt'] == 'select-finalists-v1')
    finish_ref = attempt['files']['finish.json']
    finish = json.loads(Path(finish_ref['path']).read_bytes())
    row = next(r for r in finish['completed_objects'] if r['path'] == 'selection.json')
    return ArtifactRef(Path(finish_ref['path']).parent/'selection.json', row['sha256'])


def test_actual_source_authenticated_refusal_is_not_a_scientific_pilot(tmp_path):
    p = api(); selection = actual_selection()
    assert hasattr(p, 'plan_confirmation'), 'E02 planning API missing'
    result = json.loads(p.plan_confirmation([], {}, [], {
        'purpose': 'scientific-development', 'output': str(tmp_path/'actual.json'),
        'selection': {'path': str(selection.path), 'sha256': selection.sha256}}).path.read_bytes())
    assert result['status'] == 'prerequisite_unresolved'
    assert result['pilot_runs'] == [] and result['curves'] == []
    assert result['locked_n'] is None
    assert result['scientific_prerequisites']['WP04'] == 'inconclusive'
    assert result['scientific_prerequisites']['WP06'] == 'NEEDS_REVISION'
    assert result['scientific_prerequisites']['candidate'] is None
    assert result['actual_paired_units'] == 0
    assert result['global_unaffordability'] is False


def test_forged_selected_flag_cannot_upgrade_actual_null_arms(tmp_path):
    p = api(); actual = actual_selection()
    assert hasattr(p, 'plan_confirmation'), 'E02 planning API missing'
    forged = json.loads(actual.path.read_bytes()); forged['confirmation_eligible'] = True
    fake = save(tmp_path, 'fake.json', forged)
    with pytest.raises(ValueError):
        p.plan_confirmation([], {}, [], {'purpose': 'scientific-development', 'output': str(tmp_path/'out.json'),
            'selection': {'path': str(fake.path), 'sha256': fake.sha256}})


def test_cli_emits_explicit_disposition_and_refuses_overwrite(tmp_path, capsys):
    from ralup.__main__ import main
    p = api(); selection = actual_selection()
    assert hasattr(p, 'plan_confirmation'), 'E02 planning API missing'
    pilot = save(tmp_path, 'input.json', {'pilot_runs': [], 'assumptions': {
        'purpose': 'scientific-development', 'selection': {'path': str(selection.path), 'sha256': selection.sha256}}})
    resources = save(tmp_path, 'resources.json', {'costs': {}, 'precision_choices': [],
        'resources': {'seconds': None, 'bytes': None}, 'output': str(tmp_path/'cli-plan.json')})
    assert main(['plan-confirmation', '--pilot', str(pilot.path), '--resources', str(resources.path)]) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout['status'] == 'prerequisite_unresolved'
    with pytest.raises(SystemExit) as error:
        main(['plan-confirmation', '--pilot', str(pilot.path), '--resources', str(resources.path)])
    assert error.value.code == 2
