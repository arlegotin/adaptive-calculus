"""S33 paired precision planning, separate from confirmation and score access.

A unit is a fresh complete training seed. The success event is the upper end
of a two-sided interval below zero, not rejection in either direction. SD
coverage is conditional on independent normal paired differences; diagnostics
never certify that assumption from a four-unit probe.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2, nct, norm, t

from ..records import ArtifactRef


class InadequatePilot(ValueError):
    """Missing, nonfinite or degenerate data cannot determine a usable variance."""


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _probability(value, name):
    value = _positive(value, name)
    if value >= 1:
        raise ValueError(f'{name} must be below one')
    return value


def _deltas(deltas):
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise InadequatePilot('At least four complete finite paired differences required; retain missing/failing arms')
    sd = float(np.std(values, ddof=1))
    if not math.isfinite(sd) or sd <= 0:
        raise InadequatePilot('Degenerate paired variance: no small-n inference permitted')
    return values, sd


def paired_sd_upper(deltas: np.ndarray, coverage: float) -> float:
    """One-sided normal-theory SD upper bound; raises an inadequate-pilot flag."""
    coverage = _probability(coverage, 'SD upper coverage')
    values, sd = _deltas(deltas)
    result = sd * math.sqrt((len(values)-1) / chi2.ppf(1-coverage, len(values)-1))
    if not math.isfinite(result):
        raise InadequatePilot('Unbounded SD forecast')
    return result


def normal_n_approximation(sd_upper, delta_design, power=.95, interval_confidence=.975):
    sd_upper = _positive(sd_upper, 'SD upper')
    delta_design = _positive(delta_design, 'detectable BPB difference')
    power = _probability(power, 'power')
    confidence = _probability(interval_confidence, 'interval confidence')
    return float(((norm.ppf((1+confidence)/2) + norm.ppf(power)) * sd_upper/delta_design)**2)


def computed_power(n, sd_upper, delta_design, interval_confidence=.975):
    """Probability U<0 for negative true delta of the declared magnitude."""
    if type(n) is not int or n < 2:
        raise ValueError('At least two independent confirmation units required')
    sd_upper = _positive(sd_upper, 'SD upper')
    delta_design = _positive(delta_design, 'detectable BPB difference')
    confidence = _probability(interval_confidence, 'interval confidence')
    critical = t.ppf((1+confidence)/2, n-1)
    return float(nct.sf(critical, n-1, delta_design*math.sqrt(n)/sd_upper))


def confirmation_n(sd_upper: float, delta_design: float, power: float = .95,
                   interval_confidence: float = .975) -> int:
    """Smallest integer n satisfying the noncentral-t upper-bound success event."""
    estimate = normal_n_approximation(sd_upper, delta_design, power, interval_confidence)
    if not math.isfinite(estimate) or estimate > 2**31:
        raise ValueError('Design exceeds supported numerical search range; do not claim unaffordability')
    high = max(2, math.ceil(estimate)); low = 1
    while computed_power(high, sd_upper, delta_design, interval_confidence) < power:
        high *= 2
        if high > 2**31:
            raise ValueError('Design exceeds supported numerical search range')
    while high-low > 1:
        middle = (high+low)//2
        if computed_power(middle, sd_upper, delta_design, interval_confidence) >= power:
            high = middle
        else:
            low = middle
    return high


def pilot_diagnostics(deltas, coverage, *, max_loo_sd_ratio=2., max_variance_share=.8,
                      max_sd_mad_ratio=3.):
    """Recorded descriptive tail/influence thresholds, not normality acceptance."""
    values, sd = _deltas(deltas)
    for number in (max_loo_sd_ratio, max_variance_share, max_sd_mad_ratio):
        _positive(number, 'diagnostic threshold')
    loo_sd = [float(np.std(np.delete(values, i), ddof=1)) for i in range(len(values))]
    loo_mean = [float(np.mean(np.delete(values, i))) for i in range(len(values))]
    share = float(np.max((values-values.mean())**2) / np.sum((values-values.mean())**2))
    mad = float(np.median(np.abs(values-np.median(values))))
    ratio = sd/(1.4826*mad) if mad else None
    stability = max(loo_sd)/min(loo_sd) if min(loo_sd) > 0 else None
    flags = []
    if ratio is None or ratio > max_sd_mad_ratio:
        flags.append('SD/MAD tail sensitivity')
    if share >= max_variance_share:
        flags.append('single-unit variance concentration')
    unstable = stability is None or stability > max_loo_sd_ratio
    rank_flip = min(loo_mean) <= 0 <= max(loo_mean)
    return {'n': len(values), 'ordered_deltas': values.tolist(), 'mean': float(values.mean()),
            'sample_sd': sd, 'sd_upper': paired_sd_upper(values, coverage), 'sd_upper_coverage': coverage,
            'leave_one_out_sd_min': min(loo_sd), 'leave_one_out_sd_max': max(loo_sd),
            'leave_one_out_sd_ratio': stability, 'leave_one_out_mean_range': [min(loo_mean), max(loo_mean)],
            'rank_sign_unstable': rank_flip, 'variance_unstable': unstable,
            'max_centered_variance_share': share, 'sd_over_scaled_mad': ratio,
            'heavy_tail_flags': flags, 'normality_certified': False,
            'review_required': bool(flags or unstable or rank_flip),
            'thresholds': {'max_loo_sd_ratio': max_loo_sd_ratio, 'max_variance_share': max_variance_share,
                           'max_sd_mad_ratio': max_sd_mad_ratio}}


def precision_curve(sd_upper, precision_choices, costs, resources):
    """Effect/conditional CI-width/cost rows; unknown work never becomes zero."""
    if set(sd_upper) != {'vs_baseline', 'vs_learned_lr'} or not precision_choices:
        raise ValueError('Two co-primary SDs and nonempty prospective precision choices required')
    for key in ('paired_seconds', 'paired_bytes'):
        if costs.get(key) is not None:
            _positive(costs[key], key)
    for key in ('seconds', 'bytes'):
        if resources.get(key) is not None:
            _positive(resources[key], key)
    rows = []
    for effect in precision_choices:
        counts = {name: confirmation_n(sd, effect) for name, sd in sd_upper.items()}
        n = max(counts.values())
        seconds = n*costs['paired_seconds'] if costs.get('paired_seconds') is not None else None
        size = n*costs['paired_bytes'] if costs.get('paired_bytes') is not None else None
        unknown = any(value is None for value in (seconds, size, resources.get('seconds'), resources.get('bytes')))
        exceeds = ((seconds is not None and resources.get('seconds') is not None and seconds > resources['seconds'])
                   or (size is not None and resources.get('bytes') is not None and size > resources['bytes']))
        rows.append({'delta_design_bpb': effect, 'normal_n_by_contrast': {
            name: normal_n_approximation(sd, effect) for name, sd in sd_upper.items()},
            'n_by_contrast': counts, 'n': n, 'power_by_contrast': {
                name: computed_power(n, sd, effect) for name, sd in sd_upper.items()},
            'ci_half_width_bpb': {name: float(t.ppf(.9875, n-1)*sd/math.sqrt(n)) for name, sd in sd_upper.items()},
            'ci_width_interpretation': 'planning width conditional on SD upper, not an assurance bound',
            'forecast_seconds': seconds, 'forecast_bytes': size,
            'resource_status': 'exceeds_declared_envelope' if exceeds else ('unknown' if unknown else 'within_declared_envelope')})
    return rows


def _current_prerequisites(selection_ref):
    """Authenticate the actual closed E01 disposition and its original producer.

    This is the current null-arm gate, not a general future selection registry.
    Exploratory measurement does not require confirmation eligibility; a future
    real pilot must bind actual frozen arms, lineage, scoring and complete costs
    through the appropriate producer once that scientific authority exists.
    """
    from ..search import _read, _prerequisite_disposition
    from ..records import sha256_file
    root = Path(__file__).resolve().parents[3]
    index = json.loads((root/'docs/harness/e01-evidence.json').read_bytes())
    attempt = next(a for a in index['attempts'] if a['attempt'] == 'select-finalists-v1')
    startup_ref = ArtifactRef(**attempt['files']['startup.json'])
    finish_ref = ArtifactRef(**attempt['files']['finish.json'])
    startup, finish = _read(startup_ref), _read(finish_ref)
    objects = {row['path']: row for row in finish['completed_objects']}
    expected = ArtifactRef(finish_ref.path.parent/'selection.json', objects['selection.json']['sha256'])
    if selection_ref != expected:
        raise ValueError('Selection must be the indexed original source-authenticated E01 disposition')
    value = _read(selection_ref)
    if (finish['exit_code'] != 0 or value['source_identity'] != {
            k: v['sha256'] for k, v in startup['source'].items() if k.startswith('src/')}):
        raise ValueError('Selection source/producer mismatch')
    for name, row in startup['source'].items():
        if sha256_file(finish_ref.path.parent/'producer'/name) != row['sha256']:
            raise ValueError('Selection producer archive changed')
    if objects['startup.json']['sha256'] != startup_ref.sha256:
        raise ValueError('Selection startup lacks completed authentication')
    dates = [datetime.fromisoformat(x) for x in (startup['utc'], value['created_utc'], finish['completed_utc'])]
    if any(x.tzinfo is None for x in dates) or dates != sorted(dates):
        raise ValueError('Selection producer dates are inconsistent')
    gates = value['prerequisite_evidence']
    measured = _prerequisite_disposition({'b0_evidence': gates['B0'], 'controller_evidence': gates['controller']})
    if any(value.get(key) != item for key, item in measured.items()):
        raise ValueError('Selection differs from indexed measured prerequisite gates')
    if value.get('confirmation_eligible') is not False:
        raise ValueError('Current E01 selection is not confirmation authority')
    return {**measured, 'selection': {'path': str(selection_ref.path), 'sha256': selection_ref.sha256},
            'producer': {'startup': attempt['files']['startup.json'], 'finish': attempt['files']['finish.json']},
            'unresolved_methods': value['unresolved_methods'], 'blocked_methods': value['blocked_methods']}


def _write_plan(path, report):
    # No training-module import needed for a statistical JSON publication.
    import os
    from ..config import canonical_bytes
    from ..records import sha256_file
    path = Path(path)
    encoded = canonical_bytes(report)+b'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    return ArtifactRef(path, sha256_file(path))


def _protocol(bootstrap_seed):
    return {'sensitivities': {
        'bootstrap': {'resampling_unit': 'paired complete training seed', 'joint_contrast_resampling': True,
                      'replicates': 10000, 'seed': bootstrap_seed, 'interval_confidence': .975,
                      'method': 'percentile; descriptive sensitivity, small-n limitations reported'},
        'sign': {'method': 'exact two-sided binomial at probability 0.5', 'zero_differences': 'omit and report count',
                 'comparison_alpha': .025},
        'rank': {'method': 'two-sided paired signed-rank; exact sign enumeration including tied midranks',
                 'zero_differences': 'omit and report count', 'comparison_alpha': .025,
                 'large_n': 'seeded sign permutation Monte Carlo with 100000 draws and plus-one correction'},
        'role': 'predeclared sensitivities; never replace both primary paired t upper bounds below zero'},
        'failure_protocol': {'replace_divergent_seed': False, 'finite_budget_divergence': 'substantive failure',
            'missing_or_nonfinite': 'retain all arms; no finite paired t claim and no variance-based lock',
            'implementation_invalidation': 'explicit documented adjudication of affected paired units; retain original receipts',
            'optional_seed_addition_after_protected_scores': False,
            'development_extension': 'new prospective disjoint batch and rationale for unstable variance/ranks; retain every pilot'}}


def plan_confirmation(pilot_runs: list[ArtifactRef], costs: dict, precision_choices: list[float],
                      assumptions: dict) -> ArtifactRef:
    """Publish engineering precision design or the authenticated actual gate.

    Current genuine final arms do not exist. No caller flag, engineering data or
    initialized policy can upgrade this interface to a scientific design. A
    future exploratory pilot need not be confirmation-eligible, but requires a
    real producer for complete matched frozen arms/lineages/outcomes/costs. Until
    E01/reopened WP06 supplies it, the scientific path records its exact refusal.
    """
    from ..search import _read, _development_only, source_identity
    _development_only(assumptions); _development_only(costs)
    purpose = assumptions.get('purpose')
    if purpose not in ('synthetic-engineering', 'scientific-development'):
        raise ValueError('Explicit engineering or scientific-development purpose required')
    output = Path(assumptions['output'])
    if output.exists():
        raise FileExistsError('Immutable confirmation plan already exists')
    refs = [{'path': str(ref.path), 'sha256': ref.sha256} for ref in pilot_runs]
    values = [_read(ref) for ref in pilot_runs]
    for value in values:
        _development_only(value)
    report = {'schema': 'ralup.confirmation_plan.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'research_cutoff': '2026-10-03', 'purpose': purpose, 'pilot_runs': refs, 'assumptions': assumptions,
        'costs': costs, 'precision_choices_bpb': precision_choices, 'source_identity': source_identity(),
        'interval_confidence': .975, 'comparison_power': .95, 'joint_power_lower_bound': .90,
        'sd_bound_assumption': 'independent identically distributed normal paired differences; coverage is conditional',
        'confirmation_eligible': False, 'locked_n': None, 'engineering_n': None,
        'global_unaffordability': False, 'uses_protected_scores': False, 'curves': [], 'diagnostics': {},
        **_protocol(assumptions.get('bootstrap_seed'))}
    if purpose == 'scientific-development':
        if not assumptions.get('selection'):
            raise ValueError('Actual selected-arm source authority required; caller pilot labels are insufficient')
        prerequisites = _current_prerequisites(ArtifactRef(**assumptions['selection']))
        report.update(status='prerequisite_unresolved', scientific_prerequisites=prerequisites,
            actual_paired_units=0, unadmitted_pilot_refs=refs,
            reasons=['No actual same-scope selected candidate, learned-LR and B*; four fresh paired units cannot be formed.',
                     'WP04/B0 applicability and WP06 final controller/budgets remain unresolved.'],
            forecast_scope='No measured pilot variance, effect/CI/cost curve or usable locked design available.',
            next_owner='E01/reopened WP06 supplies genuine trained/selected arms; E02 then authenticates fresh pilot provenance.',
            exploratory_rule='Exploratory pilots may precede confirmation eligibility/manifest, but need authentic frozen arms/config/data/budget, independent complete lineages disjoint tuning, final development outcomes and complete failures/costs; provisional policies only support diagnostic variance.')
        return _write_plan(output, report)
    coverage = _probability(assumptions['sd_upper_coverage'], 'SD coverage')
    inventories = [assumptions[k] for k in ('pilot_seeds', 'tuning_seeds', 'confirmation_seeds')]
    if any(not isinstance(items, list) or any(type(x) is not int or x < 0 for x in items)
           or len(set(items)) != len(items) for items in inventories):
        raise ValueError('Unique integer seed inventories required')
    if any(set(inventories[i]) & set(inventories[j]) for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Pilot, tuning and confirmation seeds must be disjoint')
    if any(value.get('schema') != 'ralup.engineering_paired_pilot.v1' or value.get('synthetic') is not True for value in values):
        raise ValueError('Engineering fixtures require explicit synthetic artifact schema')
    seeds = [value['seed'] for value in values]
    if seeds != inventories[0] or len(set(seeds)) != len(seeds):
        raise ValueError('Retain the complete prospective ordered pilot seed inventory, including failures')
    if type(assumptions['bootstrap_seed']) is not int or assumptions['bootstrap_seed'] < 0:
        raise ValueError('Predeclared nonnegative bootstrap seed required')
    if not assumptions.get('decision_rationale') or not costs.get('scope'):
        raise ValueError('Prospective precision rationale and complete cost boundary required')
    if assumptions['target_delta_bpb'] not in precision_choices:
        raise ValueError('Prospective target must appear in precision choices')
    report.update(actual_paired_units=0, engineering_paired_units=len(values), reasons=[],
                  forecast_scope='Synthetic engineering only; no scientific pilot or resource inference')
    try:
        if any(value.get('status') != 'completed' for value in values):
            raise InadequatePilot('Retained failed/divergent/incomplete unit: no silent replacement or finite-pair subset')
        contrasts = {'vs_baseline': [], 'vs_learned_lr': []}
        for value in values:
            scores = value['development_bpb']
            if set(scores) != {'candidate', 'baseline', 'learned_lr'} or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores.values()):
                raise InadequatePilot('All three final development scores must be finite')
            contrasts['vs_baseline'].append(scores['candidate']-scores['baseline'])
            contrasts['vs_learned_lr'].append(scores['candidate']-scores['learned_lr'])
        diagnostics = {name: pilot_diagnostics(np.array(deltas), coverage,
            **assumptions.get('diagnostic_thresholds', {})) for name, deltas in contrasts.items()}
        report['diagnostics'] = diagnostics
        report['curves'] = precision_curve({k: v['sd_upper'] for k, v in diagnostics.items()},
                                            precision_choices, costs, assumptions['resources'])
        if any(row['review_required'] for row in diagnostics.values()):
            raise InadequatePilot('Variance/rank/tail diagnostics require a prospective development extension or review; no lock')
        if not assumptions.get('tail_review'):
            raise InadequatePilot('Explicit development tail review required before design freeze')
    except InadequatePilot as error:
        report.update(status='inadequate_pilot', reasons=[str(error)])
        return _write_plan(output, report)
    chosen = next(row for row in report['curves'] if row['delta_design_bpb'] == assumptions['target_delta_bpb'])
    if chosen['resource_status'] != 'within_declared_envelope':
        report.update(status='insufficient_resources', reasons=[chosen['resource_status']])
    else:
        report.update(status='engineering_fixed_design', engineering_n=chosen['n'])
    return _write_plan(output, report)


def plan_confirmation_from_files(pilot_path, resources_path):
    from ..records import sha256_file
    pilot_path, resources_path = Path(pilot_path), Path(resources_path)
    pilot = json.loads(pilot_path.read_bytes()); resources = json.loads(resources_path.read_bytes())
    assumptions = {**pilot['assumptions'], 'resources': resources['resources'], 'output': resources['output'],
                   'input_files': [{'path': str(path), 'sha256': sha256_file(path)} for path in (pilot_path, resources_path)]}
    return plan_confirmation([ArtifactRef(**ref) for ref in pilot['pilot_runs']], resources['costs'],
                             resources['precision_choices'], assumptions)
