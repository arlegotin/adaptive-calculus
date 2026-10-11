"""Summarize already measured calibration receipts; never executes a model."""
import hashlib
import json
import math
import os
from pathlib import Path

base = Path(os.environ['M11_ATTEMPT_DIR'])
calibration = base.parent / 'native-calibration-v1'


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def norm(vector):
    return math.sqrt(sum(x * x for x in vector))


def cosine(a, b):
    denominator = norm(a) * norm(b)
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else None


report = read(calibration / 'report.json')
assert report['status'] == 'calibration_measured_pending_interpretation'
design = read(Path(report['design']['path']))
results = {r['run']['id']: r['result'] for r in [read(Path(item['path'])) for item in report['runs']]}
assert len(results) == len(design['runs']) == 55


def gradient(prefix, sigma, count, first=0):
    differences = [results[f'{prefix}-{i}-1']['objective'] - results[f'{prefix}-{i}--1']['objective'] for i in range(first, first + count)]
    values = [sum(d * design['perturbations'][i][j] for i, d in zip(range(first, first + count), differences)) / (2 * count * sigma) for j in range(45)]
    return {'gradient': values, 'l2': norm(values), 'paired_differences': differences,
            'mean_absolute_pair_difference': sum(abs(d) for d in differences) / count}


signals = {str(sigma): gradient(f'signal-{sigma}', sigma, 8 if sigma == .05 else 4) for sigma in [.025, .05, .1]}
first4 = gradient('signal-0.05', .05, 4)
second4 = gradient('signal-0.05', .05, 4, 4)
large = gradient('query-large', .05, 4)
rankings = {}
for horizon in [16, 32, 64, 508]:
    rows = [{'candidate': name, 'result': results[f'rank-{name}-{horizon}']} for name in ['initialized', 'outer_step1', 'perturbed_direction1']]
    rankings[str(horizon)] = [{'candidate': r['candidate'], 'objective': r['result']['objective'], 'final_query': r['result']['final_query']} for r in sorted(rows, key=lambda r: r['result']['objective'])]
clipping = {}
for name in ['initialized', 'outer_step1', 'perturbed_direction1']:
    steps = results[f'rank-{name}-508']['work']['steps']
    bins = []
    for lower, upper in [(4, 128), (128, 256), (256, 384), (384, 512)]:
        subset = steps[lower - 4:upper - 4]
        bins.append({'global_step_start': lower, 'global_step_end_exclusive': upper,
                     'clipped': sum(s['feature_work']['clipped'] for s in subset),
                     'scalars': sum(s['feature_work']['scalars'] for s in subset)})
    clipping[name] = bins
initial = results['rank-initialized-508']
duplicate = results['duplicate-full-508']
lr = [results[f'lr49-max32-h16-{i}'] for i in range(2)]
value = {
    'schema': 'ralup.m11_calibration_interpretation.v1', 'status': 'measured_diagnostic_calibration',
    'report': ref(calibration / 'report.json'), 'design': report['design'], 'rankings': rankings,
    'signals': signals, 'm4_first': first4, 'm4_second': second4, 'query32768': large,
    'gradient_cosines': {'m4_first_vs_m8': cosine(first4['gradient'], signals['0.05']['gradient']),
        'm4_first_vs_m4_second': cosine(first4['gradient'], second4['gradient']),
        'query8192_vs32768_m4': cosine(first4['gradient'], large['gradient']),
        'sigma025_vs05_m4': cosine(signals['0.025']['gradient'], first4['gradient']),
        'sigma1_vs05_m4': cosine(signals['0.1']['gradient'], first4['gradient'])},
    'query_pair_sign_agreement': sum(a * b > 0 for a, b in zip(first4['paired_differences'], large['paired_differences'])),
    'full_repeat': {'objective_difference': duplicate['objective'] - initial['objective'],
                    'exact_final_state': duplicate['work']['final_state_hash'] == initial['work']['final_state_hash']},
    'lr49_max32_h16_repeat': {'objectives': [r['objective'] for r in lr], 'final_losses': [r['final_query'] for r in lr],
        'exact_final_state': lr[0]['work']['final_state_hash'] == lr[1]['work']['final_state_hash']},
    'aggregate_clipping_by_age': clipping,
    'per_feature_clipping': 'Not retained in this historical source; no actual per-feature counts can be recovered from aggregate receipts.',
    'work': {'optimizer_steps': sum(r['work']['optimizer_steps'] for r in results.values()),
             'training_targets': sum(r['work']['training_targets'] for r in results.values()),
             'query_targets': sum(r['work']['query_targets'] for r in results.values()),
             'summed_rollout_seconds': sum(r['work']['elapsed_seconds'] for r in results.values())},
    'scope': 'Same single META parent/task, early768-row diagnostic normalizer; duplicate engineering replay is not independent replication. No final-normalizer stability, learned gain, robust rejection, strongerB0 or confirmatory claim.',
}
encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
with (base / 'report.json').open('x') as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps({'report': ref(base / 'report.json'), 'gradient_cosines': value['gradient_cosines'],
                  'query_pair_sign_agreement': value['query_pair_sign_agreement'], 'work': value['work']}), flush=True)
