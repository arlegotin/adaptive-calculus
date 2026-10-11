"""Authenticate completed M08 receipts and write a small durable evidence index."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path.cwd()
BASE = Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m08-20261011')
NATIVE = BASE / 'native-feature-diagnostic-v2'


def ref(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


primary = json.loads((NATIVE / 'feature-results.json').read_bytes())
assert primary['status'] == 'pass' and primary['new_training_targets'] == 196608
assert primary['new_query_targets'] == 0
assert all(c['resume_exact'] for c in primary['cases'])
assert all(ref(ROOT / name)['sha256'] == digest for name, digest in primary['source_identity'].items())
attempts = []
for path in sorted(BASE.iterdir()):
    if not path.is_dir() or not (path / 'finish.json').is_file():
        continue
    finish = json.loads((path / 'finish.json').read_bytes())
    for item in finish['completed_objects']:
        actual = ref(path / item['path'])
        assert actual['sha256'] == item['sha256'], (path, item['path'])
        assert (path / item['path']).stat().st_size == item['bytes']
    attempts.append({'name': path.name, 'startup': ref(path / 'startup.json'),
                     'finish': ref(path / 'finish.json'), 'stdout': ref(path / 'stdout.txt'),
                     'stderr': ref(path / 'stderr.txt'), 'exit_code': finish['exit_code'],
                     'elapsed_seconds': finish['elapsed_seconds'],
                     'completed_object_bytes': sum(i['bytes'] for i in finish['completed_objects'])})
cases = []
for case in primary['cases']:
    whole = sum(s['work']['elapsed_seconds'] for s in case['steps'])
    replay = sum(s['work']['elapsed_seconds'] for s in case['replay'])
    cases.append({'partition': case['partition'], 'parent': case['parent'],
                  'original_whole_step_seconds': whole, 'replay_whole_step_seconds': replay,
                  'original_feature_scan_seconds': case['feature_scan_seconds'],
                  'original_history_resolve_commit_seconds': case['history_scan_seconds'],
                  'scan_share_of_short_measured_whole_steps': case['feature_scan_seconds'] / whole,
                  'complete_scope_seconds': case['complete_scope_seconds'],
                  'final_matrix_history_bytes': case['final_history_bytes'],
                  'inherited_updates': case['inherited_updates'], 'inherited_targets': case['inherited_targets'],
                  'resume_exact': case['resume_exact'], 'replay_state_sha256': case['replay_state_sha256'],
                  'clipped': case['clipped'], 'scalars': case['scalars'],
                  'clipping_interpretation': ('frozen META-only normalized target' if case['partition'] == 'target_train'
                                              else 'raw reference collected before fitting; not normalized clipping evidence'),
                  'checkpoints': case['checkpoints'], 'sampled_memory': case['memory']})
index = {'schema': 'ralup.m08_feature_index.v1', 'task_base': 'a259ca20570e61fbaf8536978decaf30b7e3c63f',
         'status': 'completed-engineering-diagnostic-pending-fresh-task-review',
         'primary': ref(NATIVE / 'feature-results.json'), 'normalizer': ref(NATIVE / 'normalizer.json'),
         'meta_reference': ref(NATIVE / 'meta_train/reference-trajectory.json'),
         'target_clipping': ref(NATIVE / 'target_train/target-features.json'),
         'native_receipts': {name: ref(NATIVE / name) for name in
                             ('startup.json', 'runtime-observation.json', 'stdout.txt', 'stderr.txt', 'supervision.json', 'finish.json')},
         'executed_design': primary['design'], 'unexecuted_design_v1': ref(ROOT / 'docs/harness/m08-feature-design-v1.json'),
         'backbone': primary['backbone'], 'source_identity': primary['source_identity'],
         'cases': cases, 'physical_updates': 24, 'new_training_targets': 196608, 'query_targets': 0,
         'normalizer_fit_seconds': primary['normalizer']['fit_seconds'], 'normalizer_reference_rows': 96,
         'global_normalizer_tensor_bytes': 112, 'diagnostic_seconds': primary['complete_seconds'],
         'completed_attempts': attempts,
         'sum_completed_attempt_wall_seconds': sum(a['elapsed_seconds'] for a in attempts),
         'attempt_wall_sum_contract': 'Separate physical launches only; internal diagnostic/step/fit partitions are included, not added again. Editing/tool/enclosing costs unknown.',
         'active_administrative_attempt_excluded': os.environ.get('M08_ATTEMPT_DIR'),
         'audit_clock_exclusions': ['ledger.elapsed', 'event.elapsed_seconds', 'event.policy_seconds',
                                    'event.feature_work.scan_seconds', 'event.history_work.resolve_seconds',
                                    'event.history_work.commit_seconds', 'existing optimizer/EWGS/learned-scale audit timing fields'],
         'comparison_contract': 'Features, history, pending codes/scales, continuation clock, EMA, counts and all scientific owners remain exact; audit clocks stay charged in raw ledgers/complete receipts.',
         'limits': primary['limits'] + '; scan share is not a baseline slowdown or final-throughput estimate; memory samples overlap and true peaks are unknown; B0 null, WP04 inconclusive, E01 required.'}
out = ROOT / 'docs/harness/m08-feature-evidence.json'
with out.open('x') as stream:
    json.dump(index, stream, sort_keys=True, indent=2)
    stream.write('\n')
print(json.dumps({'index': ref(out), 'authenticated_completed_attempts': len(attempts),
                  'sum_completed_attempt_wall_seconds': index['sum_completed_attempt_wall_seconds']}))
