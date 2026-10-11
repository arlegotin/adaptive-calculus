"""Prospective broad-age META-only reference refresh and complete final pool."""
import copy
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from ralup.calibrate import _Sampler, _limits, _offsets, memory_report
from ralup.config import canonical_bytes
from ralup.optim.base import SchedulePoint
from ralup.policy.adapter import PolicyAdapter
from ralup.policy.checks import _ref, _write, read_exploratory_backbone
from ralup.policy.controllers import Affine45
from ralup.policy.normalizer import FrozenNormalizer, fit_normalizer
from ralup.records import ArtifactRef, sha256_file
from ralup.search import create_trial_state, source_identity
from ralup.state import capture_snapshot, exact_cpu_state_hash, restore_snapshot, save_snapshot
from ralup.train import _rng_state, _restore_rng, select_logical_batch, train_step

base = Path(os.environ['M11_ATTEMPT_DIR'])
store = base.parent
bootstrap = store / 'native-bootstrap-v2'


def controller(parameters):
    result = Affine45()
    offset = 0
    with torch.no_grad():
        for parameter in result.parameters():
            parameter.copy_(torch.tensor(parameters[offset:offset + parameter.numel()]).reshape(parameter.shape))
            offset += parameter.numel()
    return result


def bind(state, parameters, normalizer, normalizer_ref):
    old = state.method
    rng = _rng_state()
    try:
        new = PolicyAdapter(old.rule, controller(parameters), normalizer, normalizer_ref=normalizer_ref)
    finally:
        _restore_rng(rng)
    new.bind(dict(state.model.named_parameters()), old.groups)
    new.optimizer_states = old.optimizer_states
    if type(old) is PolicyAdapter:
        new.last_actions = copy.deepcopy(old.last_actions)
    state.method = new
    state.optimizer_state = new.optimizer_states
    state.configuration['policy_features'] = {'enabled': True, 'full_horizon': 512, 'trace_cadence': 1}
    state.histories['feature_normalizer'] = normalizer


if sys.argv[1] == 'freeze':
    prior_report = json.loads((bootstrap / 'report.json').read_bytes())
    prior = json.loads(Path(prior_report['design']['path']).read_bytes())
    generation = json.loads((bootstrap / 'generation-1.json').read_bytes())
    calibration = store / 'native-calibration-v1' / 'report.json'
    assert json.loads(calibration.read_bytes())['status'] == 'calibration_measured_pending_interpretation'
    protocol, backbone = read_exploratory_backbone(Path(prior['backbone']['path']))
    current = source_identity()
    for name in prior['unchanged_numerical_sources']:
        assert prior['source_identity'][name] == current[name], name
    cfg = copy.deepcopy(prior['config'])
    cfg.update(source_identity=current, milestones=[16, 256, 448], finite_forecast_seconds=1200,
               label='m11-final-reference-pool', recipe=protocol['recipe'],
               source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    design = {
        'schema': 'ralup.m11_reference_design.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_identity': current, 'config': cfg, 'backbone': backbone,
        'bootstrap_generation': _ref(bootstrap / 'generation-1.json'), 'calibration': _ref(calibration),
        'calibration_interpretation': _ref(store / 'interpret-calibration-v1' / 'report.json'),
        'historical_full_owner_limitation': 'Old rollout rebinding dropped last_actions before first update; retained old objectives/hashes are observations of that limited producer, not complete-owner clone proof. This stage creates fresh corrected-source states only.',
        'prior_design': prior_report['design'], 'runtime': prior['runtime'],
        'initial_normalizer': prior_report['reference_normalizer'],
        'origins': [
            {'name': 'baseline', 'parameters': generation['phi_before']},
            {'name': 'refreshed_policy', 'parameters': generation['phi_after']},
        ],
        'reference_steps': 448, 'snapshot_steps': [16, 256, 448], 'full_horizon': 512,
        'pool_weights': [1 / 6] * 6,
        'refresh_rule': 'Bootstrap outer-step1 fixed before final reference outcomes; independent of all future antithetic candidates. No choice of best observed calibration perturbation.',
        'fit_rule': 'Uniform population mean/std over every raw14-feature matrix row in both META-TRAIN0..447 reference trajectories; floor1e-6, clip5. Rebind normalizer on each complete snapshot without resetting moments/history/cursor/RNG.',
        'normalizer_source_scope': 'Final exploratory reference fit for the frozen512-continuation recipe. Later changed recipe/distribution requires explicit reassessment; no protected or development fitting.',
        'maximum_optimizer_steps': 896, 'maximum_training_targets': 896 * 8192, 'maximum_query_targets': 0,
        'forecast_seconds': 1200, 'hard_deadline_seconds': 1260, 'forecast_bytes': 2 * 1024 ** 3,
        'forecast_basis': 'Measured bootstrap mean step.5871s and full508 calibration~270s;896updates~526s plus twelve complete milestone writes/restores/rawfeature IO. Conservative1200s ceiling, not scientific unaffordability.',
        'retention': 'Six original complete origin/age snapshots and six final-normalizer rebound complete pool milestones; raw reference rows and all step receipts. No old objects deleted; remaining successful final working tensors are disposable.',
        'risk_scope': 'Source changes limited to meta orchestration/full-owner binding and feature-clipping telemetry; inherited numerical kernels unchanged. Fresh full states only. Changed reference-normalizer/policy distribution requires prospectively scoped horizon compatibility before final learning budgets.',
        'helpers': {str(p): _ref(p) for p in map(Path, ['docs/harness/m11-reference.py', 'docs/harness/m11-attempt.py', 'docs/harness/m11-supervise.py'])},
        'claims': 'Exploratory-only, B0null/WP04inconclusive/confirmation-ineligible; reference generation alone establishes no policy gain.',
    }
    _write(base / 'design.json', design)
    print(json.dumps(_ref(base / 'design.json')), flush=True)
else:
    design_path = Path(sys.argv[2])
    design = json.loads(design_path.read_bytes())
    assert design['source_identity'] == source_identity()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0' and torch.backends.mps.is_available()
    assert subprocess.check_output(['sw_vers', '-productVersion'], text=True).strip() == design['runtime']['macOS']
    assert subprocess.check_output(['sw_vers', '-buildVersion'], text=True).strip() == design['runtime']['build']
    assert platform.machine() == design['runtime']['architecture']
    assert sha256_file('requirements.lock') == design['runtime']['lock_sha256']
    for ref in design['helpers'].values():
        assert sha256_file(ref['path']) == ref['sha256']
    started = time.perf_counter()
    report = {'design': _ref(design_path), 'source_identity': source_identity(), 'status': 'running', 'origins': [], 'pool': [], 'confirmation_eligible': False}
    normalizer_ref = ArtifactRef(**design['initial_normalizer'])
    assert sha256_file(normalizer_ref.path) == normalizer_ref.sha256
    n = json.loads(normalizer_ref.path.read_bytes())
    initial_normalizer = FrozenNormalizer(torch.tensor(n['mean']), torch.tensor(n['std']), n['source_manifest'])
    retained = []
    trajectories = []
    try:
        with _Sampler('mps', _limits(design['config']['resolved'])) as sampler:
            def cap():
                sampler.sample()
                if sampler.exceeded or time.perf_counter() - started > design['forecast_seconds']:
                    raise RuntimeError('Frozen reference time/memory envelope exhausted')

            for origin in design['origins']:
                cap()
                state = create_trial_state(design['config'])
                bind(state, origin['parameters'], initial_normalizer, normalizer_ref)
                features = []
                summary = {'name': origin['name'], 'steps': [], 'snapshots': []}
                report['origins'].append(summary)
                with (base / (origin['name'] + '-steps.jsonl')).open('x') as journal:
                    for step in range(design['reference_steps']):
                        cap()
                        state.schedule['point'] = SchedulePoint(design['config']['recipe']['learning_rate'], 0., step / 512)
                        batch = select_logical_batch(state, 8192)
                        before = len(state.ledger.events)
                        try:
                            result = train_step(state, batch)
                        finally:
                            journal.write(canonical_bytes({'step': step, 'inputs': _offsets(batch), 'events': state.ledger.events[before:]}).decode() + '\n')
                            journal.flush()
                            os.fsync(journal.fileno())
                        summary['steps'].append({'step': step, 'work': result.work})
                        for name in state.model._contexts:
                            features.append({'step': step, 'matrix': name, 'values': state.histories[name].last_features.cpu().tolist()})
                        if step + 1 in design['snapshot_steps']:
                            snapshot = capture_snapshot(state)
                            ref = save_snapshot(snapshot, base / f"{origin['name']}-{step + 1}-original")
                            item = {'origin': origin['name'], 'step': step + 1, 'snapshot': _ref(ref.path), 'state_hash': exact_cpu_state_hash(state)}
                            summary['snapshots'].append(item)
                            retained.append((origin, step + 1, snapshot, item))
                            print(json.dumps({'origin': origin['name'], 'step': step + 1, 'snapshot': item['snapshot']}), flush=True)
                path = base / (origin['name'] + '-reference.json')
                _write(path, {'schema': 'ralup.feature_trajectory.v1', 'partition': 'meta_train', 'role': 'reference', 'synthetic': False,
                    'source_identity': source_identity(), 'data_manifest': design['config']['resolved']['data']['manifest'],
                    'producer': _ref(base / 'startup.json'), 'backbone': design['backbone'], 'parent': design['config']['parent'],
                    'continuation_horizon': 512, 'rows': features})
                trajectories.append(ArtifactRef(**_ref(path)))
                _write(base / (origin['name'] + '-complete.json'), summary)
            fitted = fit_normalizer(trajectories)
            path = base / 'final-reference-normalizer.json'
            _write(path, {'mean': fitted.mean.tolist(), 'std': fitted.std.tolist(), 'source_manifest': fitted.source_manifest,
                          'scope': design['normalizer_source_scope']})
            fitted_ref = ArtifactRef(**_ref(path))
            report['normalizer'] = _ref(path)
            for origin, step, snapshot, original in retained:
                cap()
                state = restore_snapshot(snapshot, 'mps')
                bind(state, origin['parameters'], fitted, fitted_ref)
                ref = save_snapshot(capture_snapshot(state), base / f"{origin['name']}-{step}-pool")
                report['pool'].append({'origin': origin['name'], 'step': step, 'snapshot': _ref(ref.path),
                    'original': original, 'state_hash': exact_cpu_state_hash(state), 'weight': 1 / 6})
            report['memory'] = memory_report(sampler.samples)
            report['status'] = 'final_reference_pool_ready_pending_horizon_compatibility'
    except BaseException as error:
        report.update(status='failed', failure=str(error), unknown_partial_work=True)
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        _write(base / 'report.json', report)
