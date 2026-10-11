"""Affected final-normalizer signal/horizon assessment; no selected controller."""
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from ralup.calibrate import _Sampler, _limits, memory_report
from ralup.config import canonical_bytes
from ralup.data.batches import next_logical_batch
from ralup.data.blocks import TargetBlock
from ralup.data.cursor import BatchCursor
from ralup.meta.es import OuterAdam, train_generation
from ralup.meta.tasks import MetaTask, _input_digest, rollout
from ralup.policy.checks import _ref, _write
from ralup.policy.controllers import ControllerState, LearnedLR49, freeze_controller
from ralup.records import ArtifactRef, sha256_file
from ralup.search import source_identity
from ralup.state import load_snapshot, restore_snapshot

base = Path(os.environ['M11_ATTEMPT_DIR'])
store = base.parent


def read(path):
    return json.loads(Path(path).read_bytes())


def materialize(entry, blocks):
    snapshot = load_snapshot(ArtifactRef(**entry['snapshot']))
    state = restore_snapshot(snapshot, 'mps')
    batches = [next_logical_batch(state.cursor, 8192) for _ in range(512 - state.schedule['step'])]
    hashes = {b.article_hash for batch in batches for b in batch.block_slices} | {b.article_hash for b in blocks}
    return MetaTask(snapshot, batches, blocks, {'query_seed': 4112},
        {'partition': 'meta_train', 'synthetic': False, 'device': 'mps',
         'document_partitions': {h: 'meta_train' for h in hashes}})


def panel_objectives(journal, panels):
    """Pure reduction of already completed per-block evaluation receipts."""
    membership = {document: label for label, documents in panels.items() for document in documents}
    endpoints = []
    current = None
    for line in journal.read_text().splitlines():
        row = json.loads(line)
        if row['event'] != 'query_progress':
            continue
        if row['phase'] == 'before_contexts':
            current = {label: {'losses': [], 'targets': 0} for label in panels}
            endpoints.append(current)
        elif row['phase'] == 'block_completed':
            item = current[membership[row['document_id']]]
            item['losses'].append(row['block_nll_sum'])
            item['targets'] += row['block_targets']
    assert len(endpoints) % 2 == 0
    results = []
    for initial, final in zip(endpoints[::2], endpoints[1::2]):
        result = {}
        for label in panels:
            assert initial[label]['targets'] == final[label]['targets'] == 32768
            a = math.fsum(initial[label]['losses']) / 32768
            b = math.fsum(final[label]['losses']) / 32768
            result[label] = {'initial': a, 'final': b, 'delta': b - a, 'objective': (b - a) / max(a, 1e-6)}
        results.append(result)
    return results


if sys.argv[1] == 'freeze':
    reference_path = store / 'native-reference-v1' / 'report.json'
    reference = read(reference_path)
    assert reference['status'] == 'final_reference_pool_ready_pending_horizon_compatibility'
    prior = read(reference['design']['path'])
    assert reference['source_identity'] == source_identity()
    entries = {f"{entry['origin']}-{entry['step']}": entry for entry in reference['pool']}
    initial = materialize(entries['baseline-16'], [])
    inner = {b.article_hash for batch in initial.inner_batches for b in batch.block_slices}
    manifest = prior['config']['resolved']['data']['manifest']
    blocks = []
    excluded = set(inner)
    panels = {}
    for label, seed in [('A', 4112), ('B', 4113)]:
        cursor = BatchCursor.from_manifest(Path(manifest['path']), manifest['sha256'], partition='meta-train', seed=seed)
        selected = []
        count = 0
        while count < 32768:
            block = cursor.take(min(128, 32768 - count))
            if block.article_hash not in excluded:
                selected.append(block)
                count += sum(block.score_mask)
        panels[label] = sorted({b.document_id for b in selected})
        excluded |= {b.article_hash for b in selected}
        blocks.extend(selected)
    query = base / 'query-two-panels.json'
    _write(query, {'partition': 'meta_train', 'blocks': [asdict(b) | {k: list(v) for k, v in vars(b).items() if isinstance(v, tuple)} for b in blocks],
                  'panels': panels, 'panel_targets': 32768, 'seeds': [4112, 4113],
                  'rule': 'PanelA excludes every full-continuation inner document; independently seeded PanelB additionally excludes PanelA documents. No panel outcome entered selection.'})
    initial.query_blocks = blocks
    initial.provenance['document_partitions'].update({b.article_hash: 'meta_train' for b in blocks})
    bootstrap = read(store / 'native-bootstrap-v2' / 'generation-1.json')
    epsilon = torch.randn(8, 45, generator=torch.Generator().manual_seed(4111))
    torch.manual_seed(4114)
    lr = freeze_controller(LearnedLR49())
    lr_epsilon = torch.randn(49, generator=torch.Generator().manual_seed(4115))
    design = {
        'schema': 'ralup.m11_final_calibration_design.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_identity': source_identity(), 'reference': _ref(reference_path), 'normalizer': reference['normalizer'],
        'reference_design': reference['design'], 'config': prior['config'], 'runtime': prior['runtime'],
        'queries': _ref(query), 'panels': panels, 'entries': entries,
        'query_provenance': {'manifest': manifest, 'partition': 'meta_train', 'panel_A_seed': 4112, 'panel_B_seed': 4113, 'full_inner_article_hashes': sorted(inner), 'panel_A_article_hashes': sorted({b.article_hash for b in blocks if b.document_id in panels['A']}), 'panel_B_article_hashes': sorted({b.article_hash for b in blocks if b.document_id in panels['B']})},
        'training_objective': 'One normalized future-loss change (mean_final_union - mean_initial_union) / max(mean_initial_union,1e-6) across the combined65536-target union at both endpoints; only this objective enters ES/outer Adam. Panels share one inner trajectory. Separate32768-panel normalized objectives and raw deltas are sensitivity statistics only, never separately optimized or independent trajectory replications.',
        'query_accounting': {'physical_rollouts': 35, 'combined_endpoint_evaluations': 70, 'panel_endpoint_reductions': 140, 'targets_per_combined_endpoint': 65536, 'initial_query_reuse': False},
        'initial_full_input_sha256': _input_digest(initial, 496),
        'pilot': {'parameters': bootstrap['phi_before'], 'perturbations': epsilon.tolist(), 'sigma': .025,
                  'pairs': 8, 'horizon': 16, 'outer_learning_rates': [.003, .01], 'perturbation_seed': 4111},
        'horizons': [16, 32, 64, 496], 'age_cases': ['baseline-256', 'baseline-448'],
        'lr49': {'parameters': lr.parameters.tolist(), 'epsilon': lr_epsilon.tolist(), 'sigma': .025, 'horizon': 16,
                 'initialization_seed': 4114, 'perturbation_seed': 4115},
        'maximum_optimizer_steps': 2192, 'maximum_training_targets': 2192 * 8192,
        'maximum_query_targets': 35 * 2 * 65536,
        'forecast_seconds': 2400, 'hard_deadline_seconds': 2460, 'forecast_bytes': 2 * 1024 ** 3,
        'forecast_basis': 'Previous3040updates55resets/queries2083.14summed rollout seconds;2192updates35resets plus two32768panels and retained-offset IO conservatively2400s. Each panel reuses the SAME physical rollout, no duplicate inner-training cost.',
        'risk_scope': 'Changed final normalizer and broad-age current complete pool, corrected full-owner clone and telemetry. New8pair sigma.025 signal, union/panel objectives, one pilot update at two step sizes, affected K/full ranking and age coverage, actual49network perturbations. No historical numerical-kernel Cartesian rerun.',
        'interpretation': 'Report independent direction halves separately from nestedm4/m8 and query-panel sensitivity. Two query panels do not establish convergence. Smaller sigma and step size are measured candidates, not accepted fixes. Final m/outerLR/iterations remain a subsequent evidence/cost decision.',
        'eligibility': 'All pilot/calibration vectors ineligible for final selection. Common calibration information/costs retained for all final families, whose optimization starts from declared initialization.',
        'acceptance': {'initialized_h16_duplicate_objective_atol': 0., 'initialized_h16_final_hash': 'exact H07', 'finite': True},
        'retention': 'Existing corrected-source complete pool milestones retained; all new successful rollout final weights disposable and reconstructible. Retain vectors, queries, raw per-block/pair/step/offset/clipping receipts, failures and current producer archive.',
        'helpers': {str(p): _ref(p) for p in map(Path, ['docs/harness/m11-final-calibration.py', 'docs/harness/m11-attempt.py', 'docs/harness/m11-supervise.py'])},
        'confirmation_eligible': False,
    }
    _write(base / 'design.json', design)
    print(json.dumps(_ref(base / 'design.json')), flush=True)
else:
    design_path = Path(sys.argv[2])
    d = read(design_path)
    assert d['source_identity'] == source_identity()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0' and torch.backends.mps.is_available()
    assert platform.machine() == d['runtime']['architecture']
    assert subprocess.check_output(['sw_vers', '-productVersion'], text=True).strip() == d['runtime']['macOS']
    assert subprocess.check_output(['sw_vers', '-buildVersion'], text=True).strip() == d['runtime']['build']
    assert sha256_file('requirements.lock') == d['runtime']['lock_sha256']
    for ref in d['helpers'].values():
        assert sha256_file(ref['path']) == ref['sha256']
    assert sha256_file(d['queries']['path']) == d['queries']['sha256']
    q = read(d['queries']['path'])
    blocks = [TargetBlock(**{k: tuple(v) if isinstance(v, list) else v for k, v in b.items()}) for b in q['blocks']]
    started = time.perf_counter()
    report = {'design': _ref(design_path), 'status': 'running', 'runs': [], 'source_identity': source_identity(), 'confirmation_eligible': False}
    try:
        task = materialize(d['entries']['baseline-16'], blocks)
        assert _input_digest(task, 496) == d['initial_full_input_sha256']
        with _Sampler('mps', _limits(d['config']['resolved'])) as sampler:
            def cap():
                sampler.sample()
                if sampler.exceeded or time.perf_counter() - started > d['forecast_seconds']:
                    raise RuntimeError('Frozen final-normalizer assessment envelope exhausted')

            def execute(name, selected, family, parameters, horizon):
                cap()
                journal = base / (name + '-journal.jsonl')
                selected.provenance.update(journal=str(journal), deadline_monotonic=started + d['forecast_seconds'])
                result = rollout(selected, ControllerState(family, torch.tensor(parameters)), horizon)
                path = base / (name + '.json')
                _write(path, {'name': name, 'family': family, 'parameters': parameters, 'horizon': horizon,
                             'result': asdict(result), 'panels': panel_objectives(journal, d['panels'])})
                report['runs'].append(_ref(path))
                print(json.dumps({'run': name, 'objective': result.objective, 'final': result.final_query, 'elapsed_seconds': result.work['elapsed_seconds']}), flush=True)
                return result

            pilot = d['pilot']
            task.provenance.update(journal=str(base / 'pilot-journal.jsonl'), deadline_monotonic=started + d['forecast_seconds'])
            outer = OuterAdam(torch.tensor(pilot['parameters']), lr=pilot['outer_learning_rates'][0])
            generation = train_generation(task, 'affine45', outer, torch.tensor(pilot['perturbations']), pilot['horizon'], pilot['sigma'])
            generation['panels'] = panel_objectives(base / 'pilot-journal.jsonl', d['panels'])
            _write(base / 'pilot-generation.json', generation)
            report['pilot'] = _ref(base / 'pilot-generation.json')
            other = OuterAdam(torch.tensor(pilot['parameters']), lr=pilot['outer_learning_rates'][1])
            other.step(torch.tensor(generation['gradient']))
            vectors = {'initialized': pilot['parameters'], 'outer_lr003': outer.phi.tolist(), 'outer_lr01': other.phi.tolist()}
            _write(base / 'vectors.json', vectors)
            for label, parameters in vectors.items():
                for horizon in d['horizons']:
                    execute(f'{label}-h{horizon}', task, 'affine45', parameters, horizon)
            duplicate = execute('initialized-h16-duplicate', task, 'affine45', vectors['initialized'], 16)
            first = read(base / 'initialized-h16.json')['result']
            assert duplicate.objective == first['objective'] and duplicate.work['final_state_hash'] == first['work']['final_state_hash']
            for name in d['age_cases']:
                selected = materialize(d['entries'][name], blocks)
                for label in ['initialized', 'outer_lr003']:
                    execute(f'{name}-{label}', selected, 'affine45', vectors[label], 16)
            for sign in [1, -1]:
                parameters = (torch.tensor(d['lr49']['parameters']) + sign * d['lr49']['sigma'] * torch.tensor(d['lr49']['epsilon'])).tolist()
                execute(f'lr49-neural-{sign}', task, 'learned_lr49', parameters, 16)
            report['memory'] = memory_report(sampler.samples)
            report['status'] = 'final_normalizer_assessment_measured_pending_budget'
    except BaseException as error:
        report.update(status='failed', failure=str(error), unknown_partial_work=True)
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter() - started
        _write(base / 'report.json', report)
