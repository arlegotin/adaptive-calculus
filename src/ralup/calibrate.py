"""Bounded engineering calibration and authenticated model-only warm starts.

No duration, B0, convergence, final gate or protected score is selected here.
Complete H06 transactions are primary timings; H07 owns archival authentication.
"""
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

import torch

from .config import ResolvedConfig, canonical_bytes, resolve_config
from .data.articles import Article
from .data.batches import next_logical_batch
from .data.cursor import BatchCursor
from .evaluate import score_blocks
from .models.transformer import ByteTransformer, ModelConfig
from .optim.adamw import AdamWUpdateRule
from .optim.base import MethodAdapter, SchedulePoint, default_parameter_groups
from .records import ArtifactRef, RunLedger, WorkLedger, sha256_file
from .state import (IntegrityError, _validated, _validate_model_owners, _automatic_identities, capture_snapshot, exact_cpu_state_hash,
                    load_snapshot, restore_snapshot, save_snapshot)
from .train import TrainingFailure, TrainingState, _clone, _restore_rng, _synchronize, train_step, select_logical_batch


PARTITIONS = {'meta_train': 'meta-train', 'target_train': 'target-training'}


def memory_limits(physical_bytes, framework_bytes=None):
    reserve = max(8 * 2**30, physical_bytes // 4)
    return {'physical_bytes': physical_bytes, 'os_reserve_bytes': reserve,
            'resident_cap_bytes': max(0, physical_bytes - reserve),
            'driver_cap_bytes': None if framework_bytes is None else int(.8 * framework_bytes),
            'framework_recommended_bytes': framework_bytes}


def memory_report(samples):
    return {'sampled_maxima': {key: max((s[key] for s in samples if s[key] is not None), default=None)
                              for key in ('rss_bytes', 'metal_bytes', 'driver_bytes')},
            'sampling_interval_seconds': .05, 'driver_peak_bytes': None,
            'unique_total_bytes': None, 'scope': 'sampled counters overlap; true Metal/driver peaks unavailable'}


def _rss():
    if sys.platform == 'darwin':
        buffer = ctypes.create_string_buffer(256)
        count = ctypes.CDLL('/usr/lib/libproc.dylib').proc_pidinfo(os.getpid(), 4, 0, buffer, 256)
        if count >= 16:
            return int.from_bytes(buffer.raw[8:16], sys.byteorder)
    return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())])) * 1024


def _memory(device):
    return {'rss_bytes': _rss(),
            'metal_bytes': torch.mps.current_allocated_memory() if device == 'mps' else None,
            'driver_bytes': torch.mps.driver_allocated_memory() if device == 'mps' else None}


class _Sampler:
    def __init__(self, device, caps):
        self.device, self.caps = device, caps
        self.samples, self.exceeded = [], False
        self.stop = threading.Event()

    def sample(self):
        row = _memory(self.device)
        self.samples.append(row)
        self.exceeded |= row['rss_bytes'] > self.caps['resident_cap_bytes']
        self.exceeded |= (row['driver_bytes'] is not None and
                          row['driver_bytes'] > self.caps['driver_cap_bytes'])

    def __enter__(self):
        self.sample()
        def poll():
            while not self.stop.wait(.05): self.sample()
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join(); self.sample()


def fit_memory(rows):
    sizes = sorted({row['microbatch'] for row in rows})
    if len(sizes) < 2: raise ValueError('Memory fitting requires two measured microbatch sizes')
    result = {}
    for key in ('rss_bytes', 'metal_bytes', 'driver_bytes'):
        points = [(size, max((r[key] for r in rows if r['microbatch'] == size and r.get(key) is not None), default=None))
                  for size in sizes]
        points = [(x, y) for x, y in points if y is not None]
        if len(points) < 2:
            result[key] = None
            continue
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        slope = sum((x-mean_x)*(y-mean_y) for x, y in points) / sum((x-mean_x)**2 for x, _ in points)
        intercept = mean_y - slope * mean_x
        result[key] = {'slope_bytes_per_block': slope, 'intercept_bytes': intercept,
                       'max_residual_bytes': max(abs(y-intercept-slope*x) for x, y in points),
                       'measured_sizes': [x for x, _ in points], 'context_bytes': 256,
                       'extrapolation': 'local fit only; reference attention scales quadratically with context'}
    return result


def duration_jobs(initial_targets, doublings):
    if type(initial_targets) is not int or initial_targets <= 0 or type(doublings) is not int or not 1 <= doublings <= 16:
        raise ValueError('Positive initial bytes and 1..16 duration jobs required')
    return [{'target_budget': initial_targets * 2**i, 'b0': None, 'selection': 'pending M07 development'}
            for i in range(doublings)]


def audit_queries(passes, queries):
    if passes['query'] != sum(q['targets'] for q in queries):
        raise ValueError('uncharged extra queries or inconsistent query ledger')


def reconcile_cost_scope(report):
    """Project-global union, retaining true cursor-local trajectory accounting."""
    result = json.loads(canonical_bytes(report))
    intervals = {}
    attempts = result['cost']['warmup_attempts'] + [a for arm in result['arms'] for a in arm['attempts']]
    query_offsets = []
    for query in result['queries']:
        if 'target_offsets' in query:
            query_offsets.extend(query['target_offsets'])
        elif query.get('partition') != 'failed_attempt':
            # Historical H08 queries cloned the original initial cursor and used
            # the same logical byte budget as the authenticated warmup attempt.
            warmup = result['cost']['warmup_attempts'][0]['target_offsets']
            if sum(len(block['offsets']) for block in warmup) != query['targets']:
                raise IntegrityError('Historical query coverage cannot be authenticated')
            query_offsets.extend(warmup)
    blocks = [block for attempt in attempts for block in attempt['target_offsets']] + query_offsets
    for block in blocks:
        values = sorted(set(block['offsets']))
        if any(type(v) is not int or v < 0 for v in values): raise IntegrityError('Invalid scored source offset')
        spans = []
        for value in values:
            if spans and value == spans[-1][1]: spans[-1][1] += 1
            else: spans.append([value, value+1])
        combined = sorted(intervals.get(block['document_id'], []) + spans)
        merged = []
        for start, end in combined:
            if merged and start <= merged[-1][1]: merged[-1][1] = max(end, merged[-1][1])
            else: merged.append([start, end])
        intervals[block['document_id']] = merged
    coverage = {'intervals': intervals,
                'basis': 'H03 authenticated source/document identity plus raw scored-byte offsets; half-open intervals',
                'query_basis': 'explicit saved offsets or historical identical initial warmup/query cursor contract'}
    result['global_coverage'] = {**coverage, 'sha256': hashlib.sha256(canonical_bytes(coverage)).hexdigest()}
    result['cost']['trajectory_unique_sum'] = result['cost'].get('trajectory_unique_sum', result['cost']['stream']['unique'])
    result['cost']['trajectory_unique_scope'] = 'sum of cursor-unique increments within independent trajectories, including disposable warmup'
    result['cost']['stream']['unique'] = sum(end-start for spans in intervals.values() for start, end in spans)
    result['cost']['unique_scope'] = 'project-global scored source-byte union; RuntimeLedger rows stay cursor-local'
    return result


def reconcile_calibration(reference: ArtifactRef, output: Path) -> ArtifactRef:
    """Versioned aggregate correction without relabeling valid historical timings."""
    start = time.perf_counter()
    if sha256_file(reference.path) != reference.sha256: raise IntegrityError('Calibration report hash mismatch')
    report = reconcile_cost_scope(json.loads(reference.path.read_bytes()))
    report['aggregate_parent'] = {'path': str(reference.path), 'sha256': reference.sha256}
    report['aggregate_accounting_seconds'] = time.perf_counter()-start
    report['project_seconds'] += report['aggregate_accounting_seconds']
    ledger = RunLedger(reference.path.parent)
    row = {'step': 0, 'stream': 0, 'unique': 0, 'additional_loss': 0, 'loss_targets': 0, 'forward_context': 0,
           'backward': 0, 'query': 0, 'elapsed_seconds': report['aggregate_accounting_seconds'],
           'status': 'global_coverage_reconciliation'}
    ledger.record(row)
    report['cost']['elapsed']['seconds'] += row['elapsed_seconds']
    report['cost']['events'].append(row)
    publication_start = time.perf_counter()
    ref = _write(output, report)
    ledger.record({**row, 'elapsed_seconds': time.perf_counter()-publication_start,
                   'status': 'aggregate_publication_io'})
    return ref


def forecast_cost(measurements: dict, design: dict) -> dict:
    interval = [0., 0.]
    for name, count_name in (('step', 'steps'), ('query', 'queries'), ('reset', 'resets'), ('checkpoint', 'checkpoints')):
        count = design.get(count_name, 0)
        if type(count) is not int or count < 0: raise ValueError('Forecast counts must be nonnegative integers')
        values = measurements.get(name + '_seconds', [])
        if count and not values: raise ValueError(f'Missing measured {name} costs')
        if any(type(v) not in (int, float) or not 0 <= v < float('inf') for v in values):
            raise ValueError('Forecast requires finite nonnegative measurements')
        if count:
            interval[0] += min(values) * count
            interval[1] += max(values) * count
    return {'future_seconds_interval': interval,
            'prior_project_seconds': measurements.get('project_seconds', 0.) + measurements.get('parent_seconds', 0.),
            'retention_bytes': measurements['snapshot_bytes'] * design.get('retained', 2),
            'soap': 'remeasure actual state in M02/M07', 'b0': None,
            'scope': 'observed timing range; not a confidence interval or convergence forecast'}


def _offsets(batch):
    return [{'document_id': b.document_id, 'offsets': list(b.scored_offsets())} for b in batch.block_slices]


def step_with_oom_retry(state, logical_targets=8192):
    """H06 rolls back the full transaction before a smaller packing retry.

    Only a verified OOM cause permits clearing its terminal failure flag. Costs
    survive the rollback; the selected logical offsets must match on every retry.
    """
    attempts = []
    while True:
        batch = select_logical_batch(state, logical_targets)
        offsets = _offsets(batch)
        if attempts and offsets != attempts[0]['target_offsets']:
            raise IntegrityError('OOM retry changed logical selected byte stream')
        attempt = {'microbatch': state.microbatch, 'target_offsets': offsets}
        attempts.append(attempt)
        try:
            result = train_step(state, batch)
            attempt['status'] = 'completed'
            return result, attempts
        except TrainingFailure as error:
            attempt['status'], attempt['failure'] = 'failed', str(error)
            if not isinstance(error.__cause__, RuntimeError) or 'out of memory' not in str(error.__cause__).lower() or state.microbatch == 1:
                raise
            # train_step restored all scientific fields and recorded failed passes.
            if state._pending is not None or state._work is not None or state.method.proposals:
                raise IntegrityError('OOM rollback is incomplete') from error
            state.failed = False
            state.microbatch = max(1, state.microbatch // 2)
            gc.collect()
            if next(state.model.parameters()).device.type == 'mps': torch.mps.empty_cache()


def _timed(model, function):
    _synchronize(model)
    start = time.perf_counter()
    result = function()
    _synchronize(model)
    return result, time.perf_counter() - start


def _event(state, status, seconds, **work):
    state.ledger.record({'step': state.schedule['step'], 'stream': 0, 'unique': 0, 'loss_targets': 0,
                         'additional_loss': 0, 'backward': 0, 'query': 0, 'forward_context': 0,
                         'elapsed_seconds': seconds, 'status': status, **work})


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical_bytes(payload) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    return ArtifactRef(path, sha256_file(path))


def _system_observations():
    result = {}
    for name, command in (('pressure', ['vm_stat']), ('power', ['pmset', '-g', 'batt']), ('thermal', ['pmset', '-g', 'therm'])):
        if sys.platform == 'darwin':
            process = subprocess.run(command, capture_output=True, text=True)
            result[name] = {'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}
        else: result[name] = {'status': 'unavailable'}
    return result


def _inventory(snapshot):
    owners = {}
    def visit(value, path):
        if not isinstance(value, dict): return
        if value.get('type') == 'tensor': owners[value['key']] = path
        for name, item in value.get('items', value.get('fields', {})).items() if isinstance(value.get('items', value.get('fields', {})), dict) else enumerate(value.get('items', [])):
            visit(item, f'{path}.{name}')
        if 'state' in value: visit(value['state'], path+'.rng_state')
    visit(snapshot.metadata, 'state')
    return [{'key': key, 'owner': owners.get(key), 'shape': list(value.shape), 'dtype': str(value.dtype),
             'bytes': value.numel()*value.element_size(), 'device': 'cpu snapshot; runtime tensors separately reside on configured device'}
            for key, value in sorted(snapshot.tensors.items())]


def _limits(payload):
    environment = payload['runtime']['environment']
    physical = json.loads(Path(environment['path']).read_bytes())['physical_memory_bytes'] if environment else 36*2**30
    caps = memory_limits(physical, torch.mps.recommended_max_memory() if payload['runtime']['device'] == 'mps' else None)
    # Existing paging tightens conservative caps; framework limits stay enabled.
    if sys.platform == 'darwin':
        observed = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True)
        caps['swap_observation'] = observed.stdout.strip() if observed.returncode == 0 else None
        if observed.returncode == 0 and 'used = 0.00M' not in observed.stdout:
            caps['resident_cap_bytes'] = int(.8*caps['resident_cap_bytes'])
            if caps['driver_cap_bytes'] is not None: caps['driver_cap_bytes'] = int(.8*caps['driver_cap_bytes'])
            caps['tightening'] = '0.8 for existing paging; conservative engineering policy'
    return caps


def measure_capacity(state, microbatches, *, repeats=3, logical_targets=8192, output, caps=None, recipes=None, cold_process=False):
    """Disposable warmup and randomized sustained arms, preserving caller state."""
    if (not microbatches or microbatches[0] != 1 or sorted(set(microbatches)) != microbatches
            or any(type(v) is not int or v <= 0 for v in microbatches)):
        raise ValueError('Microbatch search must increase from one without duplicates')
    if type(repeats) is not int or repeats < 2: raise ValueError('Sustained timing needs at least two steps per arm')
    device = next(state.model.parameters()).device.type
    caps = caps or memory_limits(36 * 2**30, torch.mps.recommended_max_memory() if device == 'mps' else None)
    output = Path(output)
    start = time.perf_counter()
    observations = {'before': _system_observations()}
    original_hash = exact_cpu_state_hash(state)
    snapshot, capture_seconds = _timed(state.model, lambda: capture_snapshot(state))
    tensor_bytes = sum(t.numel()*t.element_size() for t in snapshot.tensors.values())
    forecast_bytes = 4 * (tensor_bytes + len(canonical_bytes(snapshot.metadata)) + sum(map(len, snapshot.producer_files.values())))
    output.mkdir(parents=True, exist_ok=True)
    storage = shutil.disk_usage(output)
    if storage.free < forecast_bytes + 256 * 2**20: raise ValueError('Insufficient storage for measured snapshot and retention')
    cost = WorkLedger(state.ledger.run_ledger)
    reset_times, step_times, query_times, checkpoint_times = [], [], [], []
    _event(TrainingState(state.model, state.method, ledger=cost), 'snapshot_capture', capture_seconds)
    ref, seconds = _timed(state.model, lambda: save_snapshot(snapshot, output / 'initial'))
    checkpoint_times.append(seconds)
    _event(TrainingState(state.model, state.method, ledger=cost), 'checkpoint_write', seconds)
    loaded, seconds = _timed(state.model, lambda: load_snapshot(ref))
    checkpoint_times.append(seconds)
    _event(TrainingState(state.model, state.method, ledger=cost), 'checkpoint_load_first_cache_uncontrolled', seconds)
    _, seconds = _timed(state.model, lambda: load_snapshot(ref))
    checkpoint_times.append(seconds)
    _event(TrainingState(state.model, state.method, ledger=cost), 'checkpoint_load_warm', seconds)

    def clone():
        value, duration = _timed(state.model, lambda: restore_snapshot(loaded, device))
        reset_times.append(duration)
        value.ledger = cost
        _event(value, 'reset_transfer', duration)
        return value

    disposable = clone()
    with _Sampler(device, caps) as warmup_sampler:
        if warmup_sampler.exceeded: raise RuntimeError('Initial state exceeds conservative memory cap')
        warmup, warmup_seconds = _timed(disposable.model, lambda: step_with_oom_retry(disposable, logical_targets))
    if warmup_sampler.exceeded: raise RuntimeError('Warmup exceeds conservative memory cap')
    complete, duration = _timed(disposable.model, lambda: capture_snapshot(disposable))
    _event(disposable, 'completed_state_snapshot_transfer', duration)
    warmup_inventory = _inventory(complete)
    complete_bytes = sum(t.numel()*t.element_size() for t in complete.tensors.values()) + len(canonical_bytes(complete.metadata)) + sum(map(len, complete.producer_files.values()))
    if shutil.disk_usage(output).free < 2*complete_bytes + 256*2**20:
        raise ValueError('Insufficient storage for actual completed-state snapshot and retention')
    complete_ref, duration = _timed(disposable.model, lambda: save_snapshot(complete, output/'completed-state'))
    checkpoint_times.append(duration)
    _event(disposable, 'completed_state_checkpoint_write', duration)
    cold_observation = None
    if cold_process:
        code = "import json,sys,time; from ralup.state import load_snapshot; from ralup.records import ArtifactRef; start=time.perf_counter(); load_snapshot(ArtifactRef(sys.argv[1],sys.argv[2])); print(json.dumps({'authentication_seconds':time.perf_counter()-start}))"
        process, duration = _timed(disposable.model, lambda: subprocess.run([sys.executable, '-c', code, str(complete_ref.path), complete_ref.sha256], capture_output=True, text=True, check=True))
        cold_observation = {**json.loads(process.stdout), 'interpreter_wall_seconds': duration, 'filesystem_cache': 'uncontrolled; new interpreter, no cache purge'}
        checkpoint_times.append(duration)
        _event(disposable, 'cold_interpreter_checkpoint_load', duration)
    complete_loaded, duration = _timed(disposable.model, lambda: load_snapshot(complete_ref))
    checkpoint_times.append(duration)
    _event(disposable, 'completed_state_checkpoint_load_warm', duration)
    _, duration = _timed(disposable.model, lambda: restore_snapshot(complete_loaded, device))
    reset_times.append(duration)
    _event(disposable, 'completed_state_restore_transfer', duration)
    del disposable
    gc.collect()
    _restore_rng(_validated(snapshot)['global_rng'])
    post_warmup_hash = exact_cpu_state_hash(state)
    if post_warmup_hash != original_hash: raise IntegrityError('Disposable warmup changed real initial state')
    arms, memory_rows = [], []
    order_rng = random.Random(808)
    if 'timing' in state.rng_streams:
        order_rng.setstate(state.rng_streams['timing'].getstate())
    selected = 1
    previous_rate = None
    outcome = 'measured_bounded_capacity'
    def measure_arm(size, recipe, *, remeasure=False):
        arm = clone()
        arm.microbatch = size
        arm.model.configure_quantization({**arm.model._recipe, 'name': recipe})
        timings, attempts = [], []
        with _Sampler(device, caps) as sampler:
            for _ in range(repeats):
                if sampler.exceeded: break
                (_, retries), duration = _timed(arm.model, lambda: step_with_oom_retry(arm, logical_targets))
                timings.append(duration); step_times.append(duration); attempts.extend(retries)
        memory = memory_report(sampler.samples)
        if arm.microbatch == size:
            memory_rows.append({'microbatch': size, **memory['sampled_maxima']})
        arms.append({'microbatch_requested': size, 'microbatch_actual': arm.microbatch, 'recipe': recipe,
                     'full_step_seconds': timings, 'memory': memory, 'attempts': attempts,
                     'oom_common_packing_remeasurement': remeasure, 'memory_cap_exceeded': sampler.exceeded})
        actual = arm.microbatch
        del arm
        gc.collect()
        rate = logical_targets * len(timings) / sum(timings) if timings else 0.
        return actual, sampler.exceeded, rate

    for size in microbatches:
        order = list(recipes or [state.model._recipe['name']])
        order_rng.shuffle(order)
        candidate_rates = []
        for recipe in order:
            actual, exceeded, rate = measure_arm(size, recipe)
            candidate_rates.append(rate)
            if exceeded:
                outcome = 'resource_limited_memory'; break
            if actual != size:
                outcome = 'resource_limited_oom_packing'
                selected = actual
                # All comparison arms use the same reduced packing and stream.
                for common_recipe in order:
                    common_actual, common_exceeded, _ = measure_arm(actual, common_recipe, remeasure=True)
                    if common_actual != actual or common_exceeded:
                        raise RuntimeError('Common OOM packing remeasurement failed its memory cap')
                break
        if outcome != 'measured_bounded_capacity': break
        rate = min(candidate_rates)
        if previous_rate is not None and rate <= previous_rate: break
        selected, previous_rate = size, rate
    query_state = clone()
    query_cursor = BatchCursor(_validated(snapshot)['cursor'])
    query = next_logical_batch(query_cursor, logical_targets)
    # Charge launched query work even if scoring fails.
    query_start = time.perf_counter()
    try:
        _, seconds = _timed(query_state.model, lambda: score_blocks(query_state.model, query.block_slices,
                           evaluation_context={'partition': 'fixture' if query_cursor._source['kind'] == 'articles' else query_cursor._source['partition'], 'seed': 808}))
    finally:
        seconds = time.perf_counter() - query_start
        query_times.append(seconds)
        _event(query_state, 'query', seconds, additional_loss=query.target_count, loss_targets=query.target_count,
               query=query.target_count, forward_context=sum(len(b.inputs) for b in query.block_slices))
    queries = [{'targets': query.target_count, 'partition': query_cursor._source.get('partition', 'fixture'), 'seconds': seconds, 'target_offsets': _offsets(query)}]
    queries += [{'targets': event['query'], 'partition': 'failed_attempt', 'seconds': event['elapsed_seconds']}
                for event in cost.events if event['status'] == 'failed' and event['query']]
    audit_queries(cost.passes, queries)
    _restore_rng(_validated(snapshot)['global_rng'])
    observations['after'] = _system_observations()
    initial_bytes = sum(p.stat().st_size for p in ref.path.parent.rglob('*') if p.is_file())
    actual_bytes = sum(p.stat().st_size for p in complete_ref.path.parent.rglob('*') if p.is_file())
    report = {'schema_version': 1, 'scope': 'H08 engineering pilot only', 'outcome': outcome, 'b0': None,
            'original_initial_hash': original_hash, 'post_warmup_initial_hash': post_warmup_hash,
            'caps': caps, 'system': observations, 'storage': {'free_before_bytes': storage.free, 'prewrite_forecast_bytes': forecast_bytes,
            'actual_initial_generation_bytes': initial_bytes, 'actual_completed_generation_bytes': actual_bytes, 'retained_generations': 2, 'no_user_files_deleted': True},
            'inventory_initial': _inventory(snapshot), 'inventory_after_warmup': warmup_inventory,
            'parameter_floor_bytes': 16*sum(p.numel() for p in state.model.parameters()),
            'arms': arms, 'selected_packing': selected, 'fit': fit_memory(memory_rows) if len({r['microbatch'] for r in memory_rows}) >= 2 else None,
            'step_seconds': step_times, 'reset_seconds': reset_times, 'query_seconds': query_times,
            'checkpoint_seconds': checkpoint_times, 'snapshot_bytes': actual_bytes, 'cold_load': cold_observation, 'completed_snapshot': {'path': str(complete_ref.path), 'sha256': complete_ref.sha256}, 'queries': queries,
            'cost': {'stream': cost.stream, 'passes': cost.passes, 'elapsed': cost.elapsed, 'events': cost.events,
                     'warmup_seconds': warmup_seconds, 'warmup_attempts': warmup[1], 'warmup_memory': memory_report(warmup_sampler.samples)},
            'project_seconds': time.perf_counter()-start, 'snapshot': {'path': str(ref.path), 'sha256': ref.sha256},
            'soap': 'remeasure actual state in M02/M07', 'peak_scope': 'sampled maxima only; no unique RSS+Metal sum'}
    accounting_start = time.perf_counter()
    report = reconcile_cost_scope(report)
    _event(query_state, 'global_coverage_accounting', time.perf_counter()-accounting_start)
    report['cost']['elapsed'] = dict(cost.elapsed)
    report['cost']['events'] = list(cost.events)
    report['project_seconds'] = time.perf_counter()-start
    return report


def _resolved(config):
    if type(config) is not ResolvedConfig: raise TypeError('Expected ResolvedConfig')
    payload = json.loads(canonical_bytes(config.payload))
    purpose = payload.pop('purpose'); gates = payload.pop('evidence_gates')
    validated = resolve_config(payload, {'gates': gates}, purpose=purpose)
    if validated.sha256 != config.sha256 or purpose != 'calibration': raise ValueError('Explicit calibration purpose required')
    return json.loads(canonical_bytes(config.payload))


def _create(config, partition, budget, seed):
    payload = _resolved(config)
    if partition not in PARTITIONS: raise ValueError('Warmstart partition must be meta_train or target_train')
    device = payload['runtime']['device']
    if device == 'mps' and (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0' or not torch.backends.mps.is_available()):
        raise ValueError('Actual MPS requires fallback zero before startup')
    if payload['synthetic']:
        raw = (b'H08 bounded synthetic ' + partition.encode() + b'\n') * 400
        cursor = BatchCursor.from_articles([Article(raw, [], hashlib.sha256(raw).hexdigest())], seed=seed, run_target_budget=budget)
    else:
        manifest = payload['data']['manifest']
        if json.loads(Path(manifest['path']).read_bytes())['revision'] != payload['data']['revision']:
            raise ValueError('Calibration source revision mismatch')
        cursor = BatchCursor.from_manifest(Path(manifest['path']), manifest['sha256'], partition=PARTITIONS[partition], seed=seed, run_target_budget=budget)
    initialization = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng():
        torch.set_rng_state(initialization.get_state())
        model = ByteTransformer(ModelConfig(**{k: payload['model'][k] for k in ('layers', 'width', 'heads')})).to(device)
        initialization.set_state(torch.get_rng_state())
    model.configure_quantization({'name': payload['training']['recipe'],
                                  'rotation': 'full_hadamard' if payload['quantization']['rotation'] == 'hadamard' else 'identity'})
    probe = payload['optimizer_probe']
    method = MethodAdapter(dict(model.named_parameters()), default_parameter_groups(model),
                           AdamWUpdateRule(betas=tuple(probe['betas']), eps=probe['epsilon']))
    stream_seed = lambda name: int.from_bytes(hashlib.sha256(f'ralup-rng-v1:{seed}:{name}'.encode()).digest()[:8], 'little')
    return TrainingState(model, method, cursor=cursor, microbatch=1,
            schedule={'step': 0, 'point': SchedulePoint(probe['learning_rate'], probe['decay'], 0.)},
            rng_streams={'initialization': initialization, 'quantizer': torch.Generator(device=device).manual_seed(stream_seed('quantization')),
                         'meta': random.Random(stream_seed('meta')), 'timing': random.Random(stream_seed('timing'))},
            configuration={'resolved': payload, 'source_partition': partition, 'seed': seed, 'pilot_target_budget': budget})


def _run_ledger(config, output, seed, partition):
    payload = json.loads(canonical_bytes(config.payload))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    diff = subprocess.check_output(['git', 'diff', 'HEAD'])
    artifact_hash = lambda name: None if payload['runtime'][name] is None else payload['runtime'][name]['sha256']
    return RunLedger(output, {'run_id': 'h08-'+uuid.uuid4().hex, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'local_timezone': 'Europe/Amsterdam', 'phase': 'smoke' if payload['synthetic'] else 'explore',
        'source': {'commit': commit, 'dirty_diff_sha256': hashlib.sha256(diff).hexdigest() if diff else None},
        'lock_sha256': artifact_hash('dependency_lock'), 'environment_sha256': artifact_hash('environment'),
        'resolved_config': payload, 'resolved_config_sha256': config.sha256, 'method': 'AdamW', 'adaptation': 'H08 calibration',
        'seeds': {'initialization': seed, 'data': seed}, 'data_sha256': None if payload['synthetic'] else payload['data']['manifest']['sha256'],
        'warm_start_sha256': None, 'policy_sha256': None, 'lineage': {'parent': None, 'resume': None},
        'endpoint': {'scope': 'engineering pilot', 'partition': partition, 'b0': None, 'cost_scope': 'project; warmstarts additionally preserved as common deployment ancestry'}})


def calibrate(config: ResolvedConfig, microbatches: list[int]) -> ArtifactRef:
    start = time.perf_counter()
    payload = _resolved(config)
    settings = payload['calibration']
    output = Path(settings['output']) / 'capacity'
    state = _create(config, 'target_train', settings['pilot_target_budget'], settings['seed'])
    state.ledger.run_ledger = _run_ledger(config, output, settings['seed'], 'target_train')
    ledger = state.ledger.run_ledger
    ledger.record_status('running')
    caps = _limits(payload)
    try:
        logical_targets = min(8192, settings['pilot_target_budget']//settings['repeats']) if payload['synthetic'] else 8192
        report = measure_capacity(state, microbatches, repeats=settings['repeats'], logical_targets=logical_targets, output=output, caps=caps,
                                  recipes=['row_fixed_max', 'row_dynamic_max', 'quest_fp4_row_reference'], cold_process=True)
        report['project_seconds'] = time.perf_counter()-start
        report['configuration'] = json.loads(canonical_bytes(config.payload))
        report['source_partition'] = 'target_train'
        report['synthetic'] = payload['synthetic']
        report['duration_jobs'] = duration_jobs(settings['warmstart_target_budget'], settings['doublings'])
        report['forecast'] = forecast_cost(report, {'steps': settings['repeats'], 'queries': 1, 'resets': 1, 'checkpoints': 1, 'retained': 2})
        ref = _write(output / 'calibration.json', report)
        ledger.record_status('completed', disposition=report['outcome'])
        return ref
    except Exception as error:
        ledger.record_status('failed', disposition=str(error)); raise
    finally:
        ledger.record({'step': 0, 'stream': 0, 'unique': 0, 'additional_loss': 0, 'backward': 0, 'query': 0,
                       'elapsed_seconds': 0., 'status': 'project_elapsed_observation', 'project_elapsed_seconds': time.perf_counter()-start})


def train_warmstart(config: ResolvedConfig, partition: str, seed: int) -> ArtifactRef:
    start = time.perf_counter()
    payload = _resolved(config); settings = payload['calibration']
    output = Path(settings['output']) / f'warmstart-{partition}-{seed}'
    state = _create(config, partition, settings['warmstart_target_budget'], seed)
    state.configuration['artifact_role'] = 'warmstart'
    ledger = _run_ledger(config, output, seed, partition); state.ledger.run_ledger = ledger
    ledger.record_status('running')
    try:
        with _Sampler(payload['runtime']['device'], _limits(payload)) as sampler:
            def check_cap(boundary):
                started = time.perf_counter()
                sampler.sample()
                if sampler.exceeded:
                    _event(state, 'memory_cap_exceeded', time.perf_counter()-started,
                           boundary=boundary, memory=memory_report(sampler.samples), caps=sampler.caps)
                    raise RuntimeError('Warmstart exceeded conservative memory cap')

            check_cap('admission')
            while state.cursor.remaining_run_targets:
                check_cap('before_transaction')
                step_with_oom_retry(state)
                check_cap('completed_transaction')
            check_cap('before_snapshot')
            snapshot, seconds = _timed(state.model, lambda: capture_snapshot(state))
            _event(state, 'snapshot_capture', seconds)
            check_cap('after_snapshot')
            bytes_needed = 2*(sum(t.numel()*t.element_size() for t in snapshot.tensors.values()) + len(canonical_bytes(snapshot.metadata)) + sum(map(len, snapshot.producer_files.values())))
            if shutil.disk_usage(output).free < bytes_needed + 256*2**20: raise ValueError('Insufficient storage for warmstart retention')
            def publish_checkpoint():
                check_cap('before_checkpoint_capture')
                current = capture_snapshot(state)
                check_cap('before_checkpoint_write')
                return save_snapshot(current, output/'checkpoint')
            checkpoint, seconds = _timed(state.model, publish_checkpoint)
            _event(state, 'checkpoint_write', seconds)
        check_cap('before_warmstart_report')
        report = {'schema_version': 1, 'scope': 'immutable H08 engineering warmstart; duration selection pending M07',
                  'source_partition': partition, 'synthetic': payload['synthetic'],
                  'checkpoint': {'path': str(checkpoint.path), 'sha256': checkpoint.sha256},
                  'cursor_contract': 'preserve source/position/carry/permutation/data RNG; extend only final byte envelope by declared continuation bytes',
                  'resets': ['optimizer', 'quantizer', 'history', 'schedule'], 'b0': None,
                  'cost': {'stream': state.ledger.stream, 'passes': state.ledger.passes, 'elapsed': state.ledger.elapsed, 'events': state.ledger.events},
                  'project_seconds': time.perf_counter()-start, 'storage_required_bytes': bytes_needed,
                  'memory': memory_report(sampler.samples), 'caps': sampler.caps,
                  'configuration_sha256': config.sha256}
        ref = _write(output/'warmstart.json', report)
        ledger.record_status('completed', disposition='duration_not_selected')
        return ref
    except Exception as error:
        ledger.record_status('failed', disposition=str(error)); raise
    finally:
        ledger.record({'step': state.schedule['step'], 'stream': 0, 'unique': 0, 'additional_loss': 0, 'backward': 0, 'query': 0,
                       'elapsed_seconds': 0., 'status': 'project_elapsed_observation', 'project_elapsed_seconds': time.perf_counter()-start})


def import_warmstart(reference: ArtifactRef, fresh: TrainingState, *, partition: str, continuation_targets: int) -> TrainingState:
    """Restricted historical import; full restore_snapshot compatibility stays strict.

    The checkpoint is authenticated in full, including archived producer bytes,
    tensor manifests, model parameterization/dtype and saved cursor provenance.
    Constructor/optimizer/quantizer/history/schedule are from the fresh runtime.
    Cursor, named/global RNG and all parent costs retain their historical identity.
    """
    original_reference = reference
    parent_report = None
    if reference.path.name == 'warmstart.json':
        if sha256_file(reference.path) != reference.sha256:
            raise IntegrityError('Warmstart report hash mismatch')
        parent_report = json.loads(reference.path.read_bytes())
        if parent_report['source_partition'] != partition:
            raise IntegrityError('Wrong partition warmstart report ancestry')
        reference = ArtifactRef(**parent_report['checkpoint'])
    producer_manifest = json.loads(reference.path.read_bytes())
    snapshot = load_snapshot(reference)
    payload = _validated(snapshot)
    if 'resolved' in payload['configuration'] and (parent_report is None or payload['configuration'].get('artifact_role') != 'warmstart'):
        raise IntegrityError('Generated warmstart requires its cost-complete warmstart artifact reference')
    if partition not in PARTITIONS or payload['configuration'].get('source_partition') != partition:
        raise IntegrityError('Wrong partition ancestry for warmstart')
    if type(continuation_targets) is not int or continuation_targets <= 0:
        raise IntegrityError('Continuation requires a declared positive byte envelope')
    if not fresh.checkpointable or fresh.schedule['step'] != 0 or fresh.histories or fresh.quantizer_state or any(fresh.optimizer_state.values()):
        raise IntegrityError('Warmstart import requires fresh reset state')
    if fresh.configuration.get('source_partition', partition) != partition:
        raise IntegrityError('Fresh continuation partition differs from parent')
    if fresh.model.config != payload['model_config'] or set(fresh.model.state_dict()) != set(payload['weights']):
        raise IntegrityError('Warmstart model parameterization mismatch')
    try:
        _validate_model_owners(fresh.model)
    except IntegrityError as error:
        raise IntegrityError(f'Warmstart model parameterization mismatch: {error}') from error
    current = _automatic_identities()
    model_names = ('src/ralup/models/transformer.py', 'src/ralup/models/linear.py', 'src/ralup/models/embedding.py')
    same_model = all(name in payload['identities']['automatic'] and
                     current[name].sha256 == payload['identities']['automatic'][name].sha256 for name in model_names)
    model_contract = 'identical_model_source'
    if not same_model:
        # Narrow N01 model-only migration: authenticate exact historical model
        # bytes. Only the imported owner and two embedding constructors differ.
        # Never execute archived Python or waive full-state strict runtime checks.
        old = snapshot.producer_files.get(model_names[0], b'')
        migrated = old.replace(b'from .linear import QuantizedLinear',
                              b'from .linear import QuantizedLinear\nfrom .embedding import OrderedEmbedding')
        migrated = migrated.replace(b'nn.Embedding(257,', b'OrderedEmbedding(257,')
        migrated = migrated.replace(b'nn.Embedding(256,', b'OrderedEmbedding(256,')
        if (model_names[2] in snapshot.producer_files or
                migrated != current[model_names[0]].path.read_bytes() or
                snapshot.producer_files.get(model_names[1]) != current[model_names[1]].path.read_bytes()):
            raise IntegrityError('Model parameterization producer differs; no model migration is registered')
        model_contract = 'n01_native_gather_identical_forward_model_only_v1'
    for name, parameter in fresh.model.state_dict().items():
        weight = payload['weights'][name]
        if parameter.shape != weight.shape or parameter.dtype != weight.dtype:
            raise IntegrityError('Warmstart model tensor/dtype mismatch')
    cursor = payload['cursor']
    if cursor is None or (cursor['source']['kind'] == 'indexed' and cursor['source']['partition'] != PARTITIONS[partition]):
        raise IntegrityError('Warmstart cursor partition mismatch')
    continuation = {**cursor, 'run_target_budget': cursor['cumulative_target_count'] + continuation_targets}
    fresh.cursor = BatchCursor(continuation)
    fresh.model.load_state_dict(payload['weights'], strict=True)
    fresh.model.zero_grad(set_to_none=True)
    fresh.model._runtime, fresh.model._contexts, fresh.model._context_step = {}, {}, None
    fresh.rng_streams = {}
    for name, rng in payload['rng_streams'].items():
        if rng['__rng_kind__'] == 'python_rng':
            generator = random.Random(); generator.setstate(rng['state'])
        else:
            generator = torch.Generator(device=rng['device']); generator.set_state(rng['state'])
        fresh.rng_streams[name] = generator
    for name, value in payload['ledger'].items(): setattr(fresh.ledger, name, _clone(value))
    if parent_report is not None:
        for name in ('stream', 'passes', 'elapsed', 'events'):
            setattr(fresh.ledger, name, _clone(parent_report['cost'][name]))
    fresh.configuration = {**fresh.configuration, 'warmstart_parent': {
        'checkpoint': {'path': str(reference.path), 'sha256': reference.sha256},
        'artifact': {'path': str(original_reference.path), 'sha256': original_reference.sha256},
        'parent_cost': parent_report if parent_report is not None else {'ledger': payload['ledger'], 'process_audit': payload['process_audit']},
        'source_identity': {name: {'original_path': str(ref.path), 'sha256': ref.sha256,
                                 'archive': {'path': str(reference.path.parent / producer_manifest['producer_files'][name]['path']), 'sha256': ref.sha256}}
                            for name, ref in payload['identities']['automatic'].items()},
        'configuration_sha256': payload['configuration_sha256'], 'source_partition': partition,
        'model_contract': model_contract,
        'continuation_contract': 'same cursor and all named/global RNG; declared budget extension only',
        'continuation_targets': continuation_targets, 'parent_cursor_sha256': hashlib.sha256(canonical_bytes(cursor)).hexdigest(),
        'resets': ['optimizer', 'quantizer', 'history', 'schedule'], 'parent_process_audit': payload['process_audit']}}
    _restore_rng(payload['global_rng'])
    return fresh
