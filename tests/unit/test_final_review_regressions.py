"""Finite synthetic regressions for the final integration findings; CPU only."""
import copy
from dataclasses import fields, is_dataclass
import functools
import hashlib
import json
from pathlib import Path
import random

import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    # CPU snapshots also inspect global RNG. Do not initialize a native RNG.
    monkeypatch.setattr(torch.backends.mps, 'is_available', functools.lru_cache(maxsize=None)(lambda: False))


def small_state(adapter='ordinary'):
    from ralup.models.transformer import ByteTransformer, ModelConfig
    from ralup.optim.adamw import AdamWUpdateRule
    from ralup.optim.base import MethodAdapter, default_parameter_groups, SchedulePoint
    from ralup.train import TrainingState
    from ralup.data.articles import Article
    from ralup.data.cursor import BatchCursor
    with torch.random.fork_rng():
        torch.manual_seed(193)
        model = ByteTransformer(ModelConfig(1, 8, 2))
    parameters = dict(model.named_parameters())
    groups = default_parameter_groups(model)
    recipe = {'name': 'row_fixed_max', 'rotation': 'identity', 'backward': 'ste'}
    if adapter == 'learned_scale':
        from ralup.baselines.learned_scale import LearnedScaleAdapter, FIXTURE_CONFIG
        method = LearnedScaleAdapter(FIXTURE_CONFIG).bind(parameters, groups)
        recipe['name'] = method.recipe
    elif adapter == 'metaquant':
        from ralup.baselines.metaquant import MetaQuantLMAdapter, FIXTURE_CONFIG
        method = MetaQuantLMAdapter(AdamWUpdateRule(), {**FIXTURE_CONFIG,
            'variant': 'online_target_training'}).bind(parameters, groups)
    else:
        method = MethodAdapter(parameters, groups, AdamWUpdateRule())
    model.configure_quantization(recipe)
    raw = b'abcdefgh'
    cursor = BatchCursor.from_articles([Article(raw, [], hashlib.sha256(raw).hexdigest())],
        seed=29, run_target_budget=32)
    return TrainingState(model, method, cursor=cursor,
        schedule={'step': 0, 'point': SchedulePoint(.002, .1, 0.)},
        rng_streams={'meta': random.Random(51), 'quantizer': torch.Generator().manual_seed(52)},
        configuration={'source_partition': 'meta_train'})


def assert_owner_equal(actual, expected):
    """Compare owners directly, including tensors excluded from replay hashes."""
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.device == expected.device
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_owner_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_owner_equal(left, right)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            assert_owner_equal(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, random.Random):
        assert actual.getstate() == expected.getstate()
    elif isinstance(expected, torch.Generator):
        assert torch.equal(actual.get_state(), expected.get_state())
    else:
        assert actual == expected


@pytest.mark.parametrize('adapter', ['ordinary', 'learned_scale'])
def test_complete_restore_rejects_changed_live_runtime_before_live_model(monkeypatch, adapter):
    import ralup.state as snapshots
    snapshot = snapshots.capture_snapshot(small_state(adapter))
    source_before = snapshots._automatic_identities()
    original = snapshots.ByteTransformer.__init__
    live_constructors = []
    def construct(self, *args, **kwargs):
        if torch.get_default_device().type != 'meta':
            live_constructors.append(True)
        original(self, *args, **kwargs)
    monkeypatch.setattr(snapshots.ByteTransformer, '__init__', construct)
    # Same files; change only the actual-runtime provider, never producer bytes.
    provider = getattr(snapshots, '_runtime_fingerprint', lambda backend: {})
    changed = provider('cpu')
    changed['python_version'] = 'injected different Python numerical runtime'
    monkeypatch.setattr(snapshots, '_runtime_fingerprint', lambda backend: changed, raising=False)
    with pytest.raises(snapshots.IntegrityError, match='runtime'):
        snapshots.restore_snapshot(snapshot, 'cpu')
    assert not live_constructors
    assert snapshots._automatic_identities() == source_before


@pytest.mark.parametrize('adapter', ['ordinary', 'learned_scale'])
def test_complete_restore_rejects_requested_backend_before_native_probe(monkeypatch, adapter):
    import ralup.state as snapshots
    snapshot = snapshots.capture_snapshot(small_state(adapter))
    def forbidden_probe():
        pytest.fail('Backend mismatch must reject before native availability probe')
    monkeypatch.setattr(torch.backends.mps, 'is_available', functools.lru_cache(maxsize=None)(forbidden_probe))
    with pytest.raises(snapshots.IntegrityError, match='backend'):
        snapshots.restore_snapshot(snapshot, 'mps')


@pytest.mark.parametrize('damage', ['absent', 'extra', 'wrong_type', 'producer_backend'])
def test_runtime_schema_and_historical_full_restore_fail_closed(damage):
    import ralup.state as snapshots
    snapshot = snapshots.capture_snapshot(small_state())
    items = snapshot.metadata['items']
    assert 'runtime' in items, 'Snapshot lacks actual numerical runtime proof'
    if damage == 'absent':
        del items['runtime']
        snapshots._validated(snapshot)  # Historical authenticated inspection remains supported.
    elif damage == 'extra':
        items['runtime']['items']['unregistered'] = True
    elif damage == 'wrong_type':
        items['runtime']['items']['python_version'] = 123
    else:
        items['runtime']['items']['backend'] = 'mps'
    with pytest.raises(snapshots.IntegrityError, match='runtime|backend'):
        snapshots.restore_snapshot(snapshot, 'cpu')


@pytest.mark.parametrize('adapter', ['ordinary', 'learned_scale'])
def test_live_runtime_same_backend_complete_roundtrip(adapter):
    from ralup.state import capture_snapshot, restore_snapshot, _validated
    from ralup.train import train_step, select_logical_batch
    state = small_state(adapter)
    train_step(state, select_logical_batch(state, 4))
    before = _validated(capture_snapshot(state))
    assert before['runtime']['backend'] == 'cpu'
    restored = restore_snapshot(capture_snapshot(state), 'cpu')
    after = _validated(capture_snapshot(restored))
    # Resume deliberately appends process audit; all numerical owners remain exact.
    before.pop('process_audit'); after.pop('process_audit')
    if adapter == 'learned_scale':
        prior_checks = before['method'].pop('scale_validation_telemetry')
        resumed_checks = after['method'].pop('scale_validation_telemetry')
        # Validation at capture/restore performs additional measured checks.
        # Preserve their real accounting rather than demand frozen audit costs.
        assert set(prior_checks) == set(resumed_checks) == {'checks', 'bytes_checked', 'seconds'}
        # One extra source capture, restore validation, and resumed capture;
        # the six matrices have 72 physical FP32 row scales (288 bytes/check).
        assert resumed_checks['checks'] - prior_checks['checks'] == 3
        assert resumed_checks['bytes_checked'] - prior_checks['bytes_checked'] == 864
        assert resumed_checks['seconds'] > prior_checks['seconds'] >= 0
    assert_owner_equal(after, before)


def test_historical_runtime_absence_allows_model_only_reset(tmp_path):
    from ralup.state import capture_snapshot, save_snapshot, load_snapshot, restore_snapshot, IntegrityError
    from ralup.calibrate import import_warmstart
    from ralup.train import train_step, select_logical_batch
    state = small_state()
    train_step(state, select_logical_batch(state, 4))
    snapshot = capture_snapshot(state)
    snapshot.metadata['items'].pop('runtime', None)
    reference = save_snapshot(snapshot, tmp_path/'historical')
    with pytest.raises(IntegrityError, match='runtime'):
        restore_snapshot(load_snapshot(reference), 'cpu')
    fresh = import_warmstart(reference, small_state(), partition='meta_train', continuation_targets=4)
    assert fresh.schedule['step'] == 0 and fresh.histories == {}
    assert all(not values for values in fresh.optimizer_state.values())
    for name, weight in state.model.named_parameters():
        assert torch.equal(dict(fresh.model.named_parameters())[name], weight)


def test_late_metaquant_failure_restores_every_owner_and_charges_actual_work(monkeypatch):
    import ralup.train as training
    from ralup.state import capture_snapshot, IncompleteStep
    state = small_state('metaquant')
    for _ in range(2):
        training.train_step(state, training.select_logical_batch(state, 4))
    method = state.method
    assert method.theta_updates == 1 and all(method.outer_states.values())
    parameter_ids = {name: id(p) for name, p in state.model.named_parameters()}
    theta_ids = {name: id(p) for name, p in method.theta.items()}
    weights = {name: p.detach().clone() for name, p in state.model.named_parameters()}
    gradients = {name: None if p.grad is None else p.grad.clone() for name, p in state.model.named_parameters()}
    method_before = training._clone({name: value for name, value in vars(method).items()
        if name not in ('parameters', 'groups', 'rule')})
    owners = training._clone({name: getattr(state, name) for name in
        ('histories', 'quantizer_state', 'schedule', 'rng_streams')})
    cursor_before = copy.deepcopy(state.cursor.state)
    cursor_id = id(state.cursor)
    rng = training._rng_state()
    ledger_before = copy.deepcopy(state.ledger.passes)
    stream_before = copy.deepcopy(state.ledger.stream)
    events_before = len(state.ledger.events)
    transition = training._transition
    changed = []
    def fail_after_transition(current, *args):
        assert current.method.theta_updates == method_before['theta_updates'] + 1
        assert any(not torch.equal(p, method_before['theta'][name]) for name, p in method.theta.items())
        assert any(not torch.equal(p, weights[name]) for name, p in state.model.named_parameters())
        transition(current, *args)
        assert current.histories['loss_updates'] == owners['histories']['loss_updates'] + 1
        random.random(); torch.rand(3)
        current.rng_streams['meta'].random()
        torch.rand(2, generator=current.rng_streams['quantizer'])
        changed.append(True)
        raise RuntimeError('injected after actual theta/W/history mutation')
    monkeypatch.setattr(training, '_transition', fail_after_transition)
    with pytest.raises(training.TrainingFailure, match='actual theta/W/history mutation'):
        training.train_step(state, training.select_logical_batch(state, 4))
    assert changed == [True]
    assert parameter_ids == {name: id(p) for name, p in state.model.named_parameters()}
    assert theta_ids == {name: id(p) for name, p in method.theta.items()}
    assert_owner_equal(dict(state.model.named_parameters()), weights)
    assert_owner_equal({name: p.grad for name, p in state.model.named_parameters()}, gradients)
    assert_owner_equal({name: getattr(method, name) for name in method_before}, method_before)
    for name, parameter in method.theta.items():
        assert parameter.grad is method.theta_gradients[name]
        assert_owner_equal(parameter.grad, method_before['theta_gradients'][name])
    assert state.optimizer_state is method.optimizer_states
    assert_owner_equal({name: getattr(state, name) for name in owners}, owners)
    assert id(state.cursor) == cursor_id and state.cursor.state == cursor_before
    assert_owner_equal(training._rng_state(), rng)
    assert state.ledger.stream == stream_before
    assert len(state.ledger.events) == events_before + 1
    failed = state.ledger.events[-1]
    assert failed['status'] == 'failed' and failed['stream'] == 0
    assert failed['loss_targets'] == failed['backward'] == 4
    assert failed['elapsed_seconds'] > 0
    assert failed['metaquant_work']['theta_optimizer_updates'] == len(method.theta)
    assert state.ledger.passes['backward'] == ledger_before['backward'] + 4
    assert all(e['forward_completed'] and e['backward_launched']
               for e in failed['metaquant_work']['loss_pass_events'])
    assert state.failed and not state.checkpointable
    with pytest.raises(IncompleteStep):
        capture_snapshot(state)


@pytest.mark.parametrize('mid_iterator_failure', [False, True])
def test_reconstruction_reuse_preserves_prior_generation(tmp_path, mid_iterator_failure):
    from ralup.data.articles import write_reconstruction
    write_reconstruction([' = Original = ', 'sentinel'], tmp_path, split='synthetic')
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    consumed = []
    def rows():
        consumed.append(True)
        yield ' = Replacement = '
        yield ' = Second = '
        if mid_iterator_failure:
            raise RuntimeError('mid-iterator failure')
        yield 'replacement'
    if mid_iterator_failure:
        # A genuinely new split may fail mid-iteration. Retain both its partial
        # evidence and the prior valid generation; neither may be overwritten.
        with pytest.raises(RuntimeError, match='mid-iterator'):
            write_reconstruction(rows(), tmp_path, split='candidate')
        assert {path: path.read_bytes() for path in before} == before
        partial = {path: path.read_bytes() for path in tmp_path.glob('candidate.*')}
        with pytest.raises(ValueError, match='existing|generation'):
            write_reconstruction(['retry'], tmp_path, split='candidate')
        assert {path: path.read_bytes() for path in partial} == partial
        consumed.clear()
    with pytest.raises(ValueError, match='existing|generation'):
        write_reconstruction(rows(), tmp_path, split='synthetic')
    assert not consumed
    assert {path: path.read_bytes() for path in before} == before


def test_preparation_reuse_rejects_before_touching_source_or_outputs(tmp_path, monkeypatch):
    from ralup.data import source
    output = tmp_path/'derived'; output.mkdir()
    (output/'preparation.json').write_bytes(b'original preparation sentinel')
    (output/'train.content.bin').write_bytes(b'original content sentinel')
    before = {path: path.read_bytes() for path in output.iterdir()}
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'kind': 'pinned_source', 'repository': source.IDENTITY[0],
        'configuration': source.IDENTITY[1], 'revision': source.IDENTITY[2],
        'cache': str(tmp_path/'cache'), 'output': str(output)}))
    def forbidden(*args):
        pytest.fail('Existing generation must reject before source observation/download')
    monkeypatch.setattr(source, 'fetch_source', forbidden)
    with pytest.raises(ValueError, match='existing|generation'):
        source.prepare_data(config)
    assert {path: path.read_bytes() for path in output.iterdir()} == before
    assert not (tmp_path/'cache').exists()


@pytest.mark.parametrize('damage', [False, True])
def test_authenticated_cached_source_preserves_original_observation(tmp_path, monkeypatch, damage):
    from ralup.data import source
    from ralup.records import sha256_file
    asset = tmp_path/'fixture.parquet'; asset.write_bytes(b'synthetic cache bytes')
    pins = {'assets': [{'name': asset.name, 'bytes': asset.stat().st_size,
        'sha256': sha256_file(asset), 'pinned_head': {'x-repo-commit': source.IDENTITY[2]}}],
        'intended_use': 'synthetic test only', 'protected_test_policy': 'no targets',
        'license_discrepancy': 'synthetic'}
    monkeypatch.setattr(source, '_pins', lambda: pins)
    def forbidden(*args):
        pytest.fail('Authenticated cache reuse must not download')
    monkeypatch.setattr(source, '_download', forbidden)
    reference = source.fetch_source(*source.IDENTITY, tmp_path)
    original = reference.path.read_bytes()
    if damage:
        record = json.loads(original); record['assets'][0]['observed_sha256'] = '0'*64
        reference.path.write_text(json.dumps(record))
        before = reference.path.read_bytes()
        with pytest.raises(ValueError, match='observation|manifest'):
            source.fetch_source(*source.IDENTITY, tmp_path)
        assert reference.path.read_bytes() == before
    else:
        assert source.fetch_source(*source.IDENTITY, tmp_path) == reference
        assert reference.path.read_bytes() == original


def test_post_capture_cap_failure_retains_attempt_component(tmp_path, monkeypatch):
    import ralup.calibrate as calibration
    from ralup.config import resolve_config
    template = json.loads((Path(__file__).parents[2]/'configs/smoke.json').read_bytes())
    template['calibration'] = {'output': str(tmp_path/'runs'), 'seed': 808, 'repeats': 2,
        'pilot_target_budget': 10, 'warmstart_target_budget': 15,
        'continuation_target_budget': 7, 'doublings': 2}
    config = resolve_config(template, {'gates': {}}, purpose='calibration')
    create, capture, step = calibration._create, calibration.capture_snapshot, calibration.step_with_oom_retry
    states = []; captures = []
    def create_state(*args):
        state = create(*args); states.append(state); return state
    def capture_state(state):
        snapshot = capture(state)
        captures.append(snapshot)
        return snapshot
    def memory(device):
        return {'rss_bytes': 2 if len(captures) >= 2 else 0, 'metal_bytes': None, 'driver_bytes': None}
    monkeypatch.setattr(calibration, '_create', create_state)
    monkeypatch.setattr(calibration, 'capture_snapshot', capture_state)
    monkeypatch.setattr(calibration, '_memory', memory)
    monkeypatch.setattr(calibration, '_limits', lambda payload: {'resident_cap_bytes': 1, 'driver_cap_bytes': None})
    monkeypatch.setattr(calibration, 'step_with_oom_retry', lambda state: step(state, 5))
    with pytest.raises(RuntimeError, match='memory cap'):
        calibration.train_warmstart(config, 'meta_train', 808)
    assert len(captures) == 2
    state = states[0]
    events = [event for event in state.ledger.events if event['status'] == 'checkpoint_write']
    assert len(events) == 1, 'Real second capture work vanished at failed publication boundary'
    event = events[0]
    assert event['outcome'] == 'failed' and event['elapsed_seconds'] > 0
    assert event['cost_role'] == 'component_within_project_elapsed'
    assert state.ledger.events[-2]['boundary'] == 'before_checkpoint_write'
    directory = tmp_path/'runs/warmstart-meta_train-808'
    assert not (directory/'checkpoint').exists() and not (directory/'warmstart.json').exists()
    records = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    enclosing = [row for row in records if row['status'] == 'project_elapsed_observation'][-1]
    assert enclosing['elapsed_seconds'] == 0
    assert enclosing['project_elapsed_seconds'] >= event['elapsed_seconds']


def cost_fixture(tmp_path):
    """Authenticated synthetic byte receipts with hand-counted UTC regions."""
    from datetime import datetime, timezone
    from ralup.search import _write, _ref
    def stamp(value):
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    def execution(name, start, end, parent=None):
        return _write(tmp_path/(name+'.json'), {'trial_id': name, 'family': name,
            'status': 'completed', 'curve': [{'development_bpb': 2.}],
            'executions': [{'event_id': name, 'kind': 'loss', 'targets': 4,
                'stream_targets': 4, 'classifications': [],
                'source_intervals': [['synthetic', name, 0, 4]]}],
            'cost': {'parent': None if parent is None else _ref(parent),
                'complete_seconds': end-start,
                'wall_interval': {'clock': 'UTC', 'start': start, 'end': end}}})
    def producer(name, subject, start, measured_end, closed_end=None, extra_outputs=()):
        startup = _write(tmp_path/(name+'-startup.json'), {'utc': stamp(start),
            'launcher_started_utc': stamp(start), 'source': {}, 'inputs': []})
        finish = _write(tmp_path/(name+'-finish.json'), {'completed_utc': stamp(measured_end),
            'elapsed_seconds': measured_end-start, 'exit_code': 0,
            'completed_objects': [{'path': str(ref.path), 'sha256': ref.sha256}
                                  for ref in (subject, *extra_outputs)]})
        receipt = {'startup': _ref(startup), 'finish': _ref(finish)}
        if closed_end is not None:
            observer = _write(tmp_path/(name+'-enclosing.json'), {
                'schema': 'ralup.complete_component_interval.v1', 'subject': _ref(subject),
                'producer': receipt.copy(),
                'boundary': 'launcher_start_through_all_outputs_and_finish_publication',
                'wall_interval': {'clock': 'UTC', 'start': start, 'end': closed_end},
                'excluded_costs': ['observer_and_receipt_administration']})
            receipt['enclosing_measurement'] = _ref(observer)
        return receipt
    warm = execution('warm-result', 0, 4)
    parent = _write(tmp_path/'warmstart.json', {'cost': {'execution_record': _ref(warm)}})
    child = execution('child', 4, 9, parent)
    return execution, producer, warm, parent, child


def test_authentic_prepublication_trial_interval_is_only_lower_bound(tmp_path):
    from ralup.analysis.cost import trial_cost_record, cost_quality_frontier
    _, _, _, _, child = cost_fixture(tmp_path)
    converted = trial_cost_record(child)
    totals = cost_quality_frontier([converted], 'seconds')['totals']
    assert totals['known_interval_seconds'] == 9
    assert totals['seconds'] is None, 'Authenticated pre-publication bytes do not close elapsed cost'
    assert converted['wall_scope']['complete'] is False
    assert converted['parents'][0]['wall_scope']['complete'] is False


@pytest.mark.parametrize('closure', ['missing_parent', 'parent_footer', 'child_footer', 'complete'])
def test_each_deployment_ancestor_needs_its_complete_enclosing_boundary(tmp_path, closure):
    from ralup.analysis.cost import trial_cost_record, cost_quality_frontier
    _, producer, warm, parent, child = cost_fixture(tmp_path)
    warm_receipt = producer('warm', warm, 0, 6,
        None if closure == 'parent_footer' else 7, extra_outputs=(parent,))
    child_receipt = producer('child', child, 4, 11, None if closure == 'child_footer' else 12)
    parents = {} if closure == 'missing_parent' else {parent.sha256: warm_receipt}
    converted = trial_cost_record(child, producer=child_receipt, parent_producers=parents)
    result = cost_quality_frontier([converted], 'seconds')
    assert result['totals']['seconds'] == (12 if closure == 'complete' else None)
    assert result['totals']['known_interval_seconds'] == (11 if closure == 'child_footer' else 12)
    assert result['totals']['common_engineering_seconds'] is None
    assert result['frontier'][0]['cost'] == result['totals']['seconds']


@pytest.mark.parametrize('damage', ['wrong_subject', 'wrong_producer', 'short_boundary', 'wrong_boundary', 'corrupt'])
def test_enclosing_interval_authenticates_subject_producer_and_actual_boundary(tmp_path, damage):
    from ralup.analysis.cost import trial_cost_record
    from ralup.search import _read, _write, _ref
    from ralup.records import ArtifactRef
    _, producer, warm, _, child = cost_fixture(tmp_path)
    receipt = producer('child', child, 4, 11, 12)
    reference = ArtifactRef(**receipt['enclosing_measurement'])
    observed = _read(reference)
    if damage == 'corrupt':
        reference.path.write_text('{}')
    else:
        if damage == 'wrong_subject': observed['subject'] = _ref(warm)
        elif damage == 'wrong_producer': observed['producer']['finish'] = _ref(warm)
        elif damage == 'short_boundary': observed['wall_interval']['end'] = 10
        else: observed['boundary'] = 'before_result_publication'
        receipt['enclosing_measurement'] = _ref(_write(tmp_path/'invalid-observation.json', observed))
    with pytest.raises(ValueError, match='hash|boundary|subject|producer|enclosing'):
        trial_cost_record(child, producer=receipt)


def test_shared_warm_parent_attributed_per_deployment_deduplicated_globally(tmp_path):
    from ralup.analysis.cost import trial_cost_record, cost_quality_frontier
    execution, producer, warm, parent, first = cost_fixture(tmp_path)
    second = execution('second', 12, 17, parent)
    offline = execution('offline-meta-search', 20, 120)
    parents = {parent.sha256: producer('warm', warm, 0, 6, 7, extra_outputs=(parent,))}
    a = trial_cost_record(first, producer=producer('first', first, 4, 11, 12), parent_producers=parents)
    b = trial_cost_record(second, producer=producer('second', second, 12, 19, 20), parent_producers=parents)
    tuning = trial_cost_record(offline, producer=producer('offline', offline, 20, 120, 121))
    deployment = cost_quality_frontier([a, b], 'seconds')
    assert [row['seconds'] for row in deployment['matched_curves']] == [12, 15]
    assert deployment['totals']['seconds'] == 20
    assert deployment['totals']['run_ids'].count('warm-result') == 1
    project = cost_quality_frontier([a, b, tuning], 'seconds')
    assert project['totals']['seconds'] == 121
    assert deployment['totals']['seconds'] == 20  # Offline META/search is a separate root/view.


def ranking_rows(costs=(12, 15), other_costs=(14, 17)):
    rows = []
    for family, values in [('first', costs), ('second', other_costs)]:
        for seed, seconds in zip((71, 72), values):
            rows.append({'trial_id': f'{family}-{seed}', 'family': family, 'seed': seed,
                'budget': 100, 'status': 'completed', 'full_budget': True, 'fresh_seed': True,
                'settled': True, 'lr_bracketed': True, 'edge': False, 'missing_mechanism': False,
                'model': 'core', 'recipe': {'method': family},
                'curve': [{'budget': 100, 'development_bpb': 2.}],
                'cost': {'complete_seconds': seconds, 'offline_meta_search_seconds': 1000}})
    return rows


def test_exact_bpb_tie_exports_mean_deployment_seconds_separately_from_sum():
    from ralup.search import _rank_development
    rows = ranking_rows()
    result = _rank_development(rows, [71, 72])
    winner = result['winner']
    assert winner['recipe']['method'] == 'first'
    assert winner['mean_complete_deployment_seconds'] == 13.5
    assert winner['sum_complete_deployment_seconds'] == 27
    assert winner['deployment_cost_unit'] == 'seconds'
    assert winner['deployment_cost_boundary'] == 'warm_creation_online_training_evaluation_checkpoint_publication'
    assert _rank_development(list(reversed(rows)), [72, 71])['winner']['mean_complete_deployment_seconds'] == 13.5
    # Different reporting partitions of the SAME required seed set retain the mean.
    grouped = [rows[0], rows[2]], [rows[1], rows[3]]
    regrouped = [row for group in grouped for row in group]
    assert _rank_development(regrouped, [71, 72])['winner']['mean_complete_deployment_seconds'] == 13.5
    with pytest.raises(ValueError, match='identical fresh seeds'):
        _rank_development(rows[:-1], [71, 72])


def test_unknown_complete_seconds_cannot_decide_exact_bpb_tie():
    from ralup.search import _rank_development
    with pytest.raises(ValueError, match='complete.*cost|cost.*complete'):
        _rank_development(ranking_rows(costs=(None, 15)), [71, 72])


def test_quality_order_preserves_unknown_complete_deployment_cost():
    from ralup.search import _rank_development
    rows = ranking_rows(costs=(None, 15))
    for row in rows[2:]: row['curve'][-1]['development_bpb'] = 3.
    winner = _rank_development(rows, [71, 72])['winner']
    assert winner['recipe']['method'] == 'first'
    assert winner['mean_complete_deployment_seconds'] is None
    assert winner['sum_complete_deployment_seconds'] is None
