"""Explicit derived FP32 continuation parents, never a full-state restore waiver.

The original artifact remains a continuation. A separate immutable record
authenticates its model-only use, inherited trajectory costs and completed
project/selection receipts. Project seconds overlap trajectory time and must
not be added to it or charged again for each reuse.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from .records import ArtifactRef, sha256_file
from .state import IntegrityError, _automatic_identities, _validated, load_snapshot

SCHEMA = 'ralup.derived_model_only_warmstart.v1'
RESETS = ['optimizer', 'quantizer', 'history', 'schedule']
MODEL_SOURCES = ('src/ralup/models/transformer.py', 'src/ralup/models/linear.py', 'src/ralup/models/embedding.py')


def _ref(reference):
    return {'path': str(reference.path), 'sha256': reference.sha256}


def _read(reference):
    if sha256_file(reference.path) != reference.sha256:
        raise IntegrityError('Derived parent evidence hash mismatch')
    return json.loads(reference.path.read_bytes())


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _original(checkpoint):
    manifest = _read(checkpoint)
    snapshot = load_snapshot(checkpoint)
    payload = _validated(snapshot)
    config, cursor, ledger = payload['configuration'], payload['cursor'], payload['ledger']
    parent = config.get('warmstart_parent')
    if config.get('artifact_role') != 'continuation' or payload['model_cache']['_recipe'] is not None:
        raise IntegrityError('Derived parent requires an original FP32 continuation role')
    if not parent or parent.get('resets') != RESETS or cursor is None:
        raise IntegrityError('Derived parent requires complete model-only warm ancestry and cursor')
    if config.get('source_partition') != parent.get('source_partition'):
        raise IntegrityError('Derived parent partition ancestry mismatch')
    # Authenticate the immutable warm ancestry, rather than trusting a restated
    # count or changing the continuation into a historical warmstart.
    ancestor_ref = ArtifactRef(**parent['checkpoint'])
    ancestor = _validated(load_snapshot(ancestor_ref))
    _read(ArtifactRef(**parent['artifact']))
    if (ancestor['configuration_sha256'] != parent['configuration_sha256'] or
            ancestor['configuration'].get('source_partition') != config['source_partition'] or
            ancestor['cursor']['source'] != cursor['source'] or
            ancestor['cursor']['cumulative_target_count'] >= cursor['cumulative_target_count'] or
            ledger['stream']['targets'] != cursor['cumulative_target_count'] or
            ancestor['ledger']['stream']['updates'] + payload['schedule']['step'] != ledger['stream']['updates']):
        raise IntegrityError('Derived parent cursor/cost/mixed ancestry mismatch')
    current = _automatic_identities()
    if any(name not in payload['identities']['automatic'] or
           current[name].sha256 != payload['identities']['automatic'][name].sha256 for name in MODEL_SOURCES):
        raise IntegrityError('Derived parent model semantics differ; no migration is registered')
    bound = {
        'schema': SCHEMA, 'role': 'derived_model_only_warmstart', 'original_role': 'continuation',
        'checkpoint': _ref(checkpoint), 'configuration_sha256': payload['configuration_sha256'],
        'source_partition': config['source_partition'], 'cursor': cursor,
        'model_config': asdict(payload['model_config']),
        'model_parameters': [{'name': name, 'shape': list(weight.shape), 'dtype': str(weight.dtype)}
                             for name, weight in sorted(payload['weights'].items())],
        'source_identity': {name: ref.sha256 for name, ref in payload['identities']['automatic'].items()},
        'producer_archive': manifest['producer_files'], 'model_contract': 'identical_model_source',
        'ancestry': {'warmstart_parent': parent, 'continuation_rule': payload['rule'],
                    'continuation_steps': payload['schedule']['step'],
                    'warm_targets': ancestor['cursor']['cumulative_target_count'],
                    'continuation_targets': cursor['cumulative_target_count']-ancestor['cursor']['cumulative_target_count'],
                    'total_targets': cursor['cumulative_target_count']},
        'cost': ledger, 'resets': RESETS,
        'import_contract': 'exact W/aux/cursor/global+named RNG into fresh current state; declared cursor envelope extension only',
    }
    # Optimizer constructor schemas internally use tuples (for example betas);
    # public immutable records use canonical JSON lists throughout.
    return json.loads(json.dumps(bound)), snapshot, payload, manifest


def _project_cost(attempts, selection, checkpoint):
    if not attempts or not selection:
        raise IntegrityError('Derived parent requires completed cost receipts and selection evidence')
    identities = [(str(ref.path.resolve()), ref.sha256) for ref in attempts]
    if len(set(identities)) != len(identities) or len({p for p,h in identities}) != len(identities):
        raise IntegrityError('Duplicate physical attempt cost receipt')
    seconds = 0.; completed = set()
    for ref in attempts:
        row = _read(ref)
        if type(row.get('exit_code')) is not int or not _finite(row.get('elapsed_seconds')) or not isinstance(row.get('completed_objects'), list):
            raise IntegrityError('Missing completed attempt cost receipt')
        seconds += row['elapsed_seconds']
        for obj in row['completed_objects']:
            path = (ref.path.parent / obj['path']).resolve()
            completed.add((str(path), obj['sha256']))
    for ref in [checkpoint, *selection]:
        _read(ref)
        if (str(ref.path.resolve()), ref.sha256) not in completed:
            raise IntegrityError('Parent/selection evidence lacks a completed physical cost receipt')
    return {'completed_attempts': [_ref(ref) for ref in attempts], 'completed_seconds': seconds,
            'selection_evidence': [_ref(ref) for ref in selection],
            'scope': 'listed completed physical attempts once globally; overlaps inherited trajectory ledger, never add both views or multiply by parent reuse',
            'unrecorded_common_engineering_seconds': None}


def create_derived_warmstart(checkpoint: ArtifactRef, output: Path, *, completed_attempts: list[ArtifactRef],
                             selection_evidence: list[ArtifactRef]) -> ArtifactRef:
    """Publish a new typed model-only parent; preserve all original artifact bytes."""
    from .calibrate import _write
    start = time.perf_counter()
    bound, _, _, _ = _original(checkpoint)
    bound['project_cost'] = _project_cost(completed_attempts, selection_evidence, checkpoint)
    bound['creation'] = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'validation_seconds': time.perf_counter()-start, 'new_training_targets':0, 'new_query_targets':0,
        'publication_io_cost': 'charged by enclosing completed attempt receipt; not yet terminal at record creation'}
    return _write(output, bound)


def import_derived_warmstart(reference: ArtifactRef, fresh, *, partition: str, continuation_targets: int):
    """Authenticate every binding before mutating fresh owners; never restore old optimizer state."""
    from .calibrate import _event, _import_authenticated_parent
    start = time.perf_counter()
    record = _read(reference)
    try:
        checkpoint = ArtifactRef(**record['checkpoint'])
        bound, snapshot, payload, manifest = _original(checkpoint)
        if set(record) != set(bound) | {'project_cost','creation'} or any(record[k] != v for k,v in bound.items()):
            raise IntegrityError('Derived parent schema or authenticated semantic binding mismatch')
        project = record['project_cost']
        expected = _project_cost([ArtifactRef(**r) for r in project['completed_attempts']],
                                 [ArtifactRef(**r) for r in project['selection_evidence']], checkpoint)
        if project != expected or not _finite(record['creation']['validation_seconds']):
            raise IntegrityError('Derived parent cost binding mismatch')
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, IntegrityError): raise
        raise IntegrityError(f'Invalid derived parent record: {error}') from error
    if fresh.cursor is None or fresh.cursor.state['source'] != payload['cursor']['source']:
        raise IntegrityError('Derived parent data source differs from fresh declared data')
    # The shared private body retains ordinary H08 model/dtype/freshness,
    # partition/cursor/RNG/Parameter-owner checks. Its public ordinary entry
    # still rejects continuation artifacts without this distinct schema.
    result = _import_authenticated_parent(checkpoint, reference, None, snapshot, payload, manifest, fresh,
                                          partition=partition, continuation_targets=continuation_targets)
    result.configuration['warmstart_parent']['derived_record'] = record
    _event(result, 'derived_model_only_import', time.perf_counter()-start)
    return result
