"""Reproducible H08 actual-source/model-only paired continuation evidence.

Run after the separately authenticated warmstart CLI; no protected data scoring.
"""
import json
from pathlib import Path
import sys
import time
import uuid
from datetime import datetime, timezone

import torch

from ralup.__main__ import _calibration_config
from ralup.calibrate import _create, _event, _timed, _write, import_warmstart
from ralup.data.cursor import BatchCursor
from ralup.records import ArtifactRef, RunLedger, sha256_file
from ralup.state import capture_snapshot, exact_cpu_state_hash, load_snapshot, save_snapshot, _validated
from ralup.train import select_logical_batch, train_step


def run(config_path, refs_path, output):
    start = time.perf_counter()
    resolved, _ = _calibration_config(config_path)
    refs = json.loads(Path(refs_path).read_bytes())
    output = Path(output)
    results = {}
    project_seconds = 0.
    for partition, value in refs.items():
        ref = ArtifactRef(**value)
        assert sha256_file(ref.path) == ref.sha256
        parent = json.loads(ref.path.read_bytes())
        checkpoint = ArtifactRef(**parent['checkpoint'])
        saved = _validated(load_snapshot(checkpoint))
        parent_cursor = BatchCursor(saved['cursor'])
        assert parent_cursor.epoch == 0  # This explicit pilot is smaller than either partition.
        parent_offsets = []
        for position in range(parent_cursor.permutation_position + (parent_cursor._carry is not None)):
            document, first, last = parent_cursor._reference(parent_cursor._permutation[position])
            if position == parent_cursor.permutation_position:
                last = parent_cursor._carry['score_start']
            parent_offsets.append({'document_id': parent_cursor._source['hashes']['source_sha256'] + ':' + document['id'],
                                   'offsets': list(range(first, last))})
        states, selected, hashes, costs, checkpoints = [], [], [], [], []
        for index in range(2):
            arm_start = time.perf_counter()
            state = _create(resolved, partition, resolved.payload['calibration']['continuation_target_budget'], 808)
            state, seconds = _timed(state.model, lambda: import_warmstart(ref, state, partition=partition,
                continuation_targets=resolved.payload['calibration']['continuation_target_budget']))
            assert all(torch.equal(p.detach().cpu(), saved['weights'][name]) for name, p in state.model.named_parameters())
            assert state.cursor.state['rng_states'] == saved['cursor']['rng_states']
            assert state.cursor.cumulative_target_count == parent['cost']['stream']['targets']
            assert state.ledger.passes == parent['cost']['passes']
            assert state.configuration['warmstart_parent']['parent_cost']['project_seconds'] == parent['project_seconds']
            assert state.histories == state.quantizer_state == {}
            assert all(not values for values in state.optimizer_state.values())
            assert state.schedule['step'] == 0
            metadata = json.loads((checkpoint.path.parents[2]/'run.json').read_bytes())
            metadata.pop('status_history')
            metadata.update({'run_id': 'h08-import-'+uuid.uuid4().hex,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'warm_start_sha256': ref.sha256,
                'lineage': {'parent': metadata['run_id'], 'resume': checkpoint.sha256},
                'endpoint': {'scope': 'actual-source paired model-only engineering import', 'partition': partition,
                             'continuation_targets': 8192, 'b0': None}})
            ledger = RunLedger(output/partition/f'arm-{index}', metadata)
            state.ledger.run_ledger = ledger
            ledger.record_status('running')
            _event(state, 'authenticated_model_only_import', seconds)
            # Retain fresh constructor/index opening as distinct setup elapsed.
            _event(state, 'fresh_state_setup', time.perf_counter()-arm_start-seconds)
            batch = select_logical_batch(state)
            selected.append([(block.document_id, block.scored_offsets()) for block in batch.block_slices])
            result, duration = _timed(state.model, lambda: train_step(state, batch))
            assert result.target_count == 8192
            assert state.ledger.passes['backward'] == parent['cost']['passes']['backward'] + 8192
            hashes.append(exact_cpu_state_hash(state))
            published, duration = _timed(state.model, lambda: save_snapshot(capture_snapshot(state), output/partition/f'arm-{index}'/'checkpoint'))
            _event(state, 'continuation_checkpoint', duration)
            elapsed = time.perf_counter()-arm_start
            ledger.record({'step': state.schedule['step'], 'stream': 0, 'unique': 0, 'additional_loss': 0,
                           'backward': 0, 'query': 0, 'elapsed_seconds': 0., 'status': 'project_elapsed_observation',
                           'project_elapsed_seconds': elapsed})
            ledger.record_status('completed', disposition='paired_engineering_import')
            costs.append({'new_process_seconds': elapsed, 'cumulative_stream': state.ledger.stream,
                          'cumulative_passes': state.ledger.passes, 'cumulative_elapsed': state.ledger.elapsed,
                          'new_step_full_seconds': result.work['elapsed_seconds']})
            checkpoints.append({'path': str(published.path), 'sha256': published.sha256})
            states.append(state)
            project_seconds += elapsed
        assert selected[0] == selected[1]
        # Wall-clock setup observations differ across arms; compare actual arrays,
        # cursor and named streams explicitly instead of hashing audit costs.
        assert all(torch.equal(p, dict(states[1].model.named_parameters())[name]) for name, p in states[0].model.named_parameters())
        assert states[0].cursor.state == states[1].cursor.state
        for name, rng in states[0].rng_streams.items():
            other = states[1].rng_streams[name]
            if isinstance(rng, torch.Generator): assert torch.equal(rng.get_state(), other.get_state())
            else: assert rng.getstate() == other.getstate()
        results[partition] = {'parent': value, 'parent_checkpoint': parent['checkpoint'], 'bitwise_weights': True,
            'identical_cursor_rng_offsets': True, 'parent_bytes': parent['cost']['stream']['targets'],
            'paired_continuation_bytes': [8192, 8192], 'costs': costs, 'checkpoints': checkpoints,
            'parent_target_offsets': parent_offsets,
            'continuation_target_offsets': [{'document_id': document_id, 'offsets': list(offsets)} for document_id, offsets in selected[0]],
            'audit_sensitive_hashes': hashes}
    report = {'schema_version': 1, 'scope': 'H08 real-source actual MPS engineering; no final duration or protected score',
              'device': 'mps', 'fallback': '0', 'results': results, 'project_seconds': time.perf_counter()-start,
              'new_arm_process_seconds': project_seconds, 'b0': None}
    ref = _write(output/'validation.json', report)
    print(json.dumps({'path': str(ref.path), 'sha256': ref.sha256}))


if __name__ == '__main__':
    run(*sys.argv[1:])
