"""Isolate real-input embedding backward duplicate variation; no bound selection."""
import json
from pathlib import Path
import sys
import time
import uuid
from datetime import datetime, timezone

import torch
from safetensors.torch import save_file

from ralup.__main__ import _calibration_config
from ralup.calibrate import _create, _event, _write, import_warmstart
from ralup.data.batches import next_logical_batch
from ralup.data.cursor import BatchCursor
from ralup.evaluate import summed_nll
from ralup.records import ArtifactRef, RunLedger, sha256_file
from ralup.state import _validated, load_snapshot


def run(config_path, refs_path, output):
    start = time.perf_counter()
    resolved, _ = _calibration_config(config_path)
    ref = ArtifactRef(**json.loads(Path(refs_path).read_bytes())['meta_train'])
    parent = json.loads(ref.path.read_bytes())
    checkpoint = ArtifactRef(**parent['checkpoint'])
    saved = _validated(load_snapshot(checkpoint))
    state = import_warmstart(ref, _create(resolved, 'meta_train', 8192, 808),
                            partition='meta_train', continuation_targets=8192)
    output = Path(output)
    metadata = json.loads((checkpoint.path.parents[2]/'run.json').read_bytes())
    metadata.pop('status_history')
    metadata.update({'run_id': 'h08-embedding-'+uuid.uuid4().hex, 'created_utc': datetime.now(timezone.utc).isoformat(),
                     'warm_start_sha256': ref.sha256,
                     'lineage': {'parent': metadata['run_id'], 'resume': checkpoint.sha256},
                     'endpoint': {'scope': 'real-input operator component isolation only; no bound change', 'b0': None}})
    ledger = RunLedger(output, metadata)
    state.ledger.run_ledger = ledger
    ledger.record_status('running')
    _event(state, 'diagnostic_setup', time.perf_counter()-start)
    cursor = BatchCursor(state.cursor.state)
    batch = next_logical_batch(cursor, 128)
    captured = {}
    state.model.prepare_quantized_views(0, state.rng_streams['quantizer'])
    def capture(module, inputs, value):
        captured['inputs'] = inputs[0].detach().clone()
        captured['output'] = value
    hook = state.model.embedding.register_forward_hook(capture)
    region = time.perf_counter()
    try:
        loss = summed_nll(state.model, batch)
        upstream = torch.autograd.grad(loss, captured['output'])[0].detach().clone()
        torch.mps.synchronize()
    finally:
        hook.remove()
        _event(state, 'actual_upstream_gradient_query', time.perf_counter()-region,
               loss_targets=batch.target_count, additional_loss=batch.target_count, query=batch.target_count,
               backward=batch.target_count, forward_context=sum(len(b.inputs) for b in batch.block_slices))
    indices = captured['inputs']
    weight = saved['weights']['embedding.weight']
    rows, gradients = [], []
    for device in ('cpu', 'mps'):
        outputs, local_gradients = [], []
        for repeat in range(12):
            region = time.perf_counter()
            module = torch.nn.Embedding.from_pretrained(weight.to(device).clone(), freeze=False)
            values = module(indices.to(device))
            outputs.append(values.detach().cpu())
            values.backward(upstream.to(device))
            if device == 'mps': torch.mps.synchronize()
            local_gradients.append(module.weight.grad.detach().cpu().clone())
            _event(state, 'embedding_component_gradient', time.perf_counter()-region,
                   backward=batch.target_count, forward_context=indices.numel(),
                   component_device=device, component_repeat=repeat, component_context_tokens=indices.numel())
        assert all(torch.equal(outputs[0], value) for value in outputs[1:])
        assert all(torch.equal(outputs[0], weight[indices.cpu()]) for _ in [0])
        pairs = [(float((local_gradients[i]-local_gradients[j]).abs().max()),
                  int((local_gradients[i] != local_gradients[j]).sum()))
                 for i in range(12) for j in range(i)]
        rows.append({'device': device, 'repeats': 12, 'embedding_forward_exact': True,
                     'pairwise_gradient_max_abs': max(x[0] for x in pairs),
                     'max_changed_entries': max(x[1] for x in pairs),
                     'input_identity': 'identical actual byte indices and identical saved upstream gradient'})
        gradients.extend((f'{device}_gradient_{i}', value) for i, value in enumerate(local_gradients))
    payload = {'indices': indices.cpu().contiguous(), 'upstream': upstream.cpu().contiguous(),
               'weight': weight.contiguous(), **dict(gradients)}
    tensors = output/'repeat-data.safetensors'
    save_file(payload, str(tensors))
    report = {'schema_version': 1, 'scope': 'isolated real-input embedding backward diagnosis; no numerical bound selected',
              'parent': {'path': str(ref.path), 'sha256': ref.sha256}, 'rows': rows,
              'context_tokens': indices.numel(), 'distinct_input_tokens': int(indices.unique().numel()),
              'scored_targets': batch.target_count, 'input_shape': list(indices.shape),
              'upstream_shape': list(upstream.shape), 'tensor_manifest': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k,v in payload.items()},
              'repeat_data': {'path': str(tensors), 'sha256': sha256_file(tensors)},
              'project_seconds': time.perf_counter()-start, 'cost': {'passes': state.ledger.passes, 'events': state.ledger.events},
              'disposition': 'root numerical contract adjudication required; historical zero bounds unchanged'}
    ref = _write(output/'diagnostic.json', report)
    ledger.record({'step': 0, 'stream': 0, 'unique': 0, 'additional_loss': 0, 'backward': 0, 'query': 0,
                   'elapsed_seconds': 0., 'status': 'project_elapsed_observation', 'project_elapsed_seconds': time.perf_counter()-start})
    ledger.record_status('completed', disposition=report['disposition'])
    print(json.dumps({'path': str(ref.path), 'sha256': ref.sha256, 'rows': rows}))


if __name__ == '__main__':
    run(*sys.argv[1:])
