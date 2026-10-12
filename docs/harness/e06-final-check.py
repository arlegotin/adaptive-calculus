"""One source admission: covering CPU checks, finite native composition, actual CLI.

All operations retain separate immutable command/status/wall/process receipts.
The initialized core fixture is synthetic mechanics, never the missing selected
research final model. No training or protected-data work is performed here.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

base = Path(os.environ['E06_ATTEMPT_DIR'])


def run(name, command):
    start = time.perf_counter()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (base/(name+'-stdout.txt')).open('xb') as out, (base/(name+'-stderr.txt')).open('xb') as err:
        result = subprocess.run(command, stdout=out, stderr=err)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {'command':command,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-start,
        'child_user_seconds':after.ru_utime-usage.ru_utime,'child_system_seconds':after.ru_stime-usage.ru_stime,
        'cost_scope':'nested operation wall/process; do not add to enclosing launcher totals'}
    with (base/(name+'-operation.json')).open('x') as f: json.dump(record,f,sort_keys=True,indent=2)
    print(json.dumps(record),flush=True)
    print((base/(name+'-stdout.txt')).read_text(),flush=True)
    print((base/(name+'-stderr.txt')).read_text(),flush=True)
    return result.returncode


cpu = [sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
    'tests/unit/test_packed_export.py','tests/integration/test_export_forward.py',
    'tests/unit/test_checkpoint_integrity.py','tests/unit/test_learned_scale.py',
    'tests/unit/test_quantization_context.py','tests/unit/test_transformer.py',
    'tests/unit/test_e2m1.py','tests/integration/test_scoring.py','-k','not mps']
codes = [run('covering-cpu',cpu)]
codes.append(run('finite-mps',[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
    'tests/integration/test_export_forward.py','-k','mps']))
if any(codes): raise SystemExit(1)

# Producer/source archive already exists in startup before these imports.
import torch
from ralup.models.transformer import ByteTransformer, ModelConfig
from ralup.optim.base import MethodAdapter, default_parameter_groups
from ralup.optim.adamw import AdamWUpdateRule
from ralup.state import capture_snapshot, save_snapshot
from ralup.train import TrainingState
from ralup.export import load_export
from ralup.records import ArtifactRef, sha256_file

torch.manual_seed(606)
model = ByteTransformer(ModelConfig(4,256,4))
model.configure_quantization({'name':'row_dynamic_max','rotation':'full_hadamard'})
state = TrainingState(model,MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule()),
    configuration={'synthetic':True,'purpose':'initialized E06 core export CLI mechanics; no research training'})
checkpoint = save_snapshot(capture_snapshot(state),base/'cli-fixture-checkpoint')
output = base/'core-fixture.fp4'
code = run('actual-export-cli',[sys.executable,'-m','ralup','export','--checkpoint',str(checkpoint.path),'--output',str(output)])
if code: raise SystemExit(code)
decoded = load_export(ArtifactRef(output,sha256_file(output)))
assert decoded.metadata['analytical_payload']['payload_bytes'] == 2416640
assert decoded.metadata['actual_container_bytes'] == output.stat().st_size
with (base/'actual-cli-proof.json').open('x') as f:
    json.dump({'checkpoint':{'path':str(checkpoint.path),'sha256':checkpoint.sha256},
        'export':{'path':str(output),'sha256':sha256_file(output)},'metadata':decoded.metadata,
        'scientific_final_model':False},f,sort_keys=True,indent=2)
print(json.dumps({'core_payload_bytes':2416640,'actual_container_bytes':output.stat().st_size,
    'actual_cli_export_verified':True,'scientific_final_model':False}),flush=True)
