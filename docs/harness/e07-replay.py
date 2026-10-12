"""Finite current-source native paired engineering reproduction on frozen corpus.

Two quantizers, one paired seed, four128-target updates each; repeat with
interruption after two updates and a fresh interpreter. Not a finalist/pilot.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import torch
from ralup.data.cursor import BatchCursor
from ralup.models.transformer import ByteTransformer,ModelConfig
from ralup.optim.adamw import AdamWUpdateRule
from ralup.optim.base import MethodAdapter,SchedulePoint,default_parameter_groups
from ralup.records import ArtifactRef
from ralup.state import capture_snapshot,save_snapshot,load_snapshot,restore_snapshot,exact_cpu_state_hash
from ralup.train import TrainingState,select_logical_batch,train_step


def advance(state,count):
    records=[]
    for _ in range(count):
        batch=select_logical_batch(state,128)
        inputs=[dict(document_id=b.document_id,offsets=list(b.scored_offsets())) for b in batch.block_slices]
        result=train_step(state,batch)
        records.append(dict(step=state.schedule['step'],inputs=inputs,work=dict(state.ledger.passes),
            stream=dict(state.ledger.stream),state_sha256=exact_cpu_state_hash(state)))
    return records


if len(sys.argv)>1 and sys.argv[1]=='--resume':
    state=restore_snapshot(load_snapshot(ArtifactRef(sys.argv[2],sys.argv[3])),'mps')
    rows=advance(state,2)
    Path(sys.argv[4]).write_text(json.dumps({'final_hash':exact_cpu_state_hash(state),'steps':rows},sort_keys=True)+'\n')
    raise SystemExit(0)

out=Path(os.environ['E07_OPERATION'])
assert torch.backends.mps.is_available() and os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0'
manifest=json.loads(Path('docs/data/h02-evidence.json').read_bytes())['manifest']
def initial(recipe):
    torch.manual_seed(607);random.seed(608);torch.mps.manual_seed(609)
    model=ByteTransformer(ModelConfig(1,8,2)).to('mps')
    model.configure_quantization({'name':recipe,'rotation':'full_hadamard'})
    cursor=BatchCursor.from_manifest(Path(manifest['path']),manifest['sha256'],partition='target-training',seed=33,run_target_budget=512)
    method=MethodAdapter(dict(model.named_parameters()),default_parameter_groups(model),AdamWUpdateRule(betas=(.8,.95)))
    return TrainingState(model,method,cursor=cursor,microbatch=2,clip_norm=.2,
        rng_streams={'initialization':torch.Generator().manual_seed(60),'quantizer':torch.Generator(device='mps').manual_seed(62),
            'meta':random.Random(63),'timing':random.Random(64)},
        schedule={'step':0,'point':SchedulePoint(.001,.02,0.)})

proof=[]
for recipe in ('row_fixed_max','row_dynamic_max'):
    full=initial(recipe);full_rows=advance(full,4);expected=exact_cpu_state_hash(full)
    interrupted=initial(recipe);first=advance(interrupted,2)
    snapshot=save_snapshot(capture_snapshot(interrupted),out/recipe/'interruption')
    continuation=out/recipe/'continuation.json'
    command=[sys.executable,'-B',__file__,'--resume',str(snapshot.path),snapshot.sha256,str(continuation)]
    child=subprocess.run(command,capture_output=True,text=True)
    (out/recipe/'resume-stdout.txt').write_text(child.stdout);(out/recipe/'resume-stderr.txt').write_text(child.stderr)
    assert child.returncode==0,(child.returncode,child.stderr)
    resumed=json.loads(continuation.read_bytes());assert expected==resumed['final_hash']
    assert [r['state_sha256'] for r in full_rows]==[r['state_sha256'] for r in first+resumed['steps']]
    proof.append(dict(recipe=recipe,seed=607,manifest=manifest,uninterrupted=full_rows,
        interrupted=first,resumed=resumed,checkpoint={'path':str(snapshot.path),'sha256':snapshot.sha256},
        resume_command=command,exact_scientific_state=True))
assert proof[0]['uninterrupted'][0]['inputs']==proof[1]['uninterrupted'][0]['inputs']
(out/'paired-replay.json').write_text(json.dumps(dict(schema='ralup.e07_paired_replay.v1',scope='measured M3 engineering only, current full-state source identity;512targets/arm, no tuning/selection/development/protected score',
    fresh_optimizer_updates=16,fresh_training_loss_targets=2048,protected_scores=0,proof=proof),sort_keys=True,indent=2)+'\n')
print(json.dumps({'paired_methods':2,'updates':16,'actual_mps':True,'fallback':0,'exact_interruption_resume':True}))
