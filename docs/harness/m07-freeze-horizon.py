"""Measured finite128-step scope extension, frozen before its outcomes."""
import copy,json,os
from pathlib import Path
from datetime import datetime,timezone
from ralup.search import source_identity
from ralup.records import sha256_file
from ralup.calibrate import _write
root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
old=json.loads((root/'docs/harness/m07-design-v1.json').read_bytes())
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
cases=[]
def add(index,label,method=None,quantizer=None,rotation='identity',backward='ste',parent=False):
 c=copy.deepcopy(old['cases'][index]);c.update(source_identity=source_identity(),budget=128*8192,role='continuation' if parent else 'warmstart',label=label,milestones=[32,64,128])
 if method:c['recipe']['method']=method;c['family']='Muon' if method.startswith('muon') else 'AdamW';c['recipe']['learning_rate']=.02 if method.startswith('muon') else .001
 if quantizer:c['recipe']['quantizer']=quantizer
 c['recipe'].update(rotation=rotation,backward=backward,grouping='none' if c['recipe']['quantizer']=='fp32' else 'whole_tensor' if 'tensor' in c['recipe']['quantizer'] else 'row')
 if parent:c['parent']=ref(base/'science/core-target-warm32/warmstart.json')
 c['resolved']['calibration'].update(pilot_target_budget=128*8192,warmstart_target_budget=128*8192,continuation_target_budget=128*8192)
 cases.append(c)
add(0,'proxy-fp32-warm128')
add(2,'core-fp32-warm128')
add(3,'core-muon-dynamic128',parent=True)
add(3,'core-muon-fp32128',quantizer='fp32',parent=True)
add(2,'core-adamw-dynamic128',quantizer='row_dynamic_max',parent=True)
add(2,'core-quest-row-hadamard-trust128',quantizer='quest_fp4_row_reference',rotation='full_hadamard',backward='trust',parent=True)
add(2,'core-quest-tensor-identity-trust128',quantizer='quest_fp4_tensor_reference',backward='trust',parent=True)
science=[]
for name in ['proxy-fp32-cont32','proxy-muon-dynamic-cont32','core-fp32-cont32','core-muon-dynamic-cont32']:
 p=base/'science'/name/'result.json';r=json.loads(p.read_bytes());science.append({'evidence':ref(p),'complete_seconds':r['cost']['complete_seconds'],'curve':r['curve']})
design={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'research_cutoff':'2026-10-03','source_identity':source_identity(),
 'purpose':'prospective required128-step horizon/actual new recipe numerical admission; separately charged from science; no previous-scope replay',
 'decision_basis':'32-step held-out curves still improve; current matched quantization damage is small; new matched-optimizer scopes remove geometry confounding; actual QuEST variants join ordinary pool',
 'measured_basis':science,'cases':cases,'steps':128,'comparisons':[1,32,64,128],'restore_at_step':64,
 'contract':'exact full scientific owners at declared completed checkpoints; independent duplicate initial state; second arm strict midpoint restore; no tolerance relaxation',
 'owners':'W/gradients/moments/caches/codes/scales/history/schedule/cursor/ordered input/RNG/scientific ledger; timing/process paths excluded only by H07 existing hash',
 'maximum_training_steps':1792,'finite_forecast_seconds':2400,
 'forecast_basis':'actual core science32 complete42.02–48.55s including4 checkpoints and queries;128 has4x training and sparse comparisons;7 paired cases, conservative2400s total',
 'forecast_bytes':4*1024**3,'forecast_model_state_bytes':{'proxy_adam_floor':493056*16,'core_adam_floor':3347456*16,'core_quantized_runtime_plus_snapshot_envelope':128*3347456},
 'retention':'raw run-level input/loss/seed/batch/source/cost every step; save completed127/128 plus strict64 midpoint in arm1, failures; hash only1/32/64/128; no every-step snapshots/hashes',
 'caps':'same freshly observed conservative H08 RSS/driver caps;32GiB totalstore;Datafree64GiB floor; abort first scope failure/cap and preserve all work',
 'next':'development duration doubling and quantized LR bracketing; neither initial32 nor128 is a final budget by fiat; all unsettled/edge/overlap/strong quantizers preserved'}
p=root/'docs/harness/m07-design-v2.json';_write(p,design);print(json.dumps(ref(p)))
