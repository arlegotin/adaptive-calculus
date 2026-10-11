"""Execute only the exact prospectively frozen M07 horizon scopes."""
import json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
import torch
from ralup.search import source_identity,create_trial_state
from ralup.config import canonical_bytes
from ralup.records import sha256_file
from ralup.calibrate import _write,_offsets,_Sampler,_limits,memory_report
from ralup.state import capture_snapshot,restore_snapshot,save_snapshot,exact_cpu_state_hash
from ralup.train import train_step,select_logical_batch
from ralup.optim.base import SchedulePoint
output=Path(os.environ['M07_ATTEMPT_DIR']);designpath=Path(sys.argv[1]);design=json.loads(designpath.read_bytes())
assert design['source_identity']==source_identity()
started=time.perf_counter();runs=[];scopes=[];failure=None
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
for config in design['cases']:
 case=output/config['label'];case.mkdir();row={'label':config['label'],'config':config,'arms':[]}
 try:
  state=create_trial_state(config);initial=capture_snapshot(state);expected={}
  with _Sampler('mps',_limits(config['resolved'])) as sampler:
   def cap():
    sampler.sample()
    if sampler.exceeded:raise RuntimeError('Frozen physical/driver memory cap exceeded')
    if time.perf_counter()-started>design['finite_forecast_seconds']:raise RuntimeError('Frozen complete admission wall envelope exhausted')
   cap()
   for arm in range(2):
    state=restore_snapshot(initial,'mps');armstart=time.perf_counter();saved=[]
    with (case/f'arm-{arm}.jsonl').open('x') as raw:
     for step in range(design['steps']):
      cap()
      if arm==1 and step==design['restore_at_step']:
       snapshot=capture_snapshot(state);r=save_snapshot(snapshot,case/'strict-midpoint');saved.append(ref(r.path));state=restore_snapshot(snapshot,'mps')
      recipe=config['recipe'];state.schedule['point']=SchedulePoint(recipe['learning_rate'],recipe.get('decay',0.),step/max(1,design['steps']-1))
      batch=select_logical_batch(state,8192);inputs=_offsets(batch)
      try:train_step(state,batch)
      finally:
       raw.write(canonical_bytes({'execution_id':f"{config['label']}:{arm}:{step}",'step':step+1,'inputs':inputs,'events':state.ledger.events[-1:]}).decode()+'\n');raw.flush()
      cap()
      if step+1 in design['comparisons']:
       digest=exact_cpu_state_hash(state)
       _write(case/f'comparison-{arm}-{step+1}.json',{'step':step+1,'state_sha256':digest})
       if arm==0:expected[step+1]=digest
       elif digest!=expected[step+1]:
        save_snapshot(capture_snapshot(state),case/'mismatch');raise RuntimeError(f'Exact scientific state mismatch at completed{step+1}')
      if step+1>=design['steps']-1:
       r=save_snapshot(capture_snapshot(state),case/f'completed-{arm}-{step+1}');saved.append(ref(r.path));cap()
    row['arms'].append({'arm':arm,'complete_seconds':time.perf_counter()-armstart,'saved':saved,'stream':state.ledger.stream,'passes':state.ledger.passes,'elapsed':state.ledger.elapsed,
      'parent_cost_basis':config.get('parent'),'cost_note':'parent ledger ancestry retained; new physical executions separately counted in raw execution IDs'})
    _write(case/f'arm-{arm}-completed.json',row['arms'][-1])
   row['memory']=memory_report(sampler.samples);row['caps']=sampler.caps
  scopes.append([config['resolved']['profile'],config['recipe']['method'],config['recipe']['quantizer'],config['recipe']['rotation'],config['recipe']['backward'],config['packing']])
 except Exception as error:
  failure={'label':config['label'],'error':str(error)};row['failure']=failure
  if 'sampler' in locals():row['memory']=memory_report(sampler.samples)
 finally:
  runs.append(row);_write(case/'scope-result.json',row)
 if failure:break
report={'status':'blocked' if failure else 'pass','source_identity':source_identity(),'scopes':scopes,'maximum_steps':design['steps'],'failure':failure,'runs':runs,
 'contract':design['contract'],'created_utc':datetime.now(timezone.utc).isoformat(),'design':ref(designpath),'complete_seconds':time.perf_counter()-started,
 'cost_scope':'new admission work separate from science and retained parent ancestry; raw executions distinct, diagnostic categories overlap','common_unknown_engineering_seconds':None}
r=_write(output/'admission.json',report);print(json.dumps({'admission':ref(r.path),'status':report['status'],'failure':failure}));raise SystemExit(1 if failure else 0)
