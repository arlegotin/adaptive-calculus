"""Publish completed M09 evidence pointers; active administrative stdout excluded."""
import json,os
from pathlib import Path
from ralup.records import sha256_file
from ralup.config import canonical_bytes
from ralup.search import source_identity
root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m09-20261011')
ref=lambda p:{'path':str(Path(p).resolve()),'sha256':sha256_file(p)}
native=base/'native-controller-diagnostic-v1';report=json.loads((native/'controller-evidence.json').read_bytes())
assert report['status']=='pass' and report['source_identity']==source_identity()
assert report['new_training_updates']==16 and report['new_training_targets']==131072 and report['new_query_targets']==0
attempts=[]
for name in ('cpu-support','freeze-native-v1','native-controller-diagnostic-v1'):
 folder=base/name;finish=json.loads((folder/'finish.json').read_bytes());startup=json.loads((folder/'startup.json').read_bytes())
 assert finish['exit_code']==0
 for obj in finish['completed_objects']:
  p=folder/obj['path'];assert p.stat().st_size==obj['bytes'] and sha256_file(p)==obj['sha256']
 attempts.append({'name':name,'startup':ref(folder/'startup.json'),'finish':ref(folder/'finish.json'),
  'stdout':ref(folder/'stdout.txt'),'stderr':ref(folder/'stderr.txt'),'runtime':ref(folder/'runtime-observation.json'),
  'utc':startup['utc'],'complete_seconds':finish['elapsed_seconds'],'retained_completed_object_bytes':sum(o['bytes'] for o in finish['completed_objects']),
  'git_free_before':startup['storage']['git_free_before'],'git_free_after':finish['git_free_after'],'data_free_before':startup['storage']['data_free_before'],'data_free_after':finish['data_free_after'],
  'store_bytes_before':startup['storage']['allocated_bytes'],'forecast_bytes':startup['storage']['complete_remaining_fix_forecast_bytes']})
value={'schema':'ralup.m09_controller_index.v1','status':'COMPLETE','task_base':'e5fa9996535da63b914a1babf2169bf899367a61',
 'source_identity':report['source_identity'],'primary':ref(native/'controller-evidence.json'),'design':ref(root/'docs/harness/m09-controller-design-v1.json'),
 'supervision':ref(native/'supervision.json'),'attempts':attempts,'backbone':report['backbone'],'B0':None,'WP04':'inconclusive','WP05':'scoped engineering pass',
 'normalizer':report['normalizer'],'capacities':report['capacities'],'composition':report['composition'],
 'cases':[{k:v for k,v in c.items() if k not in ('steps','replay','baseline_steps')} for c in report['cases']],
 'identity_projection_exclusions':report['identity_projection_exclusions'],
 'new_training_updates':16,'new_training_targets':131072,'new_query_targets':0,
 'physical_attempt_wall_sum_seconds':sum(a['complete_seconds'] for a in attempts),
 'wall_sum_contract':'separate physical attempt launches only; child/setup/import/scans/action/checkpoints/replay already inside enclosing attempt, never add again. Earlier sandbox/support runs retain exact stdout but no pre-import archive; editing/tool/enclosing costs unknown.',
 'retention':'four declared snapshots(midpoint2/final4 eachcase),full native baseline/step/replay receipts and exact producer/source/lock/F02/runtime archive.No old artifact deletion/move/rewrite.',
 'limits':report['limits'],'LR32_scope':'CPU49controller reachable maximum plus hand-sized rotated native additivecomposition only;no neural49-family LRh32rollout or learned49training claim',
 'assistance_scope':'actual changedrotated proxy alpha.5/h1.2/tau.5 exercised62850acceptednudges/0rejections/0acceptedboundexcess across original4steps;handcomposition probes exercised0acceptednudges;no universal rejected/unrepresentable branch or broad action range claim',
 'training_scope':'Affine45/LR49/Static3 and removals/restricted/privileged families are implemented and CPUserialized;none are future-loss trained here.M11owns actual outer learning/finalnormalizer/currentcomplete snapshot pools.',
 'active_administrative_stdout':'current M09 index publisher stdout is excluded from immutable completed refs; later final publication/commit receipts are separate, no circular self-reference'}
p=root/'docs/harness/m09-controller-evidence.json'
with p.open('x') as f:f.write(canonical_bytes(value).decode()+'\n');f.flush();os.fsync(f.fileno())
print(json.dumps(ref(p)))
