"""Prospective M09 native scope: core identity, proxy changed rotated, replay."""
import copy,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from ralup.search import source_identity,operation_contract
from ralup.records import sha256_file
from ralup.policy.checks import read_exploratory_backbone,_write
root=Path.cwd();store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03')
ref=lambda p:{'path':str(Path(p).resolve()),'sha256':sha256_file(p)}
index=json.loads((root/'docs/harness/m07-closing-evidence.json').read_bytes())['exploratory_backbone']
protocol,backbone=read_exploratory_backbone(index['path'])
prior=ref(store/'m07-20261011/applicability-v6/admission-applicability-32.json');admission=json.loads(Path(prior['path']).read_bytes())
unchanged=[name for name in source_identity() if name.startswith(('src/ralup/models/','src/ralup/quantization/'))]+[
 'src/ralup/optim/adamw.py','src/ralup/optim/muon.py','src/ralup/optim/soap.py','src/ralup/optim/madam.py','src/ralup/optim/muon_plus.py',
 'src/ralup/data/cursor.py','src/ralup/data/batches.py','src/ralup/evaluate.py','src/ralup/warmstart.py','src/ralup/calibrate.py','src/ralup/policy/actions.py']
assert all(admission['source_identity'][name]==source_identity()[name] for name in unchanged)
cases=[]
for partition,template,mode,bias,rotation in [('target_train','m07-derived-parent-design-v1.json','initialized_identity',[0.,0.,1.],'identity'),('meta_train','m07-meta-derived-design-v1.json','changed_rotated',[.5,.1823215567939546,.5],'full_hadamard')]:
 c=copy.deepcopy(json.loads((root/'docs/harness'/template).read_bytes())['fresh_config'])
 c.update(partition=partition,parent=protocol['parents'][partition]['artifact'],parent_kind='derived_model_only',budget=16*8192,
          source_identity=source_identity(),source_revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
          recipe={**protocol['recipe'],'rotation':rotation},milestones=[2],role='continuation',purpose='development-bootstrap',finite_forecast_seconds=600,
          label='m09-'+partition+'-'+mode)
 c.pop('admission',None);c.pop('operation_applicability',None)
 inherited={**c,'recipe':{**c['recipe'],'rotation':'identity'}}
 assert any(s['contract']==operation_contract(inherited) and s['maximum_steps']>=4 for s in admission['operation_scopes'])
 cases.append({'partition':partition,'mode':mode,'bias':bias,'config':c})
runtime=[]
for command in [['sw_vers'],['uname','-m']]:
 r=subprocess.run(command,capture_output=True,text=True,check=True);runtime.append({'command':command,'stdout':r.stdout,'stderr':r.stderr,'exit_code':r.returncode})
design={'schema':'ralup.m09_controller_design.v1','created_utc':datetime.now(timezone.utc).isoformat(),'purpose':'exploratory-development-bootstrap',
 'source_identity':source_identity(),'backbone':backbone,'prior_admission':prior,'normalizer':json.loads((root/'docs/harness/m08-feature-evidence.json').read_bytes())['normalizer'],
 'helper':ref(Path(__file__)),'supervisor':ref(root/'docs/harness/m09-supervise.py'),'unchanged_numerical_sources':unchanged,'runtime_commands':runtime,'fallback':0,
 'cases':cases,'continuation_horizon':16,'episode_steps':4,'resume_step':2,'trace_cadence':1,'finite_forecast_seconds':600,'hard_child_deadline_seconds':660,'forecast_bytes':2*1024**3,
 'forecast_basis':'M08 measured complete16physical step diagnostics/import/checkpoint costs support conservative600seconds;4 retained snapshots(midpoint2/final4 eachcase),full native receipts/producer/temporary tensors≤2GiB. Small source/Git transaction≤16MiB retains512MiB',
 'maximum_actual_training_updates':16,'maximum_actual_training_targets':16*8192,'new_query_targets':0,
 'identity':'same typed parent, full14feature mode, selected normalization, packing, original-W moments/aux/decay/cursor/schedule/RNG and scientific work. Independent frozen-controller/identity-telemetry checks permit ONLY explicitly enumerated metadata projection; strict policy replay unmodified H07 hash',
 'nonidentity':'one changedAffine action alpha.5/h1.2/tau.5 on actual proxy/meta rotated W;moments originalW. Two NEW hand-sized rotated CPU/MPS composition probes also include LRh32. No Cartesian family grid or new science selection',
 'tolerances':{'rotation_atol':8e-6,'rotation_rtol':2e-6,'assistance_bound_atol':0.,'assistance_bound_rtol':2.384185791015625e-7,'same_device_identity':'exact shared-owner projection','same_device_resume':'exact strict H07'},
 'normalization':'selectedM08 reference normalizer is diagnostic96meta_trainrows only;unchanged file frozen attached for all controls.No fitting/scoring protected data;M11owns final learned-distribution normalization',
 'retention':'NEWtask-ownedDatafolders:eachcase midpoint2 andcompleted4 + reconstructible baseline/step/replayreceipts;preserve all prior artifacts/failures;no deletion',
 'cost':'16new physical updates131072targets0queriesmaximum;inherited288updates2359296targets pertyped ancestry retainedonceineachstate.All import/setup/scans/action/replay/checkpoint/failure costs retainedinwholeattempt;editing/enclosing/toolcost unknown',
 'limits':'WP05scopedengineeringonly.No acceptedB0/WP04/S31,strongestselection,policybenefit,learneddistribution,longhorizon,universalnewactionrangeorproxy-to-coreclaim'}
p=root/'docs/harness'/sys.argv[1];_write(p,design);print(json.dumps(ref(p)))
