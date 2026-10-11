"""Freeze the narrow actual M08 experiment before any native feature outcome."""
import copy,json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
from ralup.search import source_identity,operation_contract
from ralup.records import sha256_file
from ralup.policy.checks import read_exploratory_backbone,_write
root=Path.cwd();store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03')
ref=lambda p:{'path':str(Path(p).resolve()),'sha256':sha256_file(p)}
index=json.loads((root/'docs/harness/m07-closing-evidence.json').read_bytes())['exploratory_backbone']
protocol,backbone=read_exploratory_backbone(index['path'])
prior=ref(store/'m07-20261011/applicability-v6/admission-applicability-32.json')
admission=json.loads(Path(prior['path']).read_bytes())
unchanged=[name for name in source_identity() if name.startswith(('src/ralup/models/','src/ralup/quantization/'))]+[
 'src/ralup/optim/adamw.py','src/ralup/optim/muon.py','src/ralup/optim/soap.py','src/ralup/optim/madam.py','src/ralup/optim/muon_plus.py','src/ralup/baselines/registry.py',
 'src/ralup/data/cursor.py','src/ralup/data/batches.py','src/ralup/evaluate.py','src/ralup/warmstart.py','src/ralup/calibrate.py']
assert all(admission['source_identity'][name]==source_identity()[name] for name in unchanged)
cases=[]
for partition,template in [('meta_train','m07-meta-derived-design-v1.json'),('target_train','m07-derived-parent-design-v1.json')]:
 c=copy.deepcopy(json.loads((root/'docs/harness'/template).read_bytes())['fresh_config'])
 c.update(partition=partition,parent=protocol['parents'][partition]['artifact'],parent_kind='derived_model_only',budget=16*8192,
          source_identity=source_identity(),source_revision=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
          recipe=protocol['recipe'],milestones=[4],role='continuation',purpose='development-bootstrap',finite_forecast_seconds=600,
          label='m08-'+partition+'-feature-diagnostic')
 c.pop('admission',None);c.pop('operation_applicability',None)
 assert any(s['contract']==operation_contract(c) and s['maximum_steps']>=8 for s in admission['operation_scopes'])
 cases.append(c)
runtime=[]
for command in [['sw_vers'],['uname','-m']]:
 r=subprocess.run(command,capture_output=True,text=True,check=True);runtime.append({'command':command,'stdout':r.stdout,'stderr':r.stderr,'exit_code':r.returncode})
design={'schema':'ralup.m08_feature_design.v1','created_utc':datetime.now(timezone.utc).isoformat(),
 'purpose':'exploratory-development-bootstrap','source_identity':source_identity(),'backbone':backbone,'prior_admission':prior,
 'helper':ref(Path(__file__)),'supervisor':ref(root/'docs/harness/m08-supervise.py'),'unchanged_numerical_sources':unchanged,
 'runtime_commands':runtime,'fallback':0,'cases':cases,'continuation_horizon':16,'episode_steps':8,'resume_step':4,'trace_cadence':4,
 'finite_forecast_seconds':600,'hard_child_deadline_seconds':660,'forecast_bytes':2*1024**3,
 'forecast_basis':'earlier two-arm16 core geometry cases completed roughly64s each plus39.6s typed-core import; meta smaller.600s includes all import/feature scans/CPU-safe midpoint+last-two checkpoints/strict replay/fit;2GiB covers six snapshots plus receipts/source/temp',
 'maximum_actual_training_targets':24*8192,'new_query_targets':0,'maximum_actual_training_updates':24,
 'modes':'full14 feature/history/new-original-W moment scans for both actual reference and target; raw ordinary defaults unchanged. Future matched RALUP/LR/static/single-action/information ablations must all enable same mode; no silent scan or common-EMA omission',
 'normalization':'META-TRAIN original8 reference trajectory only, one population mean/std fit with1e-6floor;freeze before core target. Strict replay rows never multiply fit samples. All target clipping outcomes retained; no target/development/protected fit',
 'retention':'per partition checkpoint milestone4 and completed7/8 only; reconstructible every-step raw work receipts. Strict restored4-to8 has no every-step checkpoint; no old artifacts modified/deleted/moved',
 'risk':'unchanged exact primitive model/Muon/quantizer/STE/identity/cursor/scorer sources + authenticated v6operation scopes. NEW feature/history/normalizer/finiteH07owners checked on actual typed-proxy/meta and typed-core/target parents;strict pending4-to8 replay. This grants no old-source full-state restore or universal longer/policy-composition numerical claim',
 'cost':'24actual physical logical updates at most,196608target bytes and zero queries; inherited288updates/2359296targets per typed ancestry remain historical expenditure; no local rollout or inherited-ledger feature13. All import/setup/fit/replay/I/O/failure costs stay in completeattempt; unknown interactive engineering is unknown',
 'limits':'eight-step feature overhead/reference-normalizer/clipping/replay diagnostic in intended16stepcontinuation, not final controller distribution, proxy-to-core transfer, policy benefit or accepted B0/WP04/S31;E01strongest selection pending'}
p=root/'docs/harness/m08-feature-design-v1.json';_write(p,design);print(json.dumps(ref(p)))
