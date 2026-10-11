"""Freeze initial M11 actual-data diagnostics BEFORE observing their outcomes."""
import copy,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from ralup.search import source_identity
from ralup.records import sha256_file
from ralup.policy.checks import read_exploratory_backbone,_write
root=Path.cwd();store=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03')
ref=lambda p:{'path':str(Path(p).resolve()),'sha256':sha256_file(p)}
protocol,backbone=read_exploratory_backbone(store/'m07-20261011/closing-disposition-v2/exploratory-backbone.json')
c=copy.deepcopy(json.loads((root/'docs/harness/m07-meta-derived-design-v1.json').read_bytes())['fresh_config'])
c.update(partition='meta_train',parent=protocol['parents']['meta_train']['artifact'],parent_kind='derived_model_only',budget=512*8192,
 source_identity=source_identity(),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),recipe=protocol['recipe'],milestones=[4,16,32],role='continuation',purpose='development-bootstrap',finite_forecast_seconds=1800,label='m11-proxy-initial-diagnostic')
c.pop('admission',None);c.pop('operation_applicability',None)
prior=ref(store/'m07-20261011/applicability-v6/admission-applicability-32.json');admission=json.loads(Path(prior['path']).read_bytes())
unchanged=[n for n in source_identity() if n.startswith(('src/ralup/models/','src/ralup/quantization/','src/ralup/optim/'))]+['src/ralup/train.py','src/ralup/evaluate.py','src/ralup/warmstart.py','src/ralup/data/cursor.py','src/ralup/policy/actions.py']
m09=json.loads((root/'docs/harness/m09-controller-design-v1.json').read_bytes())
m09_owned=['src/ralup/optim/base.py','src/ralup/optim/matrix_adapter.py','src/ralup/train.py']
assert all((m09 if n in m09_owned else admission)['source_identity'][n]==source_identity()[n] for n in unchanged)
d={'schema':'ralup.m11_bootstrap_design.v1','created_utc':datetime.now(timezone.utc).isoformat(),'purpose':'exploratory-development-bootstrap','config':c,
 'source_identity':source_identity(),'backbone':backbone,'prior_admission':prior,'M09_owned_sources':m09_owned,'M09_scope':ref(root/'docs/harness/m09-controller-design-v1.json'),'unchanged_numerical_sources':unchanged,
 'helpers':{str(p):ref(p) for p in map(Path,['docs/harness/m11-native.py','docs/harness/m11-attempt.py','docs/harness/m11-supervise.py','docs/harness/m11-freeze.py'])},
 'bootstrap_normalizer':json.loads((root/'docs/harness/m08-feature-evidence.json').read_bytes())['normalizer'],
 'runtime':{'macOS':'26.7.1','build':'25G241','architecture':'arm64','fallback':0,'lock_sha256':sha256_file('requirements.lock')},
 'continuation_horizon':512,'reference_steps':32,'snapshot_steps':[4,16,32],
 'reference_origins':[{'name':'baseline','bias':[0.,0.,1.]},{'name':'policy','bias':[.25,.09531017980432486,.875]}],
 'pool_scope':'initial/bootstrap age spread4/16/32 only, not final early/middle/late512 pool; no full-state cross-source import',
 'reference_fit':'META-TRAIN baseline and fixed policy origin raw14 rows, 768 total expected, freeze after full reference acquisition; assess final coverage later',
 'probe':{'horizon':16,'pairs':4,'sigma':.05,'outer_lr':.01,'seed':1111,'query_seed':1112,'query_targets':8192,'duplicates':2,'snapshot_origin':'baseline','snapshot_step':4},
 'lr49_admission':{'h_values':[.25,32.],'steps':2,'hidden_initialization_seed':1113,'scope':'actual14→3→1 max32 proxy neural forward/update composition; endpoints forced output bias, hidden retained; bounded2steps only'},
 'numeric_acceptance':{'duplicate_objective_atol':0.,'duplicate_state':'exact H07','assistance_bound_atol':0.,'assistance_bound_rtol':2.384185791015625e-7,'parameter_finite':True},
 'forecast_seconds':1800,'hard_deadline_seconds':1860,'forecast_bytes':2*1024**3,'maximum_updates':228,'maximum_query_targets':196608,
 'forecast_basis':'M09 16mixed core/proxy physical updates plus replay/import in138.91s; proxy only228updates with endpoint queries conservatively1800s; scope will stop rather than assume feasibility',
 'retention':'Six reference milestones4/16/32, fitted-normalizer rebound complete baseline4 task snapshot; successful rollout finals disposable/reconstructible from frozen source+complete snapshot+parameters+inputs+RNG; all step/query/failure receipts retained. No old objects removed.',
 'claims':'B0null/WP04inconclusive/confirmation-ineligible; preliminary future-loss/noise only; no selection, gain, final normalizer or robust horizon claim',
 'cost':'All physical work retained;288ancestorupdates2359296targets inherited and globally deduplicated, not fresh seed; parent707, query1112, perturbation1111. Common early engineering wall unknown.'}
_write(root/'docs/harness'/sys.argv[1],d);print(json.dumps(ref(root/'docs/harness'/sys.argv[1])))
