"""Publish the measured inconclusive disposition and prospective exploratory protocol."""
import json,os,subprocess,statistics
from pathlib import Path
from datetime import datetime,timezone
from ralup.search import source_identity,select_backbone,_validate_recipe_metadata
from ralup.records import sha256_file
from ralup.calibrate import _write
root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011');out=Path(os.environ['M07_ATTEMPT_DIR'])
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
read=lambda p:json.loads(p.read_bytes())
now=datetime.now(timezone.utc).isoformat();revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
correction=base/'authenticate-recipe-metadata-v2/recipe-correction.json';audit={r['label']:r for r in read(correction)['rows']}
rows=[];forecast=[]
for native,query in [('native-ordinary-geometry','query-ordinary-geometry'),('native-quantizer-geometry','query-quantizer-geometry')]:
    admission=read(base/native/'admission.json');scores=read(base/query/'query-results.json')
    assert admission['status']=='pass' and scores['status']=='completed'
    for run in admission['runs']:
        c=run['config'];label=c['label'];points=[r for r in scores['rows'] if r['point'].get('label')==label]
        assert len(points)==2 and all(r['status']=='completed' and r['before_state_sha256']==r['after_state_sha256'] for r in points)
        recipe=audit[label]['resolved_numerical_recipe'];_validate_recipe_metadata(recipe)
        steps=[e['elapsed_seconds'] for p in sorted((base/native/label).glob('arm-*.jsonl')) for line in p.open() for e in json.loads(line)['events'] if e.get('status')=='completed_step']
        assert len(steps)==32
        forecast.append({'label':label,'physical_steps':len(steps),'mean_observed_step_seconds':statistics.mean(steps),'max_observed_step_seconds':max(steps),
            'hypothetical_paired_16_to_128_new_steps':224,'tail_compute_seconds_at_observed_mean':224*statistics.mean(steps),
            'tail_compute_seconds_at_observed_max':224*max(steps),'excludes':'future checkpoint/query/setup overhead, LR brackets, fresh seeds, matched FP32 finalist; forecast only, not an authorization or hardware impossibility'})
        rows.append({'trial_id':label,'evidence_kind':'real_data','source_revision':c['source_revision'],'source_identity':admission['source_identity'],
            'data_sha256':c['resolved']['data']['manifest']['sha256'],'model':'core','seed':c['seed'],'budget':16*8192,
            'recipe':recipe,'historical_metadata_correction':ref(correction),'parent':c['parent'],
            'curve':[{'step':p['point']['step'],'development_bpb':p['score']['bpb']} for p in points],
            'settled':False,'lr_bracketed':False,'full_budget':False,'edge':False,
            'cost':{'complete_seconds':sum(a['complete_seconds'] for a in run['arms'])+sum(p['complete_seconds'] for p in points),
                'scope':'local paired arms plus own queries; shared setup/parent/reference/project total is deduplicated in accounting, not this local sum',
                'complete_attempts':[ref(base/native/'finish.json'),ref(base/query/'finish.json')]},
            'admission':ref(base/native/'admission.json'),'query_report':ref(base/query/'query-results.json')})
rationale={'disposition':'resource-limited','reason':'Bounded M07 development disposition: completed the frozen 600s ordinary and 900s quantizer admission envelopes plus declared queries; stop additional initial-selection spending under the dated sequencing amendment. Measurements remain insufficient, not globally unaffordable.',
    'remaining_information':['fresh development seeds and seed variance','LR brackets for strong SOAP/Madam and quantizer alternatives','duration/rank calibration and matched FP32 finalist continuations from the selected strong parent','proxy-to-core transfer'],
    'remaining_cost_forecasts':forecast,'advance':[r['trial_id'] for r in rows],'expand_lr':[r['trial_id'] for r in rows],
    'preserved_unqualified':['QuEST plus full-Hadamard paired128 incomplete after frozen cap; not defeated'],
    'selection_scope':'initial M07 selection inconclusive; full ordinary B* competition remains E01-owned; no no-headroom or policy-gain claim',
    'output':str(out/'b0-disposition.json')}
selection=select_backbone(rows,rationale)
recipe=dict(audit['max-row-clipped']['resolved_numerical_recipe']);recipe['backward']='ste'
recipe['method_config']=audit['max-row-clipped']['actual_rule']['config'];_validate_recipe_metadata(recipe)
parents={}
for role,name,packing in [('target_train','native-derived-parent',32),('meta_train','native-meta-derived',64)]:
    p=base/name/'derived-warmstart.json';d=read(p);assert d['source_partition']==role and d['original_role']=='continuation'
    parents[role]={'artifact':ref(p),'parent_kind':'derived_model_only','model':d['model_config'],'packing':packing,
        'original_checkpoint':d['checkpoint'],'original_role':d['original_role'],'ancestry':'same partition AdamW32 then FP32 ProjectMuon256; one seed707, 288 historical updates / 2359296 target bytes',
        'resets':d['resets'],'import_proof':ref(base/name/'import-results.json') if (base/name/'import-results.json').exists() else ref(base/name/'finish.json'),
        'saved_cursor_and_rng':'copy authenticated cursor plus global/named RNG exactly into fresh state; no restart, no development/shared seed relabel',
        'model_semantics':'typed parent exact model source/config/dtype/parameter manifest; original whole-runtime identity is historical, never full-state restored under current source'}
exploratory={'schema':'ralup.exploratory_backbone.v1','created_utc':now,'research_cutoff':'2026-10-03','purpose':'M08-M11 engineering and development only',
    'status':'EXPLORATORY','selected_b0':False,'b0':None,'wp04':'inconclusive','confirmation_eligible':False,'policy_gain':None,
    'source_revision':revision,'source_identity':source_identity(),'revision_scope':'HEAD plus exact current source_identity; later commit records these bytes, no future revision invented',
    'runtime':ref(base/'applicability-v5/runtime-observation.json'),'lock':ref(root/'requirements.lock'),
    'recipe':recipe,'parents':parents,'training_seed':707,'logical_batch_targets':8192,'dtype':'FP32','embedding':'shared OrderedEmbedding',
    'format':'full canonical signed E2M1 including +/-6, FP32 per-row max/6 scale, equal recipe/export/scale information across matched arms',
    'progress_contract':{'continuation_step_initial':0,'continuation_horizon':None,'must_freeze_horizon_before_each_policy_run':True,
        'feature13':'actual continuation schedule.step / explicitly declared continuation horizon; never inherited 288 ledger updates, project expenditure, or local rollout k/K'},
    'scientific_final_budgets':None,'numerical_applicability':[ref(base/f'applicability-v5/admission-applicability-{n}.json') for n in (16,32,128,256)],
    'proof_scope':'ProjectMuon row-max/identity/STE proxy32/core256 numerical proofs have their own historical weak-parent origins; typed strong target/meta imports separately proved. New policy compositions/longer scopes require their owning prospective risk-based checks. No strong-parent long-horizon selection claim.',
    'choice_reason':'Existing exact ProjectMuon operation/parity and longest matched-reference/LR-duration evidence; lower measured identity-transform cost than full-Hadamard, original-W moments and row-scale causal features directly defined. This is an engineering starting point, not the lowest short BPB or largest available headroom; adapter limitations do not select B0.',
    'preserve_for_E01':['SOAP refresh10 and Madam stronger short curves','QuEST row/tensor trust','max-scale clipped STE','full-Hadamard rotation','Muon+ named normalization variants','AdamW and matched FP32 references','unattempted decay/beta/schedule/refresh and unsettled rate/duration/seed scopes'],
    'parent_protocol':'Meta is genuinely trained on meta_train; meta-development is selection-only, never an outer-gradient source. Target is target_train. Fresh evaluation seeds reproduce the mixed protocol and cannot share this development parent as independent seeds.',
    'cost_contract':'Original physical ancestry counted once globally; inherited trajectory/project receipt views overlap. Charge new import/selection/rollout/setup/query costs; unknown common engineering remains unknown.',
    'selection':{'path':str(selection.path),'sha256':selection.sha256},
    'sequencing_amendment':ref(root/'docs/harness/m07-sequencing-amendment-20261011.md'),
    'later_change':'Prospective compatibility/transfer assessment or diagnosed retraining, with all prior costs retained; no old controller/backbone relabel; strongest-recipe claim requires robust B0/E01 or separately supported final disposition'}
_write(out/'exploratory-backbone.json',exploratory)
index={'created_utc':now,'disposition':ref(out/'b0-disposition.json'),'exploratory_backbone':ref(out/'exploratory-backbone.json'),
    'recipe_correction':ref(correction),'new_training_targets':0,'new_query_targets':0,
    'evidence':[ref(base/name/file) for name,file in [('native-ordinary-geometry','admission.json'),('query-ordinary-geometry','query-results.json'),('native-quantizer-geometry','admission.json'),('query-quantizer-geometry','query-results.json'),('query-prioritized-views-v2','query-results.json'),('native-meta-derived','derived-warmstart.json')]],
    'source':source_identity(),'scope':'analysis of completed receipts only; protected test untouched'}
_write(root/'docs/harness/m07-closing-evidence.json',index)
print(json.dumps({'index':ref(root/'docs/harness/m07-closing-evidence.json'),'b0':None,'wp04':'inconclusive','exploratory':ref(out/'exploratory-backbone.json'),'cost_forecasts':forecast}))
