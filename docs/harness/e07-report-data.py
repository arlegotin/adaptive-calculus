"""Deterministically derive current exploratory report inputs from retained raw records."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from ralup.report import SECTIONS,COST_FIELDS
from ralup.config import canonical_bytes
from ralup.records import sha256_file

ROOT=Path.cwd();OUT=Path(os.environ['E07_OPERATION']);BASE=Path(os.environ['E07_BASE'])
STORE=BASE.parent
COMPLETION=Path(os.environ.get('E07_COMPLETION_ROOT',str(BASE)))
sources={};rows=[];intervals={}


def ref(path):
    path=Path(path).resolve();r={'path':str(path),'sha256':sha256_file(path)};sources[str(path)]=r;return r


def auth(r):
    assert sha256_file(Path(r['path']))==r['sha256'],r['path']
    sources[str(Path(r['path']).resolve())]=dict(path=str(Path(r['path']).resolve()),sha256=r['sha256'])
    return json.loads(Path(r['path']).read_bytes())


def add_row(name,source,status='completed',**extra):
    rows.append(dict(run_id=name,status=status,seed=None,provenance='exploratory',source=source,
        development_bpb=None,cost={k:None for k in COST_FIELDS},**extra))


supplement=auth(ref(Path(os.environ['E07_NATIVE_SUPPLEMENT']))) if os.environ.get('E07_NATIVE_SUPPLEMENT') else None

# All registered attempts, including failures, holds and partials. These are
# operation rows; their walls are NOT summed with nested model operation walls.
for name in ('m07-accounting-v1','m11-evidence','e01-evidence','e01-fix1-evidence',
             'e02-evidence','e02-fix1-evidence','e03-evidence','e04-evidence','e05-evidence','e06-evidence'):
    path=ROOT/'docs/harness'/f'{name}.json'
    if not path.exists():continue
    source=ref(path);value=auth(source)
    for i,attempt in enumerate(value.get('attempts',[])):
        status='failed' if attempt.get('exit_code') not in (0,None) else 'completed'
        add_row(f'{name}:{attempt.get("attempt",attempt.get("name",i))}',source,status,
            raw_operation=attempt,accounting_scope='operation registry; nested clocks and historical statuses retained, not summed')

for root in dict.fromkeys((BASE,COMPLETION)):
    for path in sorted(root.glob('*/finish.json')):
        source=ref(path);finish=auth(source)
        status='completed' if finish['exit_code']==0 and finish['bounds_pass'] else 'failed'
        if path.parent.name=='native-replay-v2' and supplement and not supplement['child_component_bound_pass']: status='failed'
        add_row('E07:'+path.parent.name,source,status,raw_operation=finish,
            supplemental_disposition=supplement if path.parent.name=='native-replay-v2' else None,
            accounting_scope='engineering/admin capture; child exit and enclosing storage admission are independent; no new scientific final-seed evidence')

# Genuine measured M3 exploratory META rollouts: final_query is mean NLL, NOT
# official validation BPB or independent final-seed evidence.
native_index=auth(ref(STORE/'m11-20261011/native-final-calibration-v1/report.json'))
for r in native_index['runs']:
    value=auth(r);result=value['result'];work=result['work']
    add_row('M11_META:'+value['name'],sources[str(Path(r['path']).resolve())],
        meta_query_nll=result['final_query'],initial_meta_query_nll=result['initial_query'],
        family=value['family'],horizon=value['horizon'],
        observed_rollout_seconds=work['elapsed_seconds'],
        query_targets=work['query_targets'],training_targets=work['training_targets'],
        objective=result['objective'],input_sha256=work['input_sha256'],
        accounting_scope='same shared META parent and held panels, not fresh independent full-budget seeds; warm ancestry excluded from nested rollout wall, full deployment unknown')
    rows[-1]['cost'].update(stream_targets=work['training_targets'],
        loss_targets=work['training_targets']+work['query_targets'])

def add(document,a,b):
    assert isinstance(document,str) and ':' in document and 0<=a<=b
    intervals.setdefault(document,[]).append((a,b))


def offsets(value):
    if isinstance(value,dict):
        if 'document_id' in value and 'offsets' in value:
            values=value['offsets'];start=end=None
            for n in values:
                if end==n:end+=1
                else:
                    if start is not None:add(value['document_id'],start,end)
                    start=n;end=n+1
            if start is not None:add(value['document_id'],start,end)
        for v in value.values():offsets(v)
    elif isinstance(value,list):
        for v in value:offsets(v)


# PROJECT union merges authenticated source identities, never sums task uniques.
for name in ('h08-project-coverage','n01-project-coverage'):
    value=auth(ref(ROOT/'docs/harness'/f'{name}.json'))
    for doc,spans in value['intervals'].items():
        for a,b in spans:add(doc,a,b)
m01=BASE/'preservation-v1/m01-archive/task-offset-union.json'
for doc,n in auth(ref(m01))['offsets']:add(doc,n,n+1)
m11=auth(ref(STORE/'m11-20261011/receipt-accounting-v1/report.json'))
for doc,spans in auth(m11['source_offsets'])['documents'].items():
    for a,b in spans:add(doc,a,b)
m07=auth(ref(ROOT/'docs/harness/m07-accounting-v1.json'))
for r in m07['sources']:
    path=Path(r['path']);assert sha256_file(path)==r['sha256'];ref(path)
    if path.suffix=='.jsonl':
        with path.open() as f:
            for line in f:offsets(json.loads(line))
    else:offsets(json.loads(path.read_bytes()))
native_replay=auth(ref(COMPLETION/'native-replay-v2/paired-replay.json'))
offsets(native_replay)
merged={}
for doc,spans in sorted(intervals.items()):
    union=[]
    for a,b in sorted(spans):
        if union and a<=union[-1][1]:union[-1][1]=max(b,union[-1][1])
        else:union.append([a,b])
    merged[doc]=union
union={'schema':'ralup.e07_source_union.v1','documents':merged,
    'known_union_targets':sum(b-a for spans in merged.values() for a,b in spans),
    'complete_project_unique_targets':None,
    'scope':'union over authenticated H08/N01/M01/M07/M11/E07 recorded source intervals; unknown/missing earlier work not recovered; corpus content identity remains document prefix',
    'physical_passes':'source uniqueness is not loss/gradient/HVP execution multiplicity'}
(OUT/'source-union.json').write_bytes(canonical_bytes(union));union_ref=ref(OUT/'source-union.json')

history=auth(ref(BASE/'preservation-v1/history-mapping.json'))
preserved={Path(r['original_path']).name:r for r in history['files']}
def historical(name):
    item=preserved[name];assert sha256_file(Path(item['durable_path']))==item['sha256']
    return ref(item['durable_path'])

sections={k:[] for k in SECTIONS}
sections['frozen_scope']={'research_cutoff':'2026-10-03','observation_period':'2026-10-03 through2026-10-12',
    'outcome':'exploratory-only terminal; WP04 inconclusive/B0 null, WP06 NEEDS_REVISION',
    'absent':['final C','learned LR','robust B_star','final budgets','pilot variance','locked n','confirmation manifest','protected scores']}
sections['sources_and_deviations']={'primary_ledgers':[historical(n) for n in ('baseline-sources.md','numerics-report.md','dataset-card.md')],
    'limitations':['R41 unresolved primary-source access, no existence/author/priority inference',
        'QuEST and comparator implementations independently implement published math; release fidelity/software reuse grants unresolved',
        'dataset card CC3 versus proseCC4/GFDL version discrepancy retained; local research, no corpus redistribution grant',
        'supplied Linux prototype checks remain historical reports, never new CPU/M3 measurements']}
sections['provenance']={'evidence_classes':['supplied_historical','measured_cpu','measured_m3','inferred','simulated','exploratory','confirmatory'],
    'current_scope':'new CPU tests and finite M3 replay engineering; prior scientific evidence remains exploratory',
    'core_lock':ref(ROOT/'requirements.lock'),'environment_audit':ref(BASE/'preservation-proof-v2/environment-audit.json'),
    'history':ref(BASE/'preservation-v1/history-mapping.json'),
    'rulings':historical('root-ruling-index-v7.md'),'late_root_delta':'mandatory after E07/final review and before only-own-plan scratch cleanup'}
if os.environ.get('E07_REPORT_SOURCE_APPLICABILITY'):
    sections['provenance']['post_review_source_applicability']=ref(Path(os.environ['E07_REPORT_SOURCE_APPLICABILITY']))
    sections['provenance']['historical_numerical_scope']='Original ALL78 equality holds only at its captured pre-fix boundary; report.py now has a new source identity. Historical CPU1250/native proof is not final-current-tree or cross-source restore evidence.'
sections['per_seed']={'raw_rows':'tables.per_seed','independent_confirmation_seeds':0,
    'within_seed_meta_rollouts':'shared-parent panel reuse is not independent training-recipe replication',
    'effects_and_uncertainty':'No justified confirmatory paired intervals; M11 raw endpoint objectives remain available by horizon/input hash'}
sections['development_searches']={'all_attempt_registry':'tables.per_seed; retained failures/holds and raw source indexes',
    'final_selection':auth(ref(ROOT/'docs/harness/e01-evidence.json'))['selection'],
    'quality_curve_scope':'M11 same-META-panel query NLL by measured nested rollout wall and training targets, not final development BPB'}
sections['cost_ledgers']={'deployment':{'total':None,'reason':'no finalized trained arms/complete physical warm ancestry'},
    'search':{'m07':m07['distinct_real_data_counts'],'m11':m11['physical_work'],
        'shared_parent':'M11 inherited2359296 targets/288updates created once in M07; never multiply by clone/rollout count'},
    'project':{'complete_total':None,'known_source_union':union_ref,'known_source_union_targets':union['known_union_targets'],
        'unknown':['early editing','tools','reviews','partial/missing physical pass IDs','unarchived historical cost'],
        'walls':'nested component walls not added to launcher; unrelated clocks not presented as a summed complete wall'},
    'classifications':'query/backward/additional_loss overlap; loss,gradient-only,HVP physical executions are separate',
    'break_even':'only comparable quality plus positive full deployment saving; meta and method-specific tuning reported separately; currently null'}
sections['ablations']={'status':'WP08 not attempted; no genuinely eligible trained finalists',
    'catalogs':'configs/ablations; E05 engineering contracts exist, not empirical attribution',
    'record':ref(ROOT/'docs/harness/e05-evidence.json')}
sections['regressions']={'status':'genuine independent seed/article regression evidence absent; E05 arithmetic tested with synthetic controls',
    'no_meta_recipe_claim':True}
sections['export_and_reproduction']={'export':ref(ROOT/'docs/harness/e06-evidence.json'),
    'E06_dual_root':ref(BASE/'e06-proof-v3/e06-proof.json'),
    'M01_exact_member_proof':ref(BASE/'preservation-proof-v2/m01-extraction-proof.json'),
    'scope':'initialized E06 export mechanics, not selected research model; historical source-strict restore remains enforced',
    'new_replay':ref(COMPLETION/'native-replay-v2/paired-replay.json') if (COMPLETION/'native-replay-v2/paired-replay.json').exists() else None}
sections['export_and_reproduction']['native_storage_disposition']=supplement
sections['remaining_blocks']=['WP04 robust backbone unresolved','WP06 revise genuine learning and stable horizon/rank behavior',
    'E01 final learned/LR/static selection absent','E02 justified paired pilot/design absent','E03 protected gateway closed',
    'R41/source/licensing/fidelity limitations retained','all original review Declined owner dispositions preserved in history',
    'late root review/ruling capture and final branch review remain required']
sections['decision']={'research':'inconclusive / needs_revision; no H1/H2 or efficiency claim',
    'engineering':'report/reproduction infrastructure separately verified; task closure never opens a scientific gate'}

from scipy.stats import t
precision=[{'n':n,'half_width_in_units_of_paired_sd':float(t.ppf(.9875,n-1)/(n**.5)),
    'provenance':'analytical illustration, not empirical pilot','interval_confidence':.975} for n in (4,8,16,32,64,128,256)]
(OUT/'precision-illustration.json').write_text(json.dumps(precision,sort_keys=True)+'\n')
ref(OUT/'precision-illustration.json')
sections['per_seed']['precision_illustration']='Two-sided97.5% Student-t half-width in units of unknown paired SD only. No observed SD, estimated power, selected n or usable final-arm forecast.'
(OUT/'trials.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False))
evidence={'schema':'ralup.research_evidence.v1','scope':'authentic exploratory terminal evidence; diagnostic META curves are not final estimands',
    'registry':[r['run_id'] for r in rows],'trials':ref(OUT/'trials.json'),'sources':list(sources.values()),
    'scientific_status':{'wp04':'inconclusive','B0':None,'wp06':'NEEDS_REVISION','C':None,'LR':None,'B_star':None,'budgets':None,'n':None,'confirmation':None},
    'sections':sections,'precision':precision,'claim':{'outcome':'inconclusive'}}
path=OUT/'evidence.json';path.write_bytes(canonical_bytes(evidence));digest=sha256_file(path)
commands=[['-m','ralup','report','--evidence',str(path),'--sha256',digest,'--output',str(OUT/'report')],
    ['-m','ralup','reproduce','--evidence',str(path),'--sha256',digest,'--output',str(OUT/'reproduction')]]
for n,command in enumerate(commands):
    result=subprocess.run([sys.executable,'-B',*command],capture_output=True,text=True)
    (OUT/f'cli-{n}-stdout.txt').write_text(result.stdout);(OUT/f'cli-{n}-stderr.txt').write_text(result.stderr)
    assert result.returncode==0,(command,result.stderr)
result=subprocess.run(['sh','reproduce.sh',sys.executable,str(path),digest,str(OUT/'shell-reproduction')],capture_output=True,text=True)
(OUT/'shell-stdout.txt').write_text(result.stdout);(OUT/'shell-stderr.txt').write_text(result.stderr)
assert result.returncode==0,result.stderr
print(json.dumps({'evidence':str(path),'sha256':digest,'trial_rows':len(rows),'source_refs':len(sources),'known_union_targets':union['known_union_targets'],'report_and_reproduce_cli':0,'shell_reproduce':0}))
