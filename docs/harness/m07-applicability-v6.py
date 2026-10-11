"""11 October zero-pass exact-operation binding of unchanged historical proofs."""
import copy,difflib,json,os
from pathlib import Path
from datetime import datetime,timezone
from ralup.search import source_identity,operation_contract,_validate_admission
from ralup.quantization.scales import grouping_for
from ralup.records import sha256_file
from ralup.calibrate import _write
root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011');out=Path(os.environ['M07_ATTEMPT_DIR'])
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
def read(p):return json.loads(Path(p).read_bytes())
def auth(r):
    assert sha256_file(r['path'])==r['sha256'];return read(r['path'])
old=read(base/'applicability-v5/admission-applicability-256.json')['source_identity'];new=source_identity()
changed={k:{'before':old.get(k),'after':new.get(k)} for k in sorted(set(old)|set(new)) if old.get(k)!=new.get(k)}
assert set(changed)=={'src/ralup/search.py'}
check=base/'fix1-green-final/finish.json';assert read(check)['exit_code']==0
correction=ref(base/'authenticate-recipe-metadata-v2/recipe-correction.json')
audit=auth(correction);muon_defaults=next(r['actual_rule']['config'] for r in audit['rows'] if r['actual_rule']['schema']=='muon_project_fp32.v1')
candidates=[]
def add(name,config,horizon,origin):
    admission=base/name/'admission.json';d=read(admission);design=auth(d['design'])
    relative=Path(d['design']['path']).name;archive=base/name/'producer/docs/harness'/relative
    assert sha256_file(archive)==d['design']['sha256']
    candidates.append((name,copy.deepcopy(config),horizon,origin,ref(admission),d['design'],ref(archive),ref(base/name/'startup.json')))
name='native-admission-v1';d=read(base/name/'admission.json');design=auth(d['design']);assert d['status']=='pass'
for config in design['cases']:add(name,config,32,'original frozen32 case; paired32 / strict16')
name='native-horizon128';d=read(base/name/'admission.json')
for run in d['runs']:
    c=run['config'];coarse=[c['resolved']['profile'],c['recipe']['method'],c['recipe']['quantizer'],c['recipe']['rotation'],c['recipe']['backward'],c['packing']]
    if coarse in d['scopes']:add(name,c,128,'completed admitted scope within original capped wave; other failed/unattempted scopes excluded')
for name in ('native-extension256','native-warm-extension256'):
    d=read(base/name/'admission.json');assert d['status']=='pass'
    for run in d['runs']:add(name,run['entry']['config'],256,'strict authenticated128-to256 carry-state extension; original parent/cursor/ancestry unchanged')
name='native-meta-parent';d=read(base/name/'admission.json');assert d['status']=='pass'
add(name,auth(d['design'])['continuation_config'],256,'genuine proxy meta-train continuation; separate original32 prefix / paired256 / strict128')
for name in ('native-ordinary-geometry','native-quantizer-geometry'):
    d=read(base/name/'admission.json');assert d['status']=='pass'
    for run in d['runs']:add(name,run['config'],16,'actual core paired16 / strict8; no longer-horizon qualification')
for horizon in (16,32,128,256):
    prior=base/f'applicability-v5/admission-applicability-{horizon}.json';allowed=read(prior)['scopes'];scopes=[]
    for index,(name,original,bound,origin,admission,design,archive,startup) in enumerate(candidates):
        c=copy.deepcopy(original);r=c['recipe'];coarse=[c['resolved']['profile'],r['method'],r['quantizer'],r['rotation'],r['backward'],c['packing']]
        if bound!=horizon or coarse not in allowed:continue
        # Descriptive corrections reflect the original named factory/cache bytes;
        # original configs are retained and are never retrospectively rewritten.
        r['grouping']='none' if r['quantizer']=='fp32' else grouping_for(r['quantizer'])
        r['normalization']=r['method'].removeprefix('muon_plus_').removesuffix('_fp32') if r['method'].startswith('muon_plus_') else 'none'
        r['moment_coordinates']='original_W'
        variants=[('original-settings',copy.deepcopy(c))]
        if r['method']=='muon_project_fp32' and not r.get('method_config'):
            c['recipe']['method_config']=copy.deepcopy(muon_defaults);variants.append(('explicit-same-defaults',copy.deepcopy(c)))
        for variant,config in variants:
            contract=operation_contract(config)
            proof={'status':'pass','scope':contract,'source_identity':new,'maximum_steps':horizon,
                'created_utc':datetime.now(timezone.utc).isoformat(),'role':'retrospective exact scope authentication and current guard applicability, NOT a new prospective numerical freeze or replay',
                'original_admission':admission,'original_design':design,'original_design_producer_archive':archive,'original_startup':startup,
                'original_configuration':original,'origin':origin,'variant':variant,'annotation_correction':correction,
                'settings_basis':'original frozen settings; implicit defaults bound by unchanged registry/optimizer source; explicit ProjectMuon defaults independently authenticated in saved actual rule cache by correction record',
                'prior_source_applicability':ref(prior),'new_training_targets':0,'new_query_targets':0,
                'limitations':'original parents/rates/packing/horizons retain original scientific scope; no generic scalar-rate permission or cross-source full-state restore'}
            p=out/f'scope-{horizon}-{index}-{variant}.json';_write(p,proof)
            scopes.append({'contract':contract,'maximum_steps':horizon,'evidence':ref(p)})
    assert scopes
    report={'status':'pass','source_identity':new,'operation_scopes':scopes,'maximum_steps':horizon,
        'created_utc':datetime.now(timezone.utc).isoformat(),'role':'exact operation applicability, admin only','prior_coarse_scopes':allowed,
        'original_scopes':ref(prior),'changed_modules':changed,
        'exact_diffs':{k:''.join(difflib.unified_diff((base/'applicability-v5/producer'/k).read_text().splitlines(keepends=True),(root/k).read_text().splitlines(keepends=True),fromfile='old/'+k,tofile='new/'+k)) for k in changed},
        'checks':ref(check),'stdout':ref(base/'fix1-green-final/stdout.txt'),
        'delta':'exact actual operation admission; authenticated completed selection bindings, unique recipe/seed and trial IDs; supported schedule dispatch only. Numerical modules unchanged.',
        'scalar_lr':'new prospective LR-only stress may use a separately hash-bound ralup.scalar_lr_applicability.v1 record with exact admitted/requested contracts, same numerical evidence/current source and risk rationale; no refresh/model/schedule delta accepted',
        'strict_restore':'source identity remains strict; fresh state or authenticated model-only/reset parent required; historical full-state restoration not waived',
        'new_training_targets':0,'new_query_targets':0}
    for scope in scopes:
        config={'resolved':{'model':scope['contract']['model'],'runtime':scope['contract']['runtime'],'optimizer_probe':scope['contract']['optimizer_probe'],'data':scope['contract']['data_geometry']},
            'recipe':scope['contract']['recipe'],'packing':scope['contract']['packing'],'budget':horizon*8192,'source_identity':new}
        _validate_admission(config,report)
    p=out/f'admission-applicability-{horizon}.json';_write(p,report);print(json.dumps({'artifact':ref(p),'exact_scopes':len(scopes)}))
