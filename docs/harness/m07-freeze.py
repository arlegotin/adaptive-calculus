"""Freeze explicit finite admission design; no experiment outputs consumed."""
import json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
from ralup.config import canonical_bytes,resolve_config
from ralup.records import sha256_file
from ralup.search import source_identity
root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
def ref(p):return {'path':str(p),'sha256':sha256_file(p)}
old=json.loads((root/'configs/h08-real-proxy.json').read_bytes())
config=old['config'];gates=old['evidence']['gates'];gatebase=base/'current-runtime-gate'
config['runtime']['environment']=ref(gatebase/'environment.json')
gates['WP00']=json.loads((gatebase/'wp00-local-gate-v3.json').read_bytes())
config['calibration']['output']=str(base/'measured-development')
config['calibration'].update(seed=707,repeats=2,pilot_target_budget=32*8192,warmstart_target_budget=32*8192,continuation_target_budget=32*8192,doublings=3)
revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
cases=[]
for profile,method,quantizer,packing in [('proxy','adamw','fp32',64),('proxy','muon_project_fp32','row_dynamic_max',64),('core','adamw','fp32',32),('core','muon_project_fp32','row_dynamic_max',32)]:
    payload=json.loads(canonical_bytes(config));payload['profile']=profile
    if profile=='core':payload['model']=json.loads((root/'configs/core.json').read_bytes())['model']
    resolved=resolve_config(payload,{'gates':gates},purpose='calibration')
    cases.append({'purpose':'development-bootstrap','role':'warmstart','family':'AdamW' if method=='adamw' else 'Muon',
        'seed':707,'budget':32*8192,'partition':'target_train','packing':packing,'resolved':json.loads(canonical_bytes(resolved.payload)),
        'source_revision':revision,'source_identity':source_identity(),'finite_forecast_seconds':2700,
        'recipe':{'method':method,'quantizer':quantizer,'grouping':'none' if quantizer=='fp32' else 'row','rotation':'identity','backward':'ste',
            'schedule':'constant','learning_rate':.001 if method=='adamw' else .02,'decay':0.,'normalization':'original_W','clip_norm':1.,'method_config':{}},
        'query_seed':1707,'query_bytes':8192,'milestones':[8,16,32]})
design={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'research_cutoff':'2026-10-03',
    'source_identity':source_identity(),'contract':'exact same-device scientific state every completed step; independent duplicate with strict midpoint restore; no relaxed tolerance',
    'purpose':'finite current-runtime actual-shape admission and throughput; no final duration or B0',
    'steps':32,'cases':cases,'finite_forecast_seconds':2700,'forecast_basis':'historical proxy approximately1s/step; conservative core and archive allowance8s/step across256 measured steps plus650s setup',
    'forecast_bytes':3*1024**3,'model_floor_bytes':{'proxy':493056*16,'core':3347456*16},
    'snapshot_pool_forecast':'two complete endpoints + one midpoint per case; core original-W/moments plus contexts histories/gradients;3GiB includes raw input records and fixtures',
    'retention':'all failure artifacts, endpoints and midpoint; no every-step full snapshots',
    'admission_scope':'these four risk representatives only; other methods/rotations/trust/grouping await their actual admission; no universal Cartesian claim',
    'doctor':ref(gatebase/'wp00-local-gate-v3.json'),'doctor_source_delta':'after doctor: only partition exception subtype and FP32 reference-only B0 guard; environment/operations/model math unchanged; exact producer archive retained',
    'provenance':{'muon':'clean independently implemented published equations; MIT reference; project non-EMA momentum and FP32 departure explicit',
        'muon_plus':'clean local published-equation route eligible for prospective numerical gates; no copied restricted source or author grant/release claim',
        'madam':'clean local mathematical comparator; restricted author software not copied; no software rights/release/legal assurance',
        'soap':'pinned MIT mechanics with explicit timed CPU eig/QR; operations/parity/tuning remain required',
        'quest':'project row/tensor Gaussian refit; upstream grouping/release parity unresolved'},
    'next':'measure learning at8/16/32; extend budget by doubling on unresolved learning/ranks using measured cost; source-default LR log bracket then refine; preserve all unsettled/edge/overlap and headroom-eliminating arms; no policy gain'}
p=root/'docs/harness/m07-design-v1.json'
with p.open('xb') as f:f.write(canonical_bytes(design)+b'\n')
print(ref(p))
