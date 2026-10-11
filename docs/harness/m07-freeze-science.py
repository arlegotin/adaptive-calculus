"""Freeze the next measured-development stage before outcomes (explicit inputs)."""
import json,sys,copy,hashlib
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
from ralup.search import register_trial,source_identity
from ralup.config import canonical_bytes
from ralup.records import sha256_file
from ralup.calibrate import _write
root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
stage=sys.argv[1];admission_path=Path(sys.argv[2]);admission={'path':str(admission_path),'sha256':sha256_file(admission_path)}
design=json.loads((root/'docs/harness/m07-design-v1.json').read_bytes());trials=[]
def add(case,name,role,partition='target_train',parent=None):
    config=copy.deepcopy(case);config.update(source_identity=source_identity(),admission=admission,role=role,
        partition=partition,output=str(base/'science'/name),finite_forecast_seconds=600,parent=parent)
    config['forecast_basis']='measured native proxy train.20-.34s/step and core measured timings;600s includes endpoint evaluation and authenticated source/checkpoint overhead'
    trials.append(asdict(register_trial(config)))
def parent(name):
    p=base/'science'/name/'warmstart.json';return {'path':str(p),'sha256':sha256_file(p)}
if stage=='warm':
    add(design['cases'][0],'proxy-target-warm32','warmstart')
    add(design['cases'][0],'proxy-meta-warm32','warmstart','meta_train')
    add(design['cases'][2],'core-target-warm32','warmstart')
    path=root/'configs/proxy.headroom.json'
    extra={'template_sha256':sha256_file(root/'configs/proxy.json')}
elif stage=='continue':
    add(design['cases'][0],'proxy-fp32-cont32','continuation',parent=parent('proxy-target-warm32'))
    add(design['cases'][1],'proxy-muon-dynamic-cont32','continuation',parent=parent('proxy-target-warm32'))
    add(design['cases'][2],'core-fp32-cont32','continuation',parent=parent('core-target-warm32'))
    add(design['cases'][3],'core-muon-dynamic-cont32','continuation',parent=parent('core-target-warm32'))
    path=root/'configs/m07-continuations.json';extra={}
else:raise ValueError('Unknown stage')
payload={'suite':'headroom','purpose':'development-bootstrap','trials':trials,
    'created_utc':datetime.now(timezone.utc).isoformat(),'decision':'initial measured length32 remains unresolved; final science fields remain null',**extra}
_write(path,payload);print(json.dumps({'path':str(path),'sha256':sha256_file(path),'trials':trials}))
