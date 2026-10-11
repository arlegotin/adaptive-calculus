"""Freeze the next evidence-driven Muon quantized LR bracket, before outcomes."""
import copy,json,sys
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
from ralup.search import register_trial,source_identity
from ralup.records import sha256_file
from ralup.calibrate import _write

root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
stage=sys.argv[1];multipliers=[float(x) for x in sys.argv[2:]]
assert multipliers and all(0<x<=10000 for x in multipliers)
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
reference=base/'science/core-muon-dynamic-cont32/result.json'
default=json.loads(reference.read_bytes())
parent=base/'science/core-target-warm32/warmstart.json'
admission=base/'applicability-v1/admission-applicability.json'
cases=json.loads((root/'docs/harness/m07-design-v1.json').read_bytes())['cases']
trials=[]
for multiplier in multipliers:
    config=copy.deepcopy(cases[3]);label=f'muon-bracket-{stage}-m{multiplier:g}-32'
    config.update(source_identity=source_identity(),admission=ref(admission),parent=ref(parent),
        role='continuation',output=str(base/'science'/label),finite_forecast_seconds=180)
    config['recipe']['learning_rate']=.02*multiplier
    config['recipe']['method_config']={'momentum':.95,'nesterov':True,'auxiliary_lr':.001*multiplier,
        'auxiliary_betas':[.9,.99],'auxiliary_eps':1e-8}
    config['bracket']={'source_default':ref(reference),'joint_multiplier':multiplier,
        'scope':'both effective matrix/auxiliary rates; other settings fixed; refinement may separate them',
        'under_update_rule':'completed32 improvement less than one quarter of retained source-default32 improvement',
        'instability_rule':'failed nonfinite transaction or any declared development checkpoint exceeds initial BPB by0.5',
        'advancement':'missing bracket endpoint, edge optimum or unresolved curve extends; no fixed-count winner'}
    trials.append(asdict(register_trial(config)))
payload={'suite':'headroom','purpose':'development-bootstrap','created_utc':datetime.now(timezone.utc).isoformat(),
    'trials':trials,'reference':ref(reference),'forecast_seconds':180*len(trials),
    'decision':'LR bracketing only; budgets, fresh-seed finalists and B0 remain unresolved'}
p=root/f'configs/m07-bracket-{stage}.json';r=_write(p,payload);print(json.dumps(ref(r.path)))
