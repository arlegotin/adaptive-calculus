"""Bounded typed-parent publication and native model-only import proof, zero model passes."""
import copy,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
import torch
from ralup.search import source_identity,create_trial_state
from ralup.records import ArtifactRef,sha256_file
from ralup.calibrate import _write
from ralup.state import load_snapshot,_validated,capture_snapshot,save_snapshot,restore_snapshot,exact_cpu_state_hash
from ralup.train import _rng_state
from ralup.warmstart import create_derived_warmstart,import_derived_warmstart

root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
designpath=root/'docs/harness/m07-derived-parent-design-v1.json'
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
def auth(r):
    assert sha256_file(r['path'])==r['sha256']
def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return set(a)==set(b) and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

if sys.argv[1]=='freeze':
    original=base/'authenticate-strong-parent/authenticated-original.json';a=json.loads(original.read_bytes())
    assert a['unchanged_original_role']=='continuation' and a['stream']['targets']==2359296
    extension=json.loads((base/'native-extension256/admission.json').read_bytes())
    config=next(row['entry']['config'] for row in extension['runs'] if row['entry']['label']=='core-muon-fp32128')
    config=copy.deepcopy(config);config.pop('parent',None);config['budget']=8192
    receipts=[ref(p) for p in sorted(base.glob('*/finish.json'))]
    selection=[ref(base/'query-strong-conversion/conversion-results.json')]
    design={'created_utc':datetime.now(timezone.utc).isoformat(),'source_identity':source_identity(),
        'helper':ref(Path(__file__)),'original_authentication':ref(original),'checkpoint':a['checkpoint'],
        'original_role':'continuation','partition':'target_train','mixed_ancestry':a['mixed_ancestry'],
        'completed_attempts':receipts,'selection_evidence':selection,'fresh_config':config,
        'purpose':'explicit derived model-only parent support and actual native import proof; no parent role rewrite, no science outcome selection',
        'finite_forecast_seconds':120,'forecast_bytes':128*1024**2,'new_training_steps':0,'new_query_targets':0,
        'native_checks':['exact all W/aux Parameters and retained object identities','same source/cursor position/global and named RNG',
            'explicit optimizer/history/quantizer/schedule reset','inherited trajectory counters and new import cost',
            'strict CURRENT-source capture/restore exact scientific owner hash'],
        'cost_contract':'all listed completed attempts once globally, inherited trajectory as separate overlapping view; creation/import/I/O charged in this new completed attempt; historical common engineering unknown',
        'model_contract':'three exact model-source files plus exact model config and named tensor dtype/shape; whole-runtime source remains distinct',
        'future_scope':'zero training proves import only; actual partition-separated meta parent and fresh target protocol still required'}
    _write(designpath,design);print(json.dumps(ref(designpath)))
elif sys.argv[1]=='run':
    d=json.loads(designpath.read_bytes());assert d['source_identity']==source_identity();auth(d['helper']);auth(d['original_authentication'])
    start=time.perf_counter();out=Path(os.environ['M07_ATTEMPT_DIR'])
    def cap():
        if time.perf_counter()-start>d['finite_forecast_seconds']:raise RuntimeError('Frozen import envelope exhausted')
    parent=create_derived_warmstart(ArtifactRef(**d['checkpoint']),out/'derived-warmstart.json',
        completed_attempts=[ArtifactRef(**r) for r in d['completed_attempts']],selection_evidence=[ArtifactRef(**r) for r in d['selection_evidence']]);cap()
    payload=_validated(load_snapshot(ArtifactRef(**d['checkpoint'])))
    fresh=create_trial_state(d['fresh_config']);ids={n:id(p) for n,p in fresh.model.named_parameters()}
    state=import_derived_warmstart(parent,fresh,partition=d['partition'],continuation_targets=8192);cap()
    assert {n:id(p) for n,p in state.model.named_parameters()}==ids
    assert all(torch.equal(p.detach().cpu(),payload['weights'][n]) for n,p in state.model.named_parameters())
    assert state.cursor.state=={**payload['cursor'],'run_target_budget':2359296+8192}
    assert state.schedule['step']==0 and state.histories==state.quantizer_state=={}
    assert all(not value for value in state.optimizer_state.values())
    assert same(_rng_state(),payload['global_rng'])
    current=capture_snapshot(state);current_payload=_validated(current)
    assert same(current_payload['rng_streams'],payload['rng_streams'])
    assert state.ledger.stream==payload['ledger']['stream'] and state.ledger.passes==payload['ledger']['passes']
    digest=exact_cpu_state_hash(state);saved=save_snapshot(current,out/'current-imported-state');cap()
    restored=restore_snapshot(load_snapshot(saved),'mps');cap()
    assert exact_cpu_state_hash(restored)==digest
    report={'status':'pass','design':ref(designpath),'source_identity':source_identity(),'derived_parent':ref(parent.path),
        'current_state':ref(saved.path),'current_state_sha256':digest,'original_checkpoint':d['checkpoint'],
        'original_role_unchanged':'continuation','preserved_targets':2359296,'new_training_steps':0,'new_query_targets':0,
        'complete_seconds':time.perf_counter()-start,'stream':state.ledger.stream,'passes':state.ledger.passes,
        'import_event':state.ledger.events[-1],'exact_checks':d['native_checks'],'limits':d['future_scope']}
    _write(out/'import-results.json',report);print(json.dumps({'report':ref(out/'import-results.json'),'parent':ref(parent.path)}))
else:raise ValueError('Explicit freeze/run required')
