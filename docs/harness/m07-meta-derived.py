"""Publish a separately typed meta parent and verify its fresh model-only import."""
import copy,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
import torch
from ralup.search import source_identity,create_trial_state
from ralup.records import ArtifactRef,sha256_file
from ralup.calibrate import _write
from ralup.state import load_snapshot,_validated,capture_snapshot
from ralup.train import _rng_state
from ralup.warmstart import create_derived_warmstart,import_derived_warmstart

root=Path.cwd();base=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m07-20261011')
designpath=root/'docs/harness/m07-meta-derived-design-v1.json'
ref=lambda p:{'path':str(p),'sha256':sha256_file(p)}
def auth(r):
    if sha256_file(r['path'])!=r['sha256']:raise ValueError('Meta parent input hash mismatch')
def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return set(a)==set(b) and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

if sys.argv[1]=='freeze':
    admission=json.loads((base/'native-meta-parent/admission.json').read_bytes());assert admission['status']=='pass'
    saved=next(r for r in admission['arms'][0]['saved'] if r['step']==256)
    config=copy.deepcopy(json.loads((root/'docs/harness/m07-meta-parent-design-v2.json').read_bytes())['continuation_config'])
    config.update(source_identity=source_identity(),source_revision='737233e7ff8d8d2246aaa02598a270ded9068b4d',budget=8192)
    config.pop('parent',None)
    design={'created_utc':datetime.now(timezone.utc).isoformat(),'source_identity':source_identity(),'helper':ref(Path(__file__)),
        'supervisor':ref(root/'docs/harness/m07-supervise.py'),'admission':ref(base/'native-meta-parent/admission.json'),
        'checkpoint':{k:saved[k] for k in ('path','sha256')},'fresh_config':config,
        'completed_attempts':[ref(p) for p in sorted(base.glob('*/finish.json'))],
        'selection_evidence':[ref(base/'query-meta-parent/query-results.json')],
        'purpose':'explicit separate meta-train derived development parent plus actual fresh-state model-only import; original continuation immutable',
        'partition':'meta_train','original_role':'continuation','new_training_steps':0,'new_query_targets':0,
        'finite_forecast_seconds':90,'forecast_bytes':64*1024**2,
        'forecast_basis':'earlier core typed publication/full-current-snapshot import proof37.6s; smaller proxy and no retained full snapshot;90s includes authentication/I/O',
        'ancestry':'one actual meta-train AdamW32 prefix followed by FP32Muon256;2359296 trajectory targets; paired proof arm costs distinct, never independent seeds',
        'cost_contract':'completed physical attempts once globally and inherited trajectory ledger are overlapping cost views; creation/import/I/O charged by this attempt',
        'limits':'provisional development parent supports measured protocol; not a final warm-duration selection, proxy-core transfer proof, B0 or WP04 pass'}
    _write(designpath,design);print(json.dumps(ref(designpath)))
elif sys.argv[1]=='run':
    d=json.loads(designpath.read_bytes());assert d['source_identity']==source_identity()
    for key in ('helper','supervisor','admission','checkpoint'):auth(d[key])
    out=Path(os.environ['M07_ATTEMPT_DIR']);start=time.perf_counter()
    parent=create_derived_warmstart(ArtifactRef(**d['checkpoint']),out/'derived-warmstart.json',
        completed_attempts=[ArtifactRef(**r) for r in d['completed_attempts']],selection_evidence=[ArtifactRef(**r) for r in d['selection_evidence']])
    payload=_validated(load_snapshot(ArtifactRef(**d['checkpoint'])))
    fresh=create_trial_state(d['fresh_config']);ids={n:id(p) for n,p in fresh.model.named_parameters()}
    state=import_derived_warmstart(parent,fresh,partition='meta_train',continuation_targets=8192)
    assert {n:id(p) for n,p in state.model.named_parameters()}==ids
    assert all(torch.equal(p.detach().cpu(),payload['weights'][n]) for n,p in state.model.named_parameters())
    assert state.cursor.state=={**payload['cursor'],'run_target_budget':2359296+8192}
    assert state.schedule['step']==0 and state.histories==state.quantizer_state=={}
    assert all(not value for value in state.optimizer_state.values())
    assert same(_rng_state(),payload['global_rng'])
    current=_validated(capture_snapshot(state));assert same(current['rng_streams'],payload['rng_streams'])
    assert state.ledger.stream==payload['ledger']['stream'] and state.ledger.passes==payload['ledger']['passes']
    if time.perf_counter()-start>d['finite_forecast_seconds']:raise RuntimeError('Frozen meta import envelope exhausted')
    report={'status':'pass','design':ref(designpath),'source_identity':source_identity(),'derived_parent':ref(parent.path),
        'original_checkpoint':d['checkpoint'],'original_role_unchanged':'continuation','source_partition':'meta_train',
        'new_training_steps':0,'new_query_targets':0,'complete_seconds':time.perf_counter()-start,
        'preserved_targets':2359296,'stream':state.ledger.stream,'passes':state.ledger.passes,'import_event':state.ledger.events[-1],
        'exact_checks':['all W/aux and Parameter identities','cursor/data/partition/global+named RNG','optimizer/history/quantizer/schedule resets','complete inherited counters'],
        'limits':d['limits']}
    _write(out/'import-results.json',report);print(json.dumps({'report':ref(out/'import-results.json'),'parent':ref(parent.path)}))
else:raise ValueError('Explicit freeze/run required')
