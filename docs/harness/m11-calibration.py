"""Versioned M11 measured signal and full-continuation calibration, no selection."""
import copy,json,math,os,subprocess,sys,time
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import torch
from ralup.config import canonical_bytes
from ralup.records import ArtifactRef,sha256_file
from ralup.search import source_identity
from ralup.state import load_snapshot,restore_snapshot
from ralup.data.cursor import BatchCursor
from ralup.data.batches import next_logical_batch
from ralup.data.blocks import TargetBlock
from ralup.policy.controllers import ControllerState,LearnedLR49,freeze_controller
from ralup.policy.checks import _write,_ref
from ralup.meta.tasks import MetaTask,rollout,_input_digest
from ralup.meta.es import estimate_es
from ralup.calibrate import _Sampler,_limits,memory_report

base=Path(os.environ['M11_ATTEMPT_DIR']);mode=sys.argv[1]
bootstrap=Path('/Users/artemlegotin/ralup-evidence/adaptive-calculus/2026-10-03/m11-20261011/native-bootstrap-v2')

def plain(b):return {k:list(v) if isinstance(v,tuple) else v for k,v in vars(b).items()}

def inputs(snapshot):
    state=restore_snapshot(snapshot,'mps');cursor=BatchCursor(state.cursor.state)
    batches=[next_logical_batch(cursor,8192) for _ in range(512-state.schedule['step'])]
    return state,batches

if mode=='freeze':
    report=json.loads((bootstrap/'report.json').read_bytes());generation=json.loads((bootstrap/'generation-1.json').read_bytes())
    assert report['status']=='measured_bootstrap_only'
    snapshot_ref=report['probe_snapshot']['artifact'];snapshot=load_snapshot(ArtifactRef(**snapshot_ref));state,batches=inputs(snapshot)
    prior=json.loads(Path(report['design']['path']).read_bytes());assert prior['source_identity']==source_identity()
    inner={b.article_hash for batch in batches for b in batch.block_slices};manifest=prior['config']['resolved']['data']['manifest']
    # Predeclared deterministic query rule: reserve a larger nested list, exclude
    # EVERY full-continuation inner document before scoring any new outcome.
    query_seed=2112;cursor=BatchCursor.from_manifest(Path(manifest['path']),manifest['sha256'],partition='meta-train',seed=query_seed)
    query=[];count=0
    while count<32768:
        b=cursor.take(min(128,32768-count))
        if b.article_hash not in inner:query.append(b);count+=sum(b.score_mask)
    # Freeze exactly8192 scored bytes as a nested prefix, including partial mask.
    small=[];count=0
    from dataclasses import replace
    for b in query:
        if count==8192:break
        left=8192-count;selected=0;mask=[]
        for bit in b.score_mask:
            take=bool(bit and selected<left);mask.append(take);selected+=take
        small.append(replace(b,score_mask=tuple(mask)));count+=selected
    qsmall=base/'query-8192.json';qlarge=base/'query-32768.json'
    for p,q in [(qsmall,small),(qlarge,query)]:_write(p,{'partition':'meta_train','blocks':[plain(b) for b in q],'document_hashes':sorted({b.article_hash for b in q}),'selection_seed':query_seed,'inner_exclusion_sha256':__import__('hashlib').sha256(canonical_bytes({'documents':sorted(inner)})).hexdigest()})
    phi=torch.tensor(generation['phi_before']);updated=generation['phi_after'];epsilon=torch.tensor(generation['perturbations'][0])
    candidates={'initialized':phi.tolist(),'outer_step1':updated,'perturbed_direction1':(phi+.05*epsilon).tolist()}
    runs=[]
    for label,parameters in candidates.items():
        for horizon in [16,32,64,508]:runs.append({'id':f'rank-{label}-{horizon}','family':'affine45','parameters':parameters,'horizon':horizon,'query':_ref(qsmall),'kind':'horizon_rank','candidate':label})
    runs.append({**runs[3],'id':'duplicate-full-508','kind':'full_duplicate'})
    generator=torch.Generator().manual_seed(2111);eps=torch.randn(8,45,generator=generator)
    for sigma in [.025,.05,.1]:
        for i in range(8 if sigma==.05 else 4):
            for sign in [1,-1]:runs.append({'id':f'signal-{sigma}-{i}-{sign}','family':'affine45','parameters':(phi+sign*sigma*eps[i]).tolist(),'horizon':16,'query':_ref(qsmall),'kind':'signal','sigma':sigma,'pair':i,'sign':sign})
    for i in range(4):
        for sign in [1,-1]:runs.append({'id':f'query-large-{i}-{sign}','family':'affine45','parameters':(phi+sign*.05*eps[i]).tolist(),'horizon':16,'query':_ref(qlarge),'kind':'query_size','sigma':.05,'pair':i,'sign':sign})
    torch.manual_seed(2113);lr=LearnedLR49()
    with torch.no_grad():lr.output.bias.fill_(math.log(32.))
    for i in range(2):runs.append({'id':f'lr49-max32-h16-{i}','family':'learned_lr49','parameters':freeze_controller(lr).parameters.tolist(),'horizon':16,'query':_ref(qsmall),'kind':'lr49_new_horizon_admission'})
    task=MetaTask(snapshot,batches,small,{'query_seed':query_seed},{'partition':'meta_train','document_partitions':{d['content_sha256']:'meta_train' for d in state.cursor._documents},'device':'mps','synthetic':False})
    d={'schema':'ralup.m11_calibration_design.v1','created_utc':datetime.now(timezone.utc).isoformat(),'purpose':'exploratory-development-bootstrap',
      'source_identity':source_identity(),'bootstrap_report':_ref(bootstrap/'report.json'),'bootstrap_generation':_ref(bootstrap/'generation-1.json'),'snapshot':snapshot_ref,
      'snapshot_initial_state_hash':report['probe_snapshot']['state_hash'],'task_full_input_sha256':_input_digest(task,508),'query_seed':query_seed,'perturbation_seed':2111,'perturbations':eps.tolist(),
      'model_seed':707,'model':'proxy2x128/4heads','recipe':prior['config']['recipe'],'packing':64,'normalizer':report['reference_normalizer'],
      'config':prior['config'],'runtime':prior['runtime'],'runs':runs,'forecast_seconds':4200,'hard_deadline_seconds':4260,'forecast_bytes':2*1024**3,
      'maximum_optimizer_steps':sum(r['horizon'] for r in runs),'maximum_training_targets':8192*sum(r['horizon'] for r in runs),'maximum_query_targets':sum(2*(32768 if r['kind']=='query_size' else 8192) for r in runs),
      'forecast_basis':'v2 actual mean step.587117612s/reset4.579331531s/query.517097896s/snapshot2.554150114s;3040steps plus55complete resets~2205s, conservative4200s including full-input materialization/source-validation/IO; forecasts are finite engineering bounds, not universal unaffordability',
      'helpers':{str(p):_ref(p) for p in map(Path,['docs/harness/m11-calibration.py','docs/harness/m11-attempt.py','docs/harness/m11-supervise.py'])},
      'new_scope':'Same qualified proxy geometry/recipe/current source; actual K32/K64 and full remaining508-to512 plus LR49 max32 horizon16 duplicate. Exact numerical kernels/old scope unchanged. Expanded horizon accepts only exact duplicate task objective/scientific-state hash; failed finite controller outcome retained, not silently defeated family.',
      'acceptance':{'duplicate_objective_atol':0.,'duplicate_final_state':'exact H07','finite':True},
      'retention':'Existing owned complete current-source task snapshot reused exactly; all new successful final weights disposable and reconstructible from snapshot/controller/query/input digest/RNG. Retain all raw results/journals/failures and source/runtime receipts; no old artifact deletion.',
      'scope':'Calibration only. Initial768-row reference normalizer remains diagnostic. No final training budgets/final pool/selected controller/confirmation/B0 acceptance.'}
    _write(base/'design.json',d);print(json.dumps(_ref(base/'design.json')),flush=True)
else:
    design_path=Path(sys.argv[2]);d=json.loads(design_path.read_bytes());assert d['source_identity']==source_identity()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0' and torch.backends.mps.is_available()
    assert subprocess.check_output(['sw_vers','-productVersion'],text=True).strip()==d['runtime']['macOS']
    assert subprocess.check_output(['sw_vers','-buildVersion'],text=True).strip()==d['runtime']['build']
    for ref in d['helpers'].values():assert sha256_file(ref['path'])==ref['sha256']
    started=time.perf_counter();report={'design':_ref(design_path),'status':'running','runs':[],'confirmation_eligible':False}
    try:
      snapshot=load_snapshot(ArtifactRef(**d['snapshot']));state,batches=inputs(snapshot)
      docs={v['content_sha256']:'meta_train' for v in state.cursor._documents}
      with _Sampler('mps',_limits(d['config']['resolved'])) as sampler:
        for run in d['runs']:
            sampler.sample()
            if sampler.exceeded or time.perf_counter()-started>d['forecast_seconds']:raise RuntimeError('Frozen M11 calibration time/memory cap exhausted')
            ref=run['query'];assert sha256_file(ref['path'])==ref['sha256'];q=json.loads(Path(ref['path']).read_bytes())
            blocks=[TargetBlock(**{k:tuple(v) if isinstance(v,list) else v for k,v in b.items()}) for b in q['blocks']]
            task=MetaTask(snapshot,batches,blocks,{'query_seed':d['query_seed']},{'partition':'meta_train','document_partitions':docs,'device':'mps','synthetic':False,'journal':str(base/(run['id']+'-journal.jsonl'))})
            if run['kind']!='query_size':assert _input_digest(task,508)==d['task_full_input_sha256']
            result=rollout(task,ControllerState(run['family'],torch.tensor(run['parameters'])),run['horizon'])
            value={'run':run,'result':asdict(result)};p=base/(run['id']+'.json');_write(p,value);report['runs'].append(_ref(p))
            print(json.dumps({'run':run['id'],'objective':result.objective,'initial':result.initial_query,'final':result.final_query,'elapsed_seconds':result.work['elapsed_seconds']}),flush=True)
            if run['kind']=='full_duplicate':
                first=json.loads((base/'rank-initialized-508.json').read_bytes())['result']
                assert result.objective==first['objective'] and result.work['final_state_hash']==first['work']['final_state_hash'],'New full horizon duplicate contract violated'
            if run['id']=='lr49-max32-h16-1':
                first=json.loads((base/'lr49-max32-h16-0.json').read_bytes())['result']
                assert result.objective==first['objective'] and result.work['final_state_hash']==first['work']['final_state_hash'],'New LR49 horizon duplicate contract violated'
        report['memory']=memory_report(sampler.samples);report['status']='calibration_measured_pending_interpretation'
    except BaseException as error:
        report['status']='failed';report['failure']=str(error);report['unknown_partial_work']=True;raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started;_write(base/'report.json',report)
