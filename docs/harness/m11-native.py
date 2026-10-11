"""Initial genuine M11 META-TRAIN reference/noise/signal diagnostics, finite scope."""
import copy,json,math,os,random,subprocess,sys,time
from dataclasses import asdict
from pathlib import Path
import torch
from ralup.config import canonical_bytes
from ralup.records import ArtifactRef,sha256_file
from ralup.search import source_identity,create_trial_state
from ralup.policy.checks import _write,_ref
from ralup.policy.controllers import Affine45,LearnedLR49,freeze_controller
from ralup.policy.adapter import PolicyAdapter
from ralup.policy.normalizer import FrozenNormalizer,fit_normalizer
from ralup.state import capture_snapshot,save_snapshot,restore_snapshot,exact_cpu_state_hash
from ralup.train import train_step,select_logical_batch,_rng_state,_restore_rng
from ralup.optim.base import SchedulePoint
from ralup.data.cursor import BatchCursor
from ralup.data.batches import next_logical_batch
from ralup.meta.tasks import MetaTask,rollout
from ralup.meta.es import OuterAdam,train_generation
from ralup.calibrate import _Sampler,_limits,memory_report,_offsets

base=Path(os.environ['M11_ATTEMPT_DIR']);design_path=Path(sys.argv[1]);design=json.loads(design_path.read_bytes())
assert design['source_identity']==source_identity()
assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0' and torch.backends.mps.is_available()
assert subprocess.check_output(['sw_vers','-productVersion'],text=True).strip()==design['runtime']['macOS']
assert subprocess.check_output(['sw_vers','-buildVersion'],text=True).strip()==design['runtime']['build']
for entry in design['helpers'].values():assert sha256_file(entry['path'])==entry['sha256']
started=time.perf_counter();report={'design':_ref(design_path),'status':'running','references':[],'observations':[],'confirmation_eligible':False}
producer=_ref(base/'startup.json');cfg=design['config'];snapshots={};trajectories=[]
nref=ArtifactRef(**design['bootstrap_normalizer']);data=json.loads(nref.path.read_bytes());assert sha256_file(nref.path)==nref.sha256
normalizer=FrozenNormalizer(torch.tensor(data['mean']),torch.tensor(data['std']),data['source_manifest'])

def bind(s,c,n,r):
    old=s.method
    adapter=PolicyAdapter(old.rule,c,n,normalizer_ref=r).bind(dict(s.model.named_parameters()),old.groups)
    adapter.optimizer_states=old.optimizer_states
    if type(old) is PolicyAdapter:adapter.last_actions=copy.deepcopy(old.last_actions)
    s.method=adapter;s.optimizer_state=adapter.optimizer_states
    s.configuration['policy_features']={'enabled':True,'full_horizon':design['continuation_horizon'],'trace_cadence':1}
    s.histories['feature_normalizer']=n


def save(name,s):
    t=time.perf_counter();snapshot=capture_snapshot(s);ref=save_snapshot(snapshot,base/name)
    return snapshot,{'artifact':_ref(ref.path),'seconds':time.perf_counter()-t,'state_hash':exact_cpu_state_hash(s)}

try:
 with _Sampler('mps',_limits(cfg['resolved'])) as sampler:
    def cap():
        sampler.sample()
        if sampler.exceeded or time.perf_counter()-started>design['forecast_seconds']:raise RuntimeError('Frozen M11 native time/memory envelope exhausted')
    for origin in design['reference_origins']:
        cap();s=create_trial_state(cfg);c=Affine45()
        with torch.no_grad():c.linear.bias.copy_(torch.tensor(origin['bias']))
        bind(s,c,normalizer,nref);row={'origin':origin,'steps':[],'snapshots':[]};report['references'].append(row);features=[]
        with (base/(origin['name']+'-steps.jsonl')).open('x') as f:
            for step in range(design['reference_steps']):
                cap();s.schedule['point']=SchedulePoint(.02,0.,step/design['continuation_horizon'])
                batch=select_logical_batch(s,8192);before=len(s.ledger.events)
                try:r=train_step(s,batch)
                finally:
                    f.write(canonical_bytes({'step':step,'inputs':_offsets(batch),'events':s.ledger.events[before:]}).decode()+'\n');f.flush();os.fsync(f.fileno())
                row['steps'].append({'step':step,'work':r.work})
                for name in s.model._contexts:features.append({'step':step,'matrix':name,'values':s.histories[name].last_features.cpu().tolist()})
                if step+1 in design['snapshot_steps']:
                    snap,receipt=save(origin['name']+'-'+str(step+1),s);snapshots[(origin['name'],step+1)]=snap;row['snapshots'].append(receipt)
        trajectory={'schema':'ralup.feature_trajectory.v1','partition':'meta_train','role':'reference','synthetic':False,'source_identity':source_identity(),
          'data_manifest':cfg['resolved']['data']['manifest'],'producer':producer,'backbone':design['backbone'],'parent':cfg['parent'],'continuation_horizon':design['continuation_horizon'],'rows':features}
        p=base/(origin['name']+'-reference.json');_write(p,trajectory);trajectories.append(ArtifactRef(**_ref(p)))
        _write(base/(origin['name']+'-complete.json'),row)
    fitted=fit_normalizer(trajectories);p=base/'reference-normalizer.json'
    _write(p,{'mean':fitted.mean.tolist(),'std':fitted.std.tolist(),'source_manifest':fitted.source_manifest,'scope':'initial M11 reference distribution, final training coverage not yet established'})
    fitted_ref=ArtifactRef(**_ref(p));report['reference_normalizer']=_ref(p)
    probe=design['probe'];s=restore_snapshot(snapshots[(probe['snapshot_origin'],probe['snapshot_step'])],'mps')
    bind(s,Affine45(),fitted,fitted_ref)
    snapshot,receipt=save('probe-complete-state',s);report['probe_snapshot']=receipt
    cursor=BatchCursor(s.cursor.state);batches=[next_logical_batch(cursor,8192) for _ in range(probe['horizon'])]
    inner={b.article_hash for batch in batches for b in batch.block_slices};manifest=cfg['resolved']['data']['manifest']
    qcursor=BatchCursor.from_manifest(Path(manifest['path']),manifest['sha256'],partition='meta-train',seed=probe['query_seed'])
    query=[];count=0
    while count<probe['query_targets']:
        b=qcursor.take(min(128,probe['query_targets']-count))
        if b.article_hash not in inner:query.append(b);count+=sum(b.score_mask)
    def plain_block(b):return {k:list(v) if isinstance(v,tuple) else v for k,v in vars(b).items()}
    _write(base/'probe-inputs.json',{'inner':[[plain_block(b) for b in batch.block_slices] for batch in batches],'query':[plain_block(b) for b in query],
      'inner_document_hashes':sorted(inner),'query_document_hashes':sorted({b.article_hash for b in query}),'query_rng':probe['query_seed'],'seed':cfg['seed'],'partition':'meta_train'})
    docs={d['content_sha256']:'meta_train' for d in s.cursor._documents}
    task=MetaTask(snapshot,batches,query,{'query_seed':probe['query_seed']},{'partition':'meta_train','synthetic':False,'device':'mps','document_partitions':docs})
    controller=freeze_controller(Affine45());dups=[]
    for i in range(probe['duplicates']):
        cap();task.provenance['journal']=str(base/f'duplicate-{i}-journal.jsonl')
        result=rollout(task,controller,probe['horizon']);dups.append(result)
        value={'kind':'duplicate','index':i,'model':cfg['resolved']['model'],'seed':cfg['seed'],'recipe':cfg['recipe'],'result':asdict(result)}
        report['observations'].append(value);_write(base/f'duplicate-{i}.json',value)
        print(json.dumps({'observation':str(base/f'duplicate-{i}.json'),'objective':result.objective,'initial_query_nats':result.initial_query,'final_query_nats':result.final_query,'elapsed_seconds':result.work['elapsed_seconds']}),flush=True)
    assert dups[0].objective==dups[1].objective and dups[0].work['final_state_hash']==dups[1].work['final_state_hash'],'Native full-state duplicate contract violated'
    report['duplicate_noise']=0.
    # Explicit neural LR49 output endpoints, original hidden network retained.
    for h in design['lr49_admission']['h_values']:
        cap();torch.manual_seed(design['lr49_admission']['hidden_initialization_seed']);lr=LearnedLR49()
        with torch.no_grad():lr.output.bias.fill_(math.log(h))
        task.provenance['journal']=str(base/f'lr49-{h}-journal.jsonl')
        result=rollout(task,freeze_controller(lr),design['lr49_admission']['steps'])
        value={'kind':'lr49-neural-composition','h':h,'result':asdict(result)};report['observations'].append(value);_write(base/f'lr49-{h}.json',value)
    cap();task.provenance['journal']=str(base/'antithetic-journal.jsonl')
    eps=torch.randn(probe['pairs'],45,generator=torch.Generator().manual_seed(probe['seed']))
    generation=train_generation(task,'affine45',OuterAdam(controller.parameters,lr=probe['outer_lr']),eps,probe['horizon'],probe['sigma'])
    _write(base/'generation-1.json',generation);report['generation']=_ref(base/'generation-1.json')
    report['status']='measured_bootstrap_only';report['memory']=memory_report(sampler.samples)
except BaseException as error:
 report['status']='failed';report['failure']=str(error);report['unknown_partial_work']=True
 raise
finally:
 report['elapsed_seconds']=time.perf_counter()-started
 report['next_required']='Query/m/sigma and K32/K64/full-continuation calibration, real final baseline/policy early/middle/late pool, finalMETA-onlyfit, equivalent controller learning and two-stage selection remain.'
 _write(base/'report.json',report)
