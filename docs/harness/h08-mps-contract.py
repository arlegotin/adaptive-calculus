"""Prospectively frozen real-shape full-step duplicate and strict-H07 replay wave."""
import dataclasses
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

import torch
from safetensors.torch import save, load

from ralup.__main__ import _calibration_config
from ralup.calibrate import _create, _event, _limits, _offsets, _Sampler, _timed, _write as _publish, import_warmstart, memory_report
from ralup.config import canonical_bytes
from ralup.records import ArtifactRef, RunLedger, sha256_file
from ralup.state import _automatic_identities, _validated, capture_snapshot, load_snapshot, restore_snapshot, save_snapshot
from ralup.train import select_logical_batch, train_step


OUTPUT=Path('artifacts/calibration/h08-mps-contract-v4')
CONFIG='configs/h08-real-proxy.json'
PARENTS='artifacts/checks/h08-real-warmstarts-cli.txt'
SELECTED=(1,4,8)


def budget(extra=0):
    retained=sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())
    assert retained+extra<=2*2**30, '2GiB prospective retention ceiling'
    assert shutil.disk_usage(OUTPUT).free>=2**30+extra, '1GiB available-space floor'
    return retained


def _write(path,payload):
    budget(len(canonical_bytes(payload)))
    return _publish(path,payload)


def discard(arms,publication):
    assert sha256_file(publication.path)==publication.sha256
    listed=[]
    for arm in arms:
        for value in arm['states'].values():
            path=auth(value)
            assert path.is_relative_to(OUTPUT) and path.name.startswith('state-') and path.name.endswith('.safetensors.gz')
            listed.append({**value,'disposition':'declared disposable scientific comparison copy; completed rollout, authenticated summaries/witnesses retained'})
    record=_write(publication.path.parent/(publication.path.stem+'-discard.json'),{'publication':{'path':str(publication.path),'sha256':publication.sha256},'explicit_disposable_files':listed,'retention_before_bytes':budget()})
    assert sha256_file(record.path)==record.sha256
    for value in listed:Path(value['path']).unlink()



def ref(path):
    return {'path':str(path),'sha256':sha256_file(path)}


def auth(value):
    assert ref(Path(value['path']))=={k:value[k] for k in ('path','sha256')}
    return Path(value['path'])


def science(snapshot):
    payload=_validated(snapshot)
    values={k:payload[k] for k in ('weights','gradients','model_training','module_training','model_cache','method_schema','groups','rule','method','quantizer_state','histories','schedule','cursor','rng_streams','global_rng','microbatch','clip_norm')}
    tensors, scalars={},{}
    def visit(value,path):
        if isinstance(value,torch.Tensor):
            tensors[path]=value.detach().cpu().contiguous().clone()
        elif dataclasses.is_dataclass(value):
            scalars[path+'.__type__']=type(value).__name__
            for field in dataclasses.fields(value):visit(getattr(value,field.name),path+'.'+field.name)
        elif isinstance(value,dict):
            scalars[path+'.__keys__']=sorted(map(str,value))
            for key in sorted(value,key=str):visit(value[key],path+'.'+str(key))
        elif isinstance(value,(list,tuple)):
            scalars[path+'.__type__']=type(value).__name__
            scalars[path+'.__length__']=len(value)
            for index,item in enumerate(value):visit(item,path+'.'+str(index))
        elif isinstance(value,float) and path.startswith('state.histories.'):
            tensors[path]=torch.tensor(value,dtype=torch.float64)
        else:scalars[path]=value
    visit(values,'state')
    return tensors,scalars


def encoded(tensors,scalars):
    raw=save(tensors)
    metadata=canonical_bytes(scalars)
    return len(metadata).to_bytes(8,'big')+metadata+raw


def write_science(path,snapshot):
    tensors,scalars=science(snapshot)
    raw=encoded(tensors,scalars)
    budget(len(raw))
    with gzip.open(path,'xb',compresslevel=6) as stream:stream.write(raw)
    return {**ref(path),'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'uncompressed_bytes':len(raw),'owners':len(tensors),'scalar_fields':len(scalars)}


def read_science(value):
    with gzip.open(auth(value),'rb') as stream:raw=stream.read()
    assert hashlib.sha256(raw).hexdigest()==value['uncompressed_sha256']
    n=int.from_bytes(raw[:8],'big')
    return load(raw[8+n:]),json.loads(raw[8:8+n])


def comparison(first,second,witness_path):
    a,sa=read_science(first);b,sb=read_science(second)
    discrete=[]
    witnesses={};scalar_witnesses={}
    for key in sorted(sa.keys()|sb.keys()):
        if sa.get(key)!=sb.get(key):
            discrete.append(key);scalar_witnesses[key]={'a':sa.get(key),'b':sb.get(key)}
    if a.keys()!=b.keys():discrete.append('tensor inventory')
    rows={}
    for key in sorted(a.keys()&b.keys()):
        x,y=a[key],b[key]
        if x.shape!=y.shape or x.dtype!=y.dtype:discrete.append(key);continue
        exact=torch.equal(x,y)
        if not exact:
            index=int((x.double()-y.double()).abs().flatten().argmax())
            witnesses[key]=torch.stack((x.flatten()[index],y.flatten()[index])).contiguous()

        forced=any(word in key for word in ('rng','optimizer_states')) and (not x.is_floating_point() or key.endswith('.step'))
        if not x.is_floating_point() or forced:
            if not exact:discrete.append(key)
            rows[key]={'exact':exact,'kind':'discrete','max_abs':0. if exact else None}
        else:
            diff=(x.double()-y.double()).abs()
            relative=diff/(x.double().abs().clamp_min(torch.finfo(torch.float32).tiny))
            finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
            ceiling=bool((diff<=1e-4+1e-4*x.double().abs()).all()) and finite
            rows[key]={'exact':exact,'kind':'floating','max_abs':float(diff.max()) if diff.numel() else 0.,
                       'max_rel':float(relative.max()) if relative.numel() else 0.,'changed_entries':int((x!=y).sum()),'ceiling_pass':ceiling,'finite':finite}
    witness_ref=None
    if witnesses:
        raw=save(witnesses);budget(len(raw));witness_path.write_bytes(raw);witness_ref=ref(witness_path)
    for key,row in rows.items():
        row['shape']=list(a[key].shape);row['dtype']=str(a[key].dtype)
        if key in witnesses:row['raw_witness_flat_index']=int((a[key].double()-b[key].double()).abs().flatten().argmax())
    return {'discrete_mismatches':discrete,'scalar_witnesses':scalar_witnesses,'raw_max_error_witnesses':witness_ref,'owners':rows,'ceiling_pass':not discrete and all(v.get('ceiling_pass',True) for v in rows.values())}


def child(case_id,arm,mode,first_step=1,checkpoint=None):
    start=time.perf_counter()
    design=json.loads((OUTPUT/'design.json').read_bytes())
    assert ref(Path(__file__))==design['script']
    resolved,_=_calibration_config(CONFIG)
    assert resolved.sha256==design['resolved_config_sha256']
    assert {k:v.sha256 for k,v in _automatic_identities().items()}==design['runtime_identity']
    case=design['cases'][int(case_id)]
    directory=OUTPUT/case['id']/arm
    directory.mkdir(parents=True,exist_ok=True)
    reference=ArtifactRef(**design['parents'][case['partition']])
    parent=json.loads(reference.path.read_bytes())
    if checkpoint:
        state=restore_snapshot(load_snapshot(ArtifactRef(**json.loads(checkpoint))), 'mps')
        ledger=state.ledger.run_ledger
        first_report=json.loads((directory/'replay-first.json').read_bytes())
        # Boundary write cost occurred after the immutable snapshot; preserve its
        # already durable accounting before charging this interpreter's restore.
        state.ledger.elapsed=first_report['cost']['elapsed']
        state.ledger.events=first_report['cost']['events']
        _event(state,'strict_h07_new_interpreter_restore',time.perf_counter()-start)
        initial=None
    else:
        state=import_warmstart(reference,_create(resolved,case['partition'],65536,808),partition=case['partition'],continuation_targets=65536)
        state.microbatch=case['packing']
        state.model.configure_quantization({**state.model._recipe,'name':case['recipe']})
        metadata=json.loads((ArtifactRef(**parent['checkpoint']).path.parents[2]/'run.json').read_bytes())
        metadata.pop('status_history')
        metadata.update({'run_id':'h08-contract-'+uuid.uuid4().hex,'created_utc':datetime.now(timezone.utc).isoformat(),
                         'warm_start_sha256':reference.sha256,'lineage':{'parent':metadata['run_id'],'resume':parent['checkpoint']['sha256']},
                         'endpoint':{'scope':'prospective full-step 8-step MPS shape calibration','design_sha256':ref(OUTPUT/'design.json')['sha256'],'case':case,'b0':None}})
        ledger=RunLedger(directory,metadata);state.ledger.run_ledger=ledger;ledger.record_status('running')
        parent_saved=_validated(load_snapshot(ArtifactRef(**parent['checkpoint'])))
        assert all(torch.equal(p.detach().cpu(),parent_saved['weights'][n]) for n,p in state.model.named_parameters())
        assert state.cursor.state['rng_states']==parent_saved['cursor']['rng_states']
        assert state.histories==state.quantizer_state=={} and state.schedule['step']==0
        assert all(not v for v in state.optimizer_state.values())
        initial=hashlib.sha256(encoded(*science(capture_snapshot(state)))).hexdigest()
        _event(state,'fresh_setup_authenticated_model_only_exact_initial',time.perf_counter()-start)
    states,steps={},[]
    saved_ref=None
    final_step=1 if mode=='smoke' else (4 if mode=='replay-first' else 8)
    with _Sampler('mps',design['caps']) as sampler:
        for step in range(int(first_step),final_step+1):
            assert not sampler.exceeded
            batch=select_logical_batch(state)
            identity=hashlib.sha256(canonical_bytes({'offsets':_offsets(batch)}))
            for block in batch.block_slices:
                for value in (block.inputs,block.labels,block.score_mask):
                    identity.update(canonical_bytes({'tokens':list(value)}))
            result,duration=_timed(state.model,lambda:train_step(state,batch))
            assert result.target_count==8192
            steps.append({'step':step,'targets':8192,'nll_sum':result.nll_sum,'input_sha256':identity.hexdigest(),'seconds':duration})
            if step in SELECTED:
                region=time.perf_counter()
                snapshot=capture_snapshot(state)
                states[str(step)]=write_science(directory/f'state-{step}.safetensors.gz',snapshot)
                _event(state,'scientific_owner_transfer_and_compressed_evidence',time.perf_counter()-region)
            if mode.startswith('replay') and step in (4,8):
                region=time.perf_counter()
                budget(64*2**20)
                saved_ref=save_snapshot(capture_snapshot(state),directory/'checkpoint')
                _event(state,'strict_h07_replay_checkpoint',time.perf_counter()-region)
    assert not sampler.exceeded
    elapsed=time.perf_counter()-start
    ledger.record({'step':state.schedule['step'],'stream':0,'unique':0,'additional_loss':0,'backward':0,'query':0,'elapsed_seconds':0.,'status':'project_elapsed_observation','project_elapsed_seconds':elapsed})
    if mode!='replay-first':ledger.record_status('completed',disposition='numerical calibration; comparison decision separate')
    report={'case':case,'arm':arm,'mode':mode,'initial_science_sha256':initial,'states':states,'steps':steps,'checkpoint':None if saved_ref is None else {'path':str(saved_ref.path),'sha256':saved_ref.sha256},
            'project_seconds':elapsed,'memory':memory_report(sampler.samples),'parent':design['parents'][case['partition']],
            'cost':{'stream':state.ledger.stream,'passes':state.ledger.passes,'elapsed':state.ledger.elapsed,'events':state.ledger.events},'source':design['runtime_identity']}
    _write(directory/('replay-first.json' if mode=='replay-first' else 'arm.json'),report)
    print(json.dumps(report))


def launch(case,arm,mode='full',first=1,checkpoint=None):
    command=[sys.executable,__file__,'child',str(case),arm,mode,str(first)]
    if checkpoint:command.append(json.dumps(checkpoint))
    process=subprocess.run(command,capture_output=True,text=True)
    directory=OUTPUT/json.loads((OUTPUT/'design.json').read_bytes())['cases'][case]['id']/arm
    (directory/f'{mode}.stdout.txt').write_text(process.stdout)
    (directory/f'{mode}.stderr.txt').write_text(process.stderr)
    if process.returncode:raise RuntimeError(f'child failed {case}/{arm}/{mode}: {process.stderr[-1500:]}')
    return json.loads(process.stdout)


def compare_arms(a,b,directory,label):
    result={str(step):comparison(a['states'][str(step)],b['states'][str(step)],directory/f'{label}-step{step}-witness.safetensors') for step in SELECTED}
    for x,y in zip(a['steps'],b['steps']):
        assert x['step']==y['step'] and x['input_sha256']==y['input_sha256']
        delta=abs(x['nll_sum']-y['nll_sum'])
        result.setdefault(str(x['step']),{'discrete_mismatches':[],'owners':{},'ceiling_pass':True})['owners']['nll_sum']={'kind':'floating','exact':delta==0.,'max_abs':delta,'max_rel':delta/max(abs(x['nll_sum']),1e-38),'ceiling_pass':delta<=1e-4+1e-4*abs(x['nll_sum'])}
    return result


def freeze():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    resolved,_=_calibration_config(CONFIG)
    parents=json.loads(Path(PARENTS).read_bytes())
    caps=_limits(resolved.payload)
    case_id=0;cases=[]
    for partition,recipe,packing in itertools.product(('meta_train','target_train'),('row_fixed_max','row_dynamic_max','quest_fp4_row_reference'),(1,64)):
        cases.append({'id':f'{case_id:02d}-{partition}-{recipe}-{packing}','partition':partition,'recipe':recipe,'packing':packing});case_id+=1
    design={'schema_version':1,'date_utc':datetime.now(timezone.utc).isoformat(),'status':'root-authorized prospective calibration DESIGN; not acceptance of old failed check',
            'script':ref(Path(__file__)),'resolved_config':json.loads(canonical_bytes(resolved.payload)),'resolved_config_sha256':resolved.sha256,
            'runtime_identity':{k:v.sha256 for k,v in _automatic_identities().items()},'parents':parents,'caps':caps,'cases':cases,
            'logical_targets':8192,'steps':8,'comparison_steps':list(SELECTED),'separate_runner_smoke_steps':1,'calibration_arms':4,'heldout_arms':['U0','U1','R4+4-new-interpreter'],
            'all_pairwise_calibration':True,'float_atol':1e-4,'float_rtol':1e-4,'relative_denominator_floor':torch.finfo(torch.float32).tiny,
            'exact_fields':'initial all scientific owners; source/cursor/RNG/schedule/counters/codes/masks/nonfloating histories/ordered inputs',
            'heldout_rule':'per-named-owner/per-step max_abs <= separately frozen all-pairwise four-arm duplicate maximum; zero remains zero; no enlargement',
            'cost_envelope':{'logical_steps':672,'continuation_targets':5505024,'replay_generations':24,'retention_forecast_bytes':1600000000,'peak_new_retained_and_scratch_bytes':2*2**30,'minimum_available_before_writes_bytes':2**30,'reserve_bytes':2**30},
            'storage':{'free_before':shutil.disk_usage(OUTPUT).free,'deletion_scope':'ONLY explicitly listed H08-v4 completed-rollout scientific comparison copies after authenticating summaries/raw witnesses; never parents/replay checkpoints/failure process artifacts/older tasks/user files'},
            'retention_amendment':'root versioned retention ruling: per-case streaming, <=2GiB new retained+scratch, >=1GiB available before writes; reconstruction from immutable parent/source/seed/recipe/input hashes; raw max-error witnesses retained for differing owners; numerical design unchanged',
            'scope':'proxy2layers,width128,heads4,FP32,MPS,fallback0; original-W AdamW probe; enumerated recipes/packing/8-step horizon/runtime only',
            'future_rule':'longer/core/new-method/runtime/packing horizons must prospectively characterize before comparisons; historical tiny exact stays historical; any failed ceiling/discrete/envelope scope stays failed this wave'}
    assert design['storage']['free_before']>design['cost_envelope']['retention_forecast_bytes']+design['cost_envelope']['reserve_bytes']
    artifact=_write(OUTPUT/'design.json',design)
    print(json.dumps({'path':str(artifact.path),'sha256':artifact.sha256}))


def wave():
    start=time.perf_counter();design=json.loads((OUTPUT/'design.json').read_bytes());results=[]
    for index,case in enumerate(design['cases']):
        calibration=[launch(index,f'D{i}') for i in range(4)]
        assert len({a['initial_science_sha256'] for a in calibration})==1
        directory=OUTPUT/case['id']
        comparisons=[compare_arms(a,b,directory,f'calibration-pair{number}') for number,(a,b) in enumerate(itertools.combinations(calibration,2))]
        envelope={};failures=[]
        for number,pair in enumerate(comparisons):
            for step,check in pair.items():
                if check['discrete_mismatches']:failures.append({'stage':'calibration','pair':number,'step':step,'discrete':check['discrete_mismatches']})
                for owner,values in check['owners'].items():
                    if not values.get('ceiling_pass',True):failures.append({'stage':'calibration','pair':number,'step':step,'ceiling':owner})
                    key=step+':'+owner
                    if values['kind']=='floating':envelope[key]=max(envelope.get(key,0.),values['max_abs'])
        directory=OUTPUT/case['id']
        envelope_ref=_write(directory/'frozen-envelope.json',{'design':ref(OUTPUT/'design.json'),'case':case,'calibration':[ref(directory/f'D{i}'/'arm.json') for i in range(4)],'all_six_pairs':comparisons,'per_owner_step_max_abs':envelope,'preheldout_failures':failures})
        # Publication/authentication occurs before any held-out launch.
        assert sha256_file(envelope_ref.path)==envelope_ref.sha256
        discard(calibration,envelope_ref)
        heldout=[launch(index,'U0'),launch(index,'U1')]
        first=launch(index,'R','replay-first');second=launch(index,'R','replay-second',5,first['checkpoint'])
        replay={**second,'initial_science_sha256':first['initial_science_sha256'],'states':{**first['states'],**second['states']},'steps':first['steps']+second['steps'],'project_seconds':first['project_seconds']+second['project_seconds']}
        _write(directory/'R'/'combined-arm.json',replay)
        assert all(a['initial_science_sha256']==calibration[0]['initial_science_sha256'] for a in heldout+[replay])
        heldchecks=[]
        for name,arm in [('duplicate',heldout[1]),('resume',replay)]:
            checks=compare_arms(heldout[0],arm,directory,'heldout-'+name)
            for step,check in checks.items():
                if check['discrete_mismatches']:failures.append({'stage':name,'step':step,'discrete':check['discrete_mismatches']})
                for owner,values in check['owners'].items():
                    if not values.get('ceiling_pass',True):failures.append({'stage':name,'step':step,'ceiling':owner})
                    if values['kind']=='floating' and values['max_abs']>envelope[step+':'+owner]:failures.append({'stage':name,'step':step,'envelope':owner,'observed':values['max_abs'],'frozen':envelope[step+':'+owner]})
            heldchecks.append({'name':name,'comparisons':checks})
        result={'case':case,'frozen_envelope':{'path':str(envelope_ref.path),'sha256':envelope_ref.sha256},'heldout':heldchecks,'failures':failures,'disposition':'failed scope; no enlargement' if failures else 'passes bounded enumerated scope only'}
        result_ref=_write(directory/'result.json',result)
        discard(heldout+[replay],result_ref)
        results.append({'case':case,'result':{'path':str(result_ref.path),'sha256':result_ref.sha256},'failure_count':len(failures),'disposition':result['disposition']})
        print(json.dumps(results[-1]),flush=True)
    refout=_write(OUTPUT/'wave.json',{'design':ref(OUTPUT/'design.json'),'results':results,'project_seconds':time.perf_counter()-start,'all_work_retained':True,'b0':None})
    print(json.dumps({'path':str(refout.path),'sha256':refout.sha256}),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='wave':wave()
    elif sys.argv[1]=='child':child(*sys.argv[2:])
