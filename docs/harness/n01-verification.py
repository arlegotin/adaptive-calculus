"""N01 bounded source-frozen actual-MPS replay and capacity verification.

The unchanged H08 serialization/comparison helper is reused read-only; its
numerical envelope/controller is never launched. Retain all N01 evidence: no
operational deletion is needed under the prospective storage forecast.
"""
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime,timezone

import torch
from safetensors.torch import load_file,save_file
from ralup.__main__ import _calibration_config
from ralup.calibrate import (_create,_event,_limits,_offsets,_Sampler,_timed,_write,
                            import_warmstart,memory_report,measure_capacity)
from ralup.config import canonical_bytes
from ralup.models.embedding import PRIMITIVE_ID,ordered_embedding
from ralup.records import ArtifactRef,RunLedger,sha256_file
from ralup.state import (_automatic_identities,_validated,capture_snapshot,
                         load_snapshot,restore_snapshot,save_snapshot)
from ralup.train import select_logical_batch,train_step

OUTPUT=Path('artifacts/calibration/n01-v1')
DESIGN=Path('docs/harness/n01-design.json')
CONFIG='configs/h08-real-proxy.json'
HISTORICAL=Path('docs/harness/h08-mps-contract.py')
spec=importlib.util.spec_from_file_location('historical_serialization',HISTORICAL)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
helper.OUTPUT=OUTPUT
helper.SELECTED=tuple(range(1,9))


def ref(path):return {'path':str(path),'sha256':sha256_file(path)}


def budget(extra=0):
    retained=sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())
    assert retained+extra<=2*2**30,'N01 retained+scratch 2GiB ceiling'
    assert shutil.disk_usage(OUTPUT).free>=2**30+extra,'N01 free storage floor before write'
    return retained


def publish(path,payload):
    budget(len(canonical_bytes(payload)))
    return _write(path,payload)


def design():
    value=json.loads(DESIGN.read_bytes())
    assert ref(Path(__file__))==value['runner']
    assert ref(HISTORICAL)==value['read_only_serialization_helper']
    assert {k:v.sha256 for k,v in _automatic_identities().items()}==value['producer']
    assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0' and torch.backends.mps.is_available()
    resolved,_=_calibration_config(CONFIG)
    assert resolved.sha256==value['resolved_config_sha256']
    return value,resolved


def scientific(snapshot):
    tensors,scalars=helper.science(snapshot)
    payload=_validated(snapshot)
    # Scientific work/coverage owners compare exactly; elapsed/process/path IDs
    # and I/O events differ across uninterrupted versus restored arms.
    scalars['scientific_work']= {k:payload['ledger'][k] for k in ('stream','coverage','passes')}
    return tensors,scalars


def write_science(path,snapshot):
    import gzip
    tensors,scalars=scientific(snapshot)
    raw=helper.encoded(tensors,scalars)
    budget(len(raw))
    with gzip.open(path,'xb',compresslevel=1) as stream:stream.write(raw)
    # Hash every individual owner as well as the complete scientific record.
    manifest={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()} for k,v in tensors.items()}
    return {**ref(path),'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),
            'uncompressed_bytes':len(raw),'owners':manifest,'scalar_sha256':hashlib.sha256(canonical_bytes(scalars)).hexdigest()}


def child(case_index,arm,mode='full',checkpoint=None):
    start=time.perf_counter();frozen,resolved=design()
    case=frozen['cases'][int(case_index)]
    directory=OUTPUT/case['recipe']/arm;directory.mkdir(parents=True,exist_ok=True)
    parent_ref=ArtifactRef(**frozen['parent']);parent=json.loads(parent_ref.path.read_bytes())
    if checkpoint:
        state=restore_snapshot(load_snapshot(ArtifactRef(**json.loads(checkpoint))),'mps')
        first=json.loads((directory/'first.json').read_bytes())
        state.ledger.elapsed=first['cost']['elapsed'];state.ledger.events=first['cost']['events']
        initial=first['initial_science_sha256']
        _event(state,'strict_fresh_interpreter_restore',time.perf_counter()-start)
        first_step=5
    else:
        state=import_warmstart(parent_ref,_create(resolved,'target_train',65536,808),partition='target_train',continuation_targets=65536)
        state.microbatch=64
        state.model.configure_quantization({**state.model._recipe,'name':case['recipe']})
        saved=_validated(load_snapshot(ArtifactRef(**parent['checkpoint'])))
        assert all(torch.equal(p.detach().cpu(),saved['weights'][n]) for n,p in state.model.named_parameters())
        assert state.cursor.state=={**saved['cursor'],'run_target_budget':saved['cursor']['cumulative_target_count']+65536}
        assert state.ledger.passes==parent['cost']['passes']
        metadata=json.loads((ArtifactRef(**parent['checkpoint']).path.parents[2]/'run.json').read_bytes());metadata.pop('status_history')
        metadata.update({'run_id':'n01-'+uuid.uuid4().hex,'created_utc':datetime.now(timezone.utc).isoformat(),
                         'warm_start_sha256':parent_ref.sha256,'lineage':{'parent':metadata['run_id'],'resume':parent['checkpoint']['sha256']},
                         'endpoint':{'scope':'N01 bounded 8-step ordered embedding','design':ref(DESIGN),'recipe':case['recipe'],'b0':None}})
        state.ledger.run_ledger=RunLedger(directory,metadata);state.ledger.run_ledger.record_status('running')
        initial=hashlib.sha256(helper.encoded(*scientific(capture_snapshot(state)))).hexdigest()
        _event(state,'fresh_setup_authenticated_historical_model_only',time.perf_counter()-start)
        first_step=1
    states={};steps=[];checkpoints={}
    final=4 if mode=='first' else 8
    with _Sampler('mps',frozen['caps']) as sampler:
        for step in range(first_step,final+1):
            assert not sampler.exceeded
            assert time.perf_counter()-start<frozen['envelopes']['arm_wall_seconds']
            batch=select_logical_batch(state)
            digest=hashlib.sha256(canonical_bytes({'offsets':_offsets(batch)}))
            for block in batch.block_slices:
                digest.update(canonical_bytes({'inputs':list(block.inputs),'positions':list(block.positions),'labels':list(block.labels),'mask':list(block.score_mask)}))
            result,duration=_timed(state.model,lambda:train_step(state,batch))
            assert result.target_count==8192
            steps.append({'step':step,'targets':8192,'nll_sum':result.nll_sum,'input_sha256':digest.hexdigest(),'full_step_seconds':duration})
            region=time.perf_counter();snapshot=capture_snapshot(state)
            states[str(step)]=write_science(directory/f'state-{step}.safetensors.gz',snapshot)
            # Both PRE-FORWARD storage_u/scales/codes and POST-COMMIT W /
            # cached endpoint codes/scales are retained inside every state.
            _event(state,'snapshot_transfer_owner_hash_and_retention',time.perf_counter()-region)
            if step in (4,8):
                budget(frozen['storage']['checkpoint_scratch_forecast_bytes'])
                region=time.perf_counter()
                checkpoint_ref=save_snapshot(capture_snapshot(state),directory/'checkpoints')
                checkpoints[str(step)]={'path':str(checkpoint_ref.path),'sha256':checkpoint_ref.sha256}
                _event(state,'checkpoint_capture_publication',time.perf_counter()-region)
    assert not sampler.exceeded
    report={'case':case,'arm':arm,'mode':mode,'initial_science_sha256':initial,'states':states,'steps':steps,'checkpoints':checkpoints,
            'parent':frozen['parent'],'source':frozen['producer'],'primitive':PRIMITIVE_ID,'design':ref(DESIGN),
            'quantizer_context_contract':'state.model_cache._contexts contains pre-forward storage_u/scales/codes at schedule.step-1; state.weights is postcommit W; state.histories contains endpoint cached_codes/scales. identity rotation makes endpoint U=W.',
            'memory':memory_report(sampler.samples),'project_seconds':time.perf_counter()-start,
            'cost':{'stream':state.ledger.stream,'passes':state.ledger.passes,'elapsed':state.ledger.elapsed,'events':state.ledger.events}}
    publish(directory/('first.json' if mode=='first' else 'arm.json'),report)
    if mode!='first':state.ledger.run_ledger.record_status('completed',disposition='bounded scope pending comparison')
    print(json.dumps({'report':ref(directory/('first.json' if mode=='first' else 'arm.json')),'project_seconds':report['project_seconds']}))


def launch(index,arm,mode='full',checkpoint=None):
    frozen,_=design()
    command=[sys.executable,__file__,'child',str(index),arm,mode]
    if checkpoint:command.append(json.dumps(checkpoint))
    start=time.perf_counter()
    result=subprocess.run(command,capture_output=True,text=True,timeout=frozen['envelopes']['arm_wall_seconds']+30)
    directory=OUTPUT/frozen['cases'][index]['recipe']/arm
    publish(directory/(mode+'-process.json'),{'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'process_seconds':time.perf_counter()-start,'producer':frozen['producer'],'design':ref(DESIGN)})
    if result.returncode:raise RuntimeError(f'{arm}/{mode} failed: {result.stderr[-1000:]}')
    return json.loads((directory/('first.json' if mode=='first' else 'arm.json')).read_bytes())


def wave():
    start=time.perf_counter();frozen,_=design();results=[]
    for index,case in enumerate(frozen['cases']):
        u0=launch(index,'U0');u1=launch(index,'U1')
        first=launch(index,'R','first');second=launch(index,'R','second',first['checkpoints']['4'])
        replay={**second,'states':{**first['states'],**second['states']},'steps':first['steps']+second['steps'],
                'checkpoints':{**first['checkpoints'],**second['checkpoints']},'project_seconds':first['project_seconds']+second['project_seconds']}
        directory=OUTPUT/case['recipe'];publish(directory/'R'/'combined.json',replay)
        failures=[];comparisons=[]
        if len({u0['initial_science_sha256'],u1['initial_science_sha256'],replay['initial_science_sha256']})!=1:failures.append({'owner':'initial_science'})
        for label,other in [('duplicate',u1),('strict_resume',replay)]:
            checks=helper.compare_arms(u0,other,directory,label)
            for step,check in checks.items():
                if check['discrete_mismatches']:failures.append({'comparison':label,'step':step,'owners':check['discrete_mismatches']})
                for owner,row in check['owners'].items():
                    if not row['exact']:failures.append({'comparison':label,'step':step,'owner':owner,'observed':row})
            comparisons.append({'label':label,'steps':checks})
        result={'case':case,'comparisons':comparisons,'failures':failures,'accepted':not failures,
                'arms':[ref(directory/a/p) for a,p in [('U0','arm.json'),('U1','arm.json'),('R','combined.json')]],
                'retention':'all complete per-step paired owners retained; no discard, no unavailable local quantizer operands'}
        publish(directory/'result.json',result);results.append({'recipe':case['recipe'],'accepted':not failures,'failure_count':len(failures),'result':ref(directory/'result.json')})
        print(json.dumps(results[-1]),flush=True)
        if failures:
            publish(OUTPUT/'wave.json',{'design':ref(DESIGN),'results':results,'project_seconds':time.perf_counter()-start,'status':'FAILED; remaining scopes unlaunched; root ruling required'})
            raise RuntimeError('Required exact replay gate failed; no altered acceptance or alternative wave')
    publish(OUTPUT/'wave.json',{'design':ref(DESIGN),'results':results,'project_seconds':time.perf_counter()-start,'status':'PASS bounded enumerated scope only','new_continuation_steps':72,'new_scored_bytes':589824})


def components():
    start=time.perf_counter();frozen,resolved=design()
    historical=json.loads(Path('artifacts/calibration/h08-embedding-diagnostic-v1/diagnostic.json').read_bytes())
    payload=load_file(str(helper.auth(historical['repeat_data'])))
    rows=[];arrays={}
    with _Sampler('mps',frozen['caps']) as sampler:
        for shape in ('saved_actual_input','full_shape_tiled_actual_input'):
            indices=payload['indices'] if shape=='saved_actual_input' else payload['indices'].repeat(64,1)
            upstream=payload['upstream'] if shape=='saved_actual_input' else payload['upstream'].repeat(64,1,1)
            cpu_weight=payload['weight'].clone().requires_grad_()
            baseline=torch.autograd.grad(ordered_embedding(indices,cpu_weight),cpu_weight,upstream)[0]
            for device in ('cpu','mps'):
                gradients=[];timings=[]
                for repeat in range(12):
                    region=time.perf_counter()
                    weight=payload['weight'].to(device).clone().requires_grad_()
                    local_indices=indices.to(device);local_upstream=upstream.to(device)
                    if device=='mps':torch.mps.synchronize()
                    transfer=time.perf_counter()-region
                    region=time.perf_counter();output=ordered_embedding(local_indices,weight)
                    gradient=torch.autograd.grad(output,weight,local_upstream)[0]
                    if device=='mps':torch.mps.synchronize()
                    timings.append({'repeat':repeat,'transfer_seconds':transfer,'gather_selector_gemm_seconds':time.perf_counter()-region})
                    assert torch.equal(output.detach().cpu(),payload['weight'][indices])
                    gradients.append(gradient.cpu().contiguous());arrays[f'{shape}_{device}_{repeat}']=gradients[-1]
                assert all(torch.equal(gradients[0],v) for v in gradients[1:]),'candidate actual input repeat failed'
                assert torch.allclose(gradients[0],baseline,atol=1e-4,rtol=1e-4),'candidate parity failed'
                rows.append({'shape':shape,'device':device,'input_shape':list(indices.shape),'repeats':12,'same_device_exact':True,'cpu_parity':True,'timings':timings,
                             'selector_elements':indices.numel()*257,'selector_fp32_bytes':indices.numel()*257*4,'selector_bool_bytes':indices.numel()*257,
                             'pass_cost':{'backward_context_tokens':indices.numel()*12,'gather_context_tokens':indices.numel()*12}})
        # Separate selector allocation and GEMM timings on actual device, shared
        # by token and learned-position embeddings; no backend determinism claim.
        generator=torch.Generator().manual_seed(901);component=[]
        for vocabulary in (257,256):
            indices=torch.randint(vocabulary,(64,256),generator=generator).to('mps')
            upstream=torch.randn(16384,128,generator=generator).to('mps');torch.mps.synchronize()
            for repeat in range(12):
                region=time.perf_counter();selector=(indices.reshape(-1,1)==torch.arange(vocabulary,device='mps')).to(torch.float32);torch.mps.synchronize()
                allocation=time.perf_counter()-region
                region=time.perf_counter();gradient=selector.T@upstream;torch.mps.synchronize()
                component.append({'vocabulary':vocabulary,'repeat':repeat,'selector_allocation_seconds':allocation,'gemm_seconds':time.perf_counter()-region,
                                  'selector_fp32_bytes':selector.numel()*4,'selector_bool_bytes':selector.numel(),'upstream_bytes':upstream.numel()*4,'gradient_bytes':gradient.numel()*4})
    budget(sum(v.numel()*v.element_size() for v in arrays.values())+2**20)
    save_file(arrays,str(OUTPUT/'component-gradients.safetensors'))
    publish(OUTPUT/'components.json',{'design':ref(DESIGN),'historical_saved_input_report':ref(Path('artifacts/calibration/h08-embedding-diagnostic-v1/diagnostic.json')),
                                     'historical_variation_read_only':historical['rows'],'rows':rows,'component_times':component,'gradients':ref(OUTPUT/'component-gradients.safetensors'),
                                     'memory':memory_report(sampler.samples),'project_seconds':time.perf_counter()-start,'status':'PASS','all_setup_transfer_observer_io_charged_to_project_seconds':True})


def capacity():
    start=time.perf_counter();frozen,resolved=design()
    parent=ArtifactRef(**frozen['parent'])
    state=import_warmstart(parent,_create(resolved,'target_train',1048576,808),partition='target_train',continuation_targets=1048576)
    _event(state,'n01_capacity_setup_parent_import',time.perf_counter()-start)
    # Existing source-owned capacity harness performs synchronized geometric
    # search, randomized recipe arms, finite warmup/query/I/O and exact resets.
    budget(frozen['storage']['capacity_scratch_forecast_bytes'])
    report=measure_capacity(state,[1,2,4,8,16,32,64,128],repeats=2,logical_targets=8192,
                            recipes=frozen['recipes'],output=OUTPUT/'capacity-search',caps=frozen['caps'])
    search_seconds=time.perf_counter()-start
    assert search_seconds<frozen['envelopes']['capacity_wall_seconds']
    selected=report['selected_packing']
    sustained=measure_capacity(state,[selected],repeats=4,logical_targets=8192,
                               recipes=frozen['recipes'],output=OUTPUT/'capacity-sustained',caps=frozen['caps'])
    publish(OUTPUT/'capacity.json',{'design':ref(DESIGN),'primitive':PRIMITIVE_ID,'producer':frozen['producer'],
                                   'search':report,'sustained':sustained,'selected_packing':selected,'project_seconds':time.perf_counter()-start,
                                   'bounds':'short sustained pilot only; B0/duration unresolved; shared overhead in every full transaction',
                                   'old_h08_timings':'historical only; never consulted for new-source selection'})


def freeze():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    resolved,_=_calibration_config(CONFIG)
    parents=json.loads(Path('artifacts/checks/h08-real-warmstarts-cli.txt').read_bytes())
    frozen={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'status':'prospective N01 bounded verification; freeze before outcomes',
            'primitive':PRIMITIVE_ID,'backend_selection':'same device as weight/indices; FP32 product; native gather and constant equality-selector transpose GEMM; no automatic fallback',
            'runner':ref(Path(__file__)),'read_only_serialization_helper':ref(HISTORICAL),
            'producer':{k:v.sha256 for k,v in _automatic_identities().items()},'resolved_config':json.loads(canonical_bytes(resolved.payload)),
            'resolved_config_sha256':resolved.sha256,'parent':parents['target_train'],'caps':_limits(resolved.payload),
            'recipes':['row_fixed_max','row_dynamic_max','quest_fp4_row_reference'],
            'cases':[{'recipe':r,'partition':'target_train','packing':64} for r in ['row_fixed_max','row_dynamic_max','quest_fp4_row_reference']],
            'arms':['U0 independent 8 steps','U1 independent 8 steps','R strict fresh interpreter 4+4'],
            'logical_bytes':8192,'continuation_steps':72,'continuation_scored_bytes':589824,'comparison_steps':list(range(1,9)),
            'exact_contract':'initial and every matched step all weights/gradients/model/optimizer/cache/scales/codes/histories/schedule/cursor/ordered inputs/named+global RNG and scientific stream/coverage/pass counters, plus summed NLL. Any difference fails scope.',
            'exclusions':'elapsed/I/O/process audit/path/run IDs; preserve and charge separately; no exclusion of scientific state or work counters',
            'cpu_mps_parity':{'atol':1e-4,'rtol':1e-4},'failure_policy':'stop at first failed scope, retain all evidence, seek root ruling; no bounds enlarged or unplanned waves',
            'component_plan':{'saved_actual_input_repeats_per_device':12,'full_shape_tiled_repeats_per_device':12,'selector_gemm_full_shape_repeats_each_vocabulary':12,'vocabularies':[257,256],
                              'unit_full_shape_repeats_per_device_vocabulary':12,'unit_higher_order':'gradcheck, gradgradcheck, torch.func grad/jacrev and functional model grad; literal repeated-index sum, absent zeros, boundary256, dropped-owner and wrong-derivative controls'},
            'capacity_plan':{'observed_geometric_sizes':[1,2,4,8,16,32,64,128],'search_repeats_per_recipe_size':2,'stop':'worst recipe sustained rate fails to improve vs previous observed size OR memory cap/OOM; geometric expansion cannot continue past128',
                             'sustained_repeats_per_recipe':4,'randomization':'existing authenticated timing RNG shuffle recipe arms at every observed size; each restored from identical snapshot',
                             'shared_overhead':'selector boolean/FP32 allocation, transfer, GEMM; all included in full steps; separate actual component clocks and inventories',
                             'maximum_new_full_steps':64,'warmup_and_query':'two finite capacity invocations; each one warmup step and one8192-byte query, plus reset/snapshot/read/write/new-interpreter load costs'},
            'envelopes':{'arm_wall_seconds':180,'wave_wall_seconds':2400,'capacity_wall_seconds':1800,'component_wall_seconds':180,'covering_tests_wall_seconds':360,
                         'basis':'H08 observed approximately1-second full steps and10–25-second retained arms, plus conservative6–10x startup/snapshot allowance; fixed before new outcomes'},
            'storage':{'retention_forecast_bytes':1400000000,'retained_plus_scratch_cap_bytes':2*2**30,'minimum_free_before_writes_bytes':2**30,
                       'checkpoint_scratch_forecast_bytes':80*2**20,'capacity_scratch_forecast_bytes':160*2**20,'free_before_bytes':shutil.disk_usage(OUTPUT).free,
                       'retention':'all72 compressed complete per-step scientific owner records; checkpoints4/8 every arm; both local pre-forwardU/scales/codes and endpointW/scales/codes/context IDs retained; no deletion required',
                       'deletion_scope':'none; if forecast insufficient stop and seek ruling. Never remove earlier failures/task/user files'},
            'downstream':'accepted scope/packing only to M01/M02/M07/M08/M10/M11/E01/E02/E03/E07. Core/longer/new-method scopes need risk-based prospective validation; scientific budgets unresolved; protected test denied',
            'historical':'all12 H08 native v4 scopesFAILED; old source archives/costs/timings and forensic limits unchanged; no old Cartesian wave rerun'}
    assert frozen['storage']['free_before_bytes']>=frozen['storage']['retention_forecast_bytes']+2**30
    _write(DESIGN,frozen);publish(OUTPUT/'design-receipt.json',{'design':ref(DESIGN),'frozen_before_outcomes':True})
    print(json.dumps(ref(DESIGN)))


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='freeze':freeze()
    elif mode=='child':child(*sys.argv[2:])
    elif mode=='wave':wave()
    elif mode=='components':components()
    elif mode=='capacity':capacity()
    else:raise ValueError(mode)
