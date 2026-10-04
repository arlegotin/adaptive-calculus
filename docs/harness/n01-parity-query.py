"""One root-authorized, prospectively frozen whole-model parity query pair."""
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime,timezone

import torch
from safetensors.torch import save_file
from ralup.__main__ import _calibration_config
from ralup.calibrate import _create,_event,_Sampler,_timed,import_warmstart,memory_report
from ralup.config import canonical_bytes
from ralup.data.batches import pack_microbatches
from ralup.evaluate import summed_nll
from ralup.quantization.context import prepare_context
from ralup.state import _automatic_identities,_device_tree,_validate_model_owners
from ralup.records import ArtifactRef
from ralup.train import select_logical_batch
import importlib.util
spec=importlib.util.spec_from_file_location('n01_core_verification','docs/harness/n01-verification.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
DESIGN=Path('docs/harness/n01-parity-supplement.json')
OUTPUT=core.OUTPUT/'parity-supplement'


def freeze():
    primary,resolved=core.design()
    frozen={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'authority':'root authorized one supplementary8192-byte whole-model CPU/MPS query before outcomes; no optimizer/update/recipe matrix/full-suite repetition',
            'runner':core.ref(Path(__file__)),'primary_design':core.ref(core.DESIGN),'producer':primary['producer'],'parent':primary['parent'],
            'resolved_config_sha256':resolved.sha256,'logical_bytes':8192,'physical_packing':64,
            'recipe':{'name':'quest_fp4_row_reference','rotation':'full_hadamard','backward':'trust'},
            'risk_choice':'QuEST rotated/trust composition selected before outcomes; distinct extra coverage, not extension of primary identity-rotation replay acceptance',
            'initial_contract':'authenticate identical historical parent weights, continuation cursor and named/global RNG; same selected ordered bytes/masks/positions; no optimizer updates',
            'contexts':'prepare once on CPU from bitwise parent weights at step0; identical detached storageU/Q/scales/codes/trustmasks transferred to MPS; numeric context parity is reused from F03/F04, composition is this gate',
            'loss':'summed NLL across same64-packed microbatches via math.fsum',
            'gradient':'backpropagate SUM NLL then divide all gradients once by8192; compare full logical-average scientific gradients before clip/update (H06). Retain raw SUM and averaged gradients for both devices.',
            'acceptance':{'loss_atol':1e-4,'loss_rtol':1e-4,'gradient_atol':1e-4,'gradient_rtol':1e-4,'finite':True,'any_failed_owner':'supplement CLOSED; no enlargement'},
            'cost':'one distinct forward/backward query pass per device8192 scored bytes; query/additional_loss/loss_targets/backward classify the same pass, never summed as extra distinct passes; startup/setup/contexts/transfers/observer/tensor retention charged separately',
            'envelope':{'wall_seconds':180,'new_scored_query_bytes':16384,'backward_scored_bytes':16384,'new_optimizer_updates':0,'retained_scratch_forecast_bytes':40*2**20},
            'retention':'exact inputs/cursor, shared quantizer context numeric tensors+metadata, identical initial weights, raw/averaged paired gradients and loss, per-owner discrepancies, producer/parent/design/event identities; no disposal',
            'bounds_scope':'this single initial parent/byte sequence/packed QuEST rotated/trust query; no universal backend/longer/core/optimizer admission'}
    core.publish(DESIGN,frozen)
    print(json.dumps(core.ref(DESIGN)))


def run():
    start=time.perf_counter();primary,resolved=core.design()
    frozen=json.loads(DESIGN.read_bytes());assert core.ref(Path(__file__))==frozen['runner']
    assert primary['producer']==frozen['producer']
    OUTPUT.mkdir(exist_ok=False);core.budget(frozen['envelope']['retained_scratch_forecast_bytes'])
    parent=ArtifactRef(**frozen['parent'])
    state=import_warmstart(parent,_create(resolved,'target_train',8192,808),partition='target_train',continuation_targets=8192)
    state.model.to('cpu');state.microbatch=64
    _validate_model_owners(state.model)
    batch=select_logical_batch(state)
    assert batch.target_count==8192
    inputs={'cursor_after_selection':state.cursor.state,'blocks':[dataclasses.asdict(b) for b in batch.block_slices]}
    # TargetBlock offsets/inputs are JSON tuples, convert through standard JSON.
    inputs=json.loads(json.dumps(inputs));core.publish(OUTPUT/'inputs.json',inputs)
    state.model.configure_quantization(frozen['recipe'])
    parameters=dict(state.model.named_parameters())
    region=time.perf_counter()
    contexts={item['name']:prepare_context(parameters[item['name']],frozen['recipe'],0,None) for item in state.model.quantized_matrix_manifest()}
    arrays={f'initial.{n}':p.detach().cpu().contiguous().clone() for n,p in parameters.items()}
    context_records={}
    for name,c in contexts.items():
        fields={}
        for field in dataclasses.fields(c):
            value=getattr(c,field.name)
            if isinstance(value,torch.Tensor):arrays[f'context.{name}.{field.name}']=value.cpu().contiguous()
            elif field.name=='metadata':fields[field.name]=dict(value)
            else:fields[field.name]=value
        context_records[name]=fields
    setup_seconds=time.perf_counter()-start
    context_seconds=time.perf_counter()-region
    results=[]
    with _Sampler('mps',primary['caps']) as sampler:
        for device in ('cpu','mps'):
            region=time.perf_counter()
            state.model.to(device);state.model.zero_grad(set_to_none=True)
            local_contexts=_device_tree(contexts,device)
            for n,p in state.model.named_parameters():assert torch.equal(p.detach().cpu(),arrays['initial.'+n])
            for n,c in local_contexts.items():
                for field in dataclasses.fields(c):
                    value=getattr(c,field.name)
                    if isinstance(value,torch.Tensor):assert torch.equal(value.detach().cpu(),arrays[f'context.{n}.{field.name}'])
            if device=='mps':torch.mps.synchronize()
            transfer_seconds=time.perf_counter()-region
            state.model._contexts=local_contexts;state.model._context_step=0
            region=time.perf_counter();nlls=[]
            for micro in pack_microbatches(batch,64):
                loss=summed_nll(state.model,micro);loss.backward();nlls.append(float(loss.detach().cpu()))
            if device=='mps':torch.mps.synchronize()
            nll=math.fsum(nlls)
            for name,p in state.model.named_parameters():
                arrays[f'{device}.raw_gradient.{name}']=p.grad.detach().cpu().contiguous().clone()
                p.grad.div_(8192)
                arrays[f'{device}.gradient.{name}']=p.grad.detach().cpu().contiguous().clone()
            duration=time.perf_counter()-region
            event={'event_id':f'n01-parity-{device}-single-query','device':device,'distinct_passes':1,'scored_bytes':8192,
                   'query':8192,'additional_loss':8192,'loss_targets':8192,'backward':8192,
                   'forward_context':sum(len(b.inputs) for b in batch.block_slices),'query_seconds':duration,'transfer_authentication_seconds':transfer_seconds,'nll_sum':nll}
            results.append(event)
            assert not sampler.exceeded
            assert time.perf_counter()-start<frozen['envelope']['wall_seconds']
    checks={};failures=[]
    cpu_nll,mps_nll=results[0]['nll_sum'],results[1]['nll_sum']
    loss_pass=math.isfinite(cpu_nll) and math.isfinite(mps_nll) and abs(cpu_nll-mps_nll)<=1e-4+1e-4*abs(cpu_nll)
    if not loss_pass:failures.append('summed_nll')
    for name in parameters:
        a,b=arrays[f'cpu.gradient.{name}'],arrays[f'mps.gradient.{name}']
        difference=(a.double()-b.double()).abs()
        passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all()) and bool((difference<=1e-4+1e-4*a.double().abs()).all())
        raw=(arrays[f'cpu.raw_gradient.{name}'].double()-arrays[f'mps.raw_gradient.{name}'].double()).abs()
        checks[name]={'pass':passed,'max_abs':float(difference.max()),'changed_entries':int((a!=b).sum()),'raw_sum_gradient_max_abs':float(raw.max()),'shape':list(a.shape),'dtype':str(a.dtype)}
        if not passed:failures.append(name)
    core.budget(sum(v.numel()*v.element_size() for v in arrays.values())+2**20)
    save_file(arrays,str(OUTPUT/'paired-owners.safetensors'))
    core.publish(OUTPUT/'report.json',{'design':core.ref(DESIGN),'producer':primary['producer'],'parent':frozen['parent'],
                                    'inputs':core.ref(OUTPUT/'inputs.json'),'paired_owners':core.ref(OUTPUT/'paired-owners.safetensors'),'contexts':context_records,
                                    'query_events':results,'unique_distinct_query_passes':2,'new_scored_query_bytes':16384,'new_optimizer_updates':0,
                                    'summed_nll':{'cpu':cpu_nll,'mps':mps_nll,'max_abs':abs(cpu_nll-mps_nll),'pass':loss_pass},'gradients':checks,'failures':failures,
                                    'memory':memory_report(sampler.samples),'setup_seconds':setup_seconds,'context_preparation_seconds_included_in_setup':context_seconds,
                                    'project_seconds':time.perf_counter()-start,'accepted':not failures,'bounds_scope':frozen['bounds_scope']})
    print(json.dumps({'accepted':not failures,'failure_count':len(failures),'report':core.ref(OUTPUT/'report.json')}))
    if failures:raise RuntimeError('Supplementary whole-model parity gate CLOSED; no bounds enlarged')


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='run':run()
