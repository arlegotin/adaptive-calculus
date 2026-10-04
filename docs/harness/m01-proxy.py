"""Frozen M01 finite development trajectories, never scientific selection.

Every arm starts from charged fresh initialization. No historical model-only
parent, query, protected data, performance selection or budget inference exists.
"""
from datetime import datetime, timezone
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

import torch
from safetensors.torch import save
from ralup.config import canonical_bytes, resolve_config
from ralup.records import ArtifactRef, RunLedger, WorkLedger, sha256_file
from ralup.data.cursor import BatchCursor
from ralup.models.transformer import ByteTransformer, ModelConfig
from ralup.models.embedding import PRIMITIVE_ID
from ralup.optim.base import default_parameter_groups, SchedulePoint
from ralup.baselines.registry import build_matrix_method
from ralup.calibrate import _event, _offsets, _timed, _write, _Sampler, _limits, memory_report
from ralup.state import capture_snapshot, load_snapshot, restore_snapshot, save_snapshot, _automatic_identities, _validated, _Encoder, _manifest
from ralup.train import select_logical_batch, train_step

ROOT=Path.cwd()
DURABLE=ROOT/'artifacts/checks/m01-v1'
TMP=Path('/private/tmp/ralup-m01-20261004-v3')
DESIGN=ROOT/'docs/harness/m01-design-v4.json'
STORE_PATH=ROOT/'docs/harness/m01-evidence-store.py'
spec=importlib.util.spec_from_file_location('m01_evidence',STORE_PATH)
store=importlib.util.module_from_spec(spec);spec.loader.exec_module(store)
OBJECTS=DURABLE/'objects'


def ref(path):return {'path':str(path),'sha256':sha256_file(path)}


def retained(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0


def budget(extra_tmp=0, extra_durable=0):
    assert retained(TMP)+extra_tmp<=2*2**30,'M01 temporary retained+scratch envelope'
    assert retained(DURABLE)+extra_durable<=192*2**20,'M01 durable essentials envelope'
    assert shutil.disk_usage(ROOT).free>=2**30+extra_durable,'Actual git-volume free floor'
    assert shutil.disk_usage('/private/tmp').free>=2*2**30+extra_tmp,'Actual tmp-volume free floor'


def science(snapshot):
    p=_validated(snapshot)
    # Retain all scientific owners/cursor/RNG/work; exclude observation clocks,
    # I/O-only event rows and process identity from same-device numerical checks.
    values={k:v for k,v in p.items() if k not in ('process_audit','run_ledger','cursor_audit','ledger')}
    values['scientific_work']={k:p['ledger'][k] for k in ('stream','coverage','passes')}
    enc=_Encoder();metadata=enc.encode(values)
    return hashlib.sha256(canonical_bytes(metadata)+save(enc.tensors)).hexdigest()


def freeze():
    template=json.loads(Path('configs/h08-real-proxy.json').read_bytes())
    config=template['config'];config.pop('calibration')
    config['training']['microbatch']=64
    config['training']['recipe']='row_dynamic_max'
    config['quantization']['rotation']='hadamard'
    resolved=resolve_config(config,template['evidence'],purpose='pilot-bootstrap')
    baseline=Path('artifacts/calibration/n01-v1/row_fixed_max/U0/checkpoints')
    measured=[{'path':str(g),'bytes':retained(g)} for g in sorted(baseline.glob('g-*'))]
    value={'schema':'m01.bounded_development_design.v1','purpose':'pilot-bootstrap','scientific_gate':None,
        'recorded_utc':datetime.now(timezone.utc).isoformat(),'chronological_origin':json.loads((DURABLE/'proxy-wave-start.json').read_bytes())['utc'],
        'prior_failed_wrapper':{'start':ref(DURABLE/'proxy-wave-start.json'),'finish':ref(DURABLE/'proxy-wave-finish.json'),
                                'child_start':ref(DURABLE/'proxy-muon_project_fp32-U-full-start.json'),'child_finish':ref(DURABLE/'proxy-muon_project_fp32-U-full-finish.json'),
                                'runner_bytes':ref(ROOT/'docs/harness/m01-proxy-v2-failed.py'),'new_updates':0,'disposition':'FAILED before forward; preserve all costs/partials'},'runner':ref(Path(__file__)),
        'evidence_store':ref(STORE_PATH),'resolved_config':json.loads(canonical_bytes(resolved.payload)),
        'resolved_config_sha256':resolved.sha256,'source':{k:v.sha256 for k,v in _automatic_identities().items()},
        'primitive':PRIMITIVE_ID,'initialization':{'fresh':True,'seed':808,'parent':None,'all_constructor/setup_costs_charged':True},
        'methods':['muon_project_fp32','muon_plus_row_col_fp32'],
        'remaining_single_step_orders':['none','row','col','col_row'],
        'risk_basis':'new original-W matrix momentum, rectangular fc1/fc2, serial normalization and full state resume under Hadamard/dynamic scales; no recipe or source-wide Cartesian tournament',
        'recipe':'row_dynamic_max','rotation':'full_hadamard','packing':64,'logical_targets':8192,
        'main_steps':4,'resume_at':2,'arms':['U','D','R2+2-fresh-interpreter'],
        'limits':{'new_updates':28,'scored_update_bytes':28*8192,'queries':0,'protected_bytes':0,
                  'arm_process_seconds':180,'wave_process_seconds':1200},
        'tolerances':{'same_device_complete_scientific_state_mismatches':0,'codes':0,'moments':0,'loss':0},
        'storage':{'temporary_path':str(TMP),'temporary_bytes':2*2**30,'durable_path':str(DURABLE),
          'durable_bytes':192*2**20,'git_free_floor_bytes':2**30,'tmp_free_floor_bytes':2*2**30,
          'observed_n01_generations':measured,'generation_forecast_bytes':48*2**20,
          'full_generations':30,'full_generation_forecast_bytes':30*48*2**20,
          'temporary_partial_and_staging_allowance_bytes':2*2**30-30*48*2**20,
          'durable_prediction':'bounded at192MiB; actual guards enforced before each step and publication; no assumed compression savings',
          'free_git_bytes_at_freeze':shutil.disk_usage(ROOT).free,'free_tmp_bytes_at_freeze':shutil.disk_usage('/private/tmp').free,
          'preservation':'All M01 temporary full H07 generations/journals/failures remain in place. Exact manifest, tensor-container and producer bytes reconstruct from durable content-addressed archives. Per-step authenticated source→archive mappings retain original immutable tmp identities. Original tmp generations are temporarily preserved for direct resume/export, not sole final proof; no deletion or relocation authorized. Actual archive costs/bytes and full extraction proof are required before completion.'},
        'claim_limits':'four-step narrow exact replay and single-step named-order composition only; no performance/winner/core/long-run claim; scientific budgets null; protected scoring denied'}
    budget(extra_durable=192*2**20-retained(DURABLE))
    _write(DESIGN,value);print(json.dumps(ref(DESIGN)))


def design():
    d=json.loads(DESIGN.read_bytes())
    assert d['runner']==ref(Path(__file__)) and d['evidence_store']==ref(STORE_PATH)
    assert d['source']=={k:v.sha256 for k,v in _automatic_identities().items()}
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0' and torch.backends.mps.is_available()
    config=dict(d['resolved_config']);purpose=config.pop('purpose');gates=config.pop('evidence_gates')
    resolved=resolve_config(config,{'gates':gates},purpose=purpose)
    assert resolved.sha256==d['resolved_config_sha256'] and purpose=='pilot-bootstrap'
    return d,resolved


def build_metadata(d,resolved,cfg,name,arm,budget_targets):
    return {'run_id':f'm01-{name}-{arm}','created_utc':datetime.now(timezone.utc).isoformat(),
        'local_timezone':'Europe/Amsterdam','phase':'explore','method':name,'adaptation':'M01 frozen finite development',
        'source':{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  'dirty_diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest()},'producer':d['source'],'lock_sha256':cfg['runtime']['dependency_lock']['sha256'],
        'environment_sha256':cfg['runtime']['environment']['sha256'],'resolved_config':cfg,'resolved_config_sha256':resolved.sha256,
        'seeds':{'model':808,'data':808},'data_sha256':cfg['data']['manifest']['sha256'],'warm_start_sha256':None,
        'policy_sha256':None,'lineage':{'parent':None,'resume':None},
        'endpoint':{'scope':'M01 frozen finite development','purpose':'pilot-bootstrap','targets':budget_targets,'b0':None,'design':ref(DESIGN)}}

def setup(d,resolved,name,arm,budget_targets):
    torch.manual_seed(808);random.seed(809);torch.mps.manual_seed(810)
    cfg=json.loads(canonical_bytes(resolved.payload))
    manifest=cfg['data']['manifest']
    cursor=BatchCursor.from_manifest(Path(manifest['path']),manifest['sha256'],partition='target-training',seed=808,run_target_budget=budget_targets)
    initialization=torch.Generator().manual_seed(808)
    with torch.random.fork_rng():
        torch.set_rng_state(initialization.get_state())
        model=ByteTransformer(ModelConfig(**{k:cfg['model'][k] for k in ('layers','width','heads')})).to('mps')
        initialization.set_state(torch.get_rng_state())
    model.configure_quantization({'name':d['recipe'],'rotation':d['rotation']})
    method=build_matrix_method(name,dict(model.named_parameters()),default_parameter_groups(model),{})
    directory=DURABLE/'runs-v3'/name/arm
    run=RunLedger(directory,build_metadata(d,resolved,cfg,name,arm,budget_targets))
    from ralup.train import TrainingState
    return TrainingState(model,method,cursor=cursor,ledger=WorkLedger(run),microbatch=64,
        schedule={'step':0,'point':SchedulePoint(.003,0.,0.)},
        configuration={'resolved':cfg,'method':name,'development_targets':budget_targets,'seed':808},
        rng_streams={'initialization':initialization,'quantizer':torch.Generator(device='mps').manual_seed(811),
                    'meta':random.Random(812),'timing':random.Random(813)},
        process_audit=[{'kind':'launch','pid':os.getpid(),'created_utc':datetime.now(timezone.utc).isoformat()}])


def retain_state(state,directory,step):
    budget(extra_tmp=48*2**20,extra_durable=12*2**20)
    tick=time.perf_counter();snapshot=capture_snapshot(state);snapshot.milestone=True
    manifest_bytes,tensor_bytes=_manifest(snapshot)
    budget(extra_tmp=len(canonical_bytes(manifest_bytes))+len(tensor_bytes)+sum(len(v) for v in snapshot.producer_files.values())+65536)
    store.CHECK_WRITE=lambda size:budget(extra_durable=size)
    original=save_snapshot(snapshot,directory/'checkpoints')
    durable=store.archive_snapshot(snapshot,OBJECTS)
    recovered=store.recover_snapshot(durable)
    original_manifest,original_tensors=_manifest(snapshot)
    extracted_manifest,extracted_tensors=_manifest(recovered)
    assert canonical_bytes(original_manifest)==canonical_bytes(extracted_manifest)
    assert original_tensors==extracted_tensors
    mapping={'step':step,'temporary_h07':ref(original.path),'durable_archive':ref(durable.path),
             'original_manifest_bytes':len(canonical_bytes(original_manifest)),
             'original_tensor_bytes':len(original_tensors),
             'exact_manifest_sha256':hashlib.sha256(canonical_bytes(original_manifest)).hexdigest(),
             'exact_tensor_container_sha256':hashlib.sha256(original_tensors).hexdigest(),
             'complete_scientific_sha256':science(snapshot),'no_tmp_references_in_reconstruction':True,
             'temporary_disposition':'preserved; essential exact bytes also reconstruct from durable archive',
             'retained_tmp_bytes':retained(TMP),'retained_durable_bytes':retained(DURABLE)}
    _event(state,'complete_snapshot_transfer_archive_reconstruction_and_io',time.perf_counter()-tick)
    _write(directory/f'mapping-{step}.json',mapping)
    budget()
    return mapping


def child(name,arm,mode='full',reference=None):
    started=time.perf_counter();d,resolved=design()
    directory=TMP/name/arm;directory.mkdir(parents=True,exist_ok=True)
    if mode=='last':
        first=json.loads((directory/'first.json').read_bytes())
        state=restore_snapshot(load_snapshot(ArtifactRef(**reference)),'mps')
        # The boundary snapshot predates its own archive/publication. Its costs
        # are durably recorded in first.json; restore them before new work.
        state.ledger.elapsed=first['cost']['elapsed'];state.ledger.events=first['cost']['events']
        initial=first['initial'];steps=first['steps'];mappings=first['mappings']
        _event(state,'authenticated_fresh_interpreter_restore',time.perf_counter()-started)
        begin=3;finish=4
    else:
        state=setup(d,resolved,name,arm,8192 if mode=='single' else 32768)
        _event(state,'fresh_constructor_cursor_source_authentication_and_setup',time.perf_counter()-started)
        initial=science(capture_snapshot(state));steps=[];mappings=[]
        if arm=='U':mappings.append(retain_state(state,directory,0))
        begin=1;finish=1 if mode=='single' else 2 if mode=='first' else 4
    sampler=None
    try:
        with _Sampler('mps',_limits(json.loads(canonical_bytes(resolved.payload)))) as sampler:
            for step in range(begin,finish+1):
                assert not sampler.exceeded and time.perf_counter()-started<d['limits']['arm_process_seconds']
                assert (datetime.now(timezone.utc)-datetime.fromisoformat(d['chronological_origin'])).total_seconds()<d['limits']['wave_process_seconds']
                budget(extra_tmp=48*2**20,extra_durable=12*2**20)
                tick=time.perf_counter();batch=select_logical_batch(state)
                _event(state,'data_selection',time.perf_counter()-tick)
                tick=time.perf_counter();offsets=_offsets(batch);input_value=[{'inputs':list(b.inputs),'positions':list(b.positions),
                        'labels':list(b.labels),'mask':list(b.score_mask)} for b in batch.block_slices]
                inputs_ref=_write(directory/f'input-{step}.json',{'offsets':offsets,'blocks':input_value})
                _event(state,'input_context_mask_offset_retention',time.perf_counter()-tick)
                _write(directory/f'step-{step}-start.json',{'step':step,'inputs':ref(inputs_ref.path),'prior_cost':cost(state),
                         'source':d['source'],'design':ref(DESIGN),'utc':datetime.now(timezone.utc).isoformat()})
                result,seconds=_timed(state.model,lambda:train_step(state,batch))
                assert result.target_count==8192
                mapping=retain_state(state,directory,step);mappings.append(mapping)
                row={'step':step,'targets':8192,'nll_sum':result.nll_sum,'transaction_seconds':seconds,
                     'inputs':ref(inputs_ref.path),'scientific':mapping['complete_scientific_sha256'],
                     'archive':mapping['durable_archive'],'checkpoint':mapping['temporary_h07']}
                steps.append(row)
                _write(directory/f'step-{step}-complete.json',{'result':row,'cost':cost(state)})
            assert not sampler.exceeded
    except BaseException as error:
        _write(directory/f'failure-{os.getpid()}.json',{'failure':repr(error),'cost':cost(state),'elapsed_seconds':time.perf_counter()-started,
               'completed_steps':steps,'source':d['source'],'memory':None if sampler is None else memory_report(sampler.samples)})
        raise
    report={'method':name,'arm':arm,'mode':mode,'initial':initial,'steps':steps,'mappings':mappings,
        'cost':cost(state),'process_seconds_to_report':time.perf_counter()-started,
        'memory':memory_report(sampler.samples),'source':d['source'],'design':ref(DESIGN),'primitive':PRIMITIVE_ID}
    _write(directory/('first.json' if mode=='first' else'arm.json'),report)
    print(json.dumps({'report':ref(directory/('first.json' if mode=='first' else'arm.json'))}))


def cost(state):return {'stream':state.ledger.stream,'passes':state.ledger.passes,'elapsed':state.ledger.elapsed,'events':state.ledger.events}


def invoke(name,arm,mode='full',reference=None):
    command=[sys.executable,'docs/harness/m01-attempt.py',f'proxy-v3-{name}-{arm}-{mode}','docs/harness/m01-proxy.py','child',name,arm,mode]
    if reference:command.append(json.dumps(reference))
    result=subprocess.run(command,text=True,capture_output=True,timeout=180)
    _write(TMP/f'process-{name}-{arm}-{mode}.json',{'command':command,'returncode':result.returncode,
         'stdout':result.stdout,'stderr':result.stderr,'source':{k:v.sha256 for k,v in _automatic_identities().items()}})
    assert result.returncode==0,(result.stdout,result.stderr)
    return json.loads((TMP/name/arm/('first.json' if mode=='first' else'arm.json')).read_bytes())


def wave():
    started=time.perf_counter();d,_=design();TMP.mkdir(parents=True,exist_ok=True)
    comparisons=[];reports=[]
    for name in d['methods']:
        u=invoke(name,'U');duplicate=invoke(name,'D')
        first=invoke(name,'R','first');resumed=invoke(name,'R','last',first['steps'][-1]['checkpoint'])
        for label,other in [('duplicate',duplicate),('fresh_interpreter_resume',resumed)]:
            assert u['initial']==other['initial']
            for left,right in zip(u['steps'],other['steps'],strict=True):
                assert left['scientific']==right['scientific'] and left['nll_sum']==right['nll_sum']
                assert left['inputs']['sha256']==right['inputs']['sha256']
            comparisons.append({'method':name,'comparison':label,'all_four_step_complete_scientific_owners':'exact',
                                'loss_inputs_codes_moments':'exact'})
        reports += [ref(TMP/name/'U'/'arm.json'),ref(TMP/name/'D'/'arm.json'),ref(TMP/name/'R'/'first.json'),ref(TMP/name/'R'/'arm.json')]
    for order in d['remaining_single_step_orders']:
        name=f'muon_plus_{order}_fp32';invoke(name,'single','single')
        reports.append(ref(TMP/name/'single'/'arm.json'))
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(d['chronological_origin'])).total_seconds()<d['limits']['wave_process_seconds']
    # Preserve small raw process/cost/input/mapping journals durably. No mutable
    # references are rewritten: mapping originals remain original identities.
    transfers=[]
    for path in sorted(TMP.rglob('*.json')):
        if 'checkpoints' in path.parts:continue
        destination=DURABLE/'proxy-journals-v3'/path.relative_to(TMP)
        raw=path.read_bytes();destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        assert destination.read_bytes()==raw
        transfers.append({'original':ref(path),'durable':ref(destination),'bytes':len(raw),'disposition':'exact copy; original preserved'})
    report={'status':'pass','design':ref(DESIGN),'comparisons':comparisons,'temporary_reports':reports,
            'journal_mappings':transfers,'new_updates':28,'scored_update_bytes':28*8192,'queries':0,'protected_bytes':0,
            'recovery_process_seconds':time.perf_counter()-started,
            'chronological_origin':d['chronological_origin'],
            'chronological_wave_seconds':(datetime.now(timezone.utc)-datetime.fromisoformat(d['chronological_origin'])).total_seconds(),'retained_tmp_bytes':retained(TMP),
            'retained_durable_bytes':retained(DURABLE),'free_git_bytes':shutil.disk_usage(ROOT).free,
            'scope':d['claim_limits'],'source':d['source'],'primitive':PRIMITIVE_ID}
    budget();_write(DURABLE/'proxy-wave-v3.json',report);print(json.dumps(ref(DURABLE/'proxy-wave-v3.json')))


def audit():
    # Authenticate all complete mandatory snapshots without ever reading the
    # original temporary checkpoint payloads; retain immutable original refs.
    started=time.perf_counter();rows=[]
    original_read=Path.read_bytes
    def reject_original(self):
        if self.resolve().is_relative_to(TMP):raise AssertionError('Archive audit read original temporary payload')
        return original_read(self)
    Path.read_bytes=reject_original
    try:
        for path in sorted((DURABLE/'proxy-journals-v3').rglob('mapping-*.json')):
            mapping=json.loads(path.read_bytes())
            snapshot=store.recover_snapshot(ArtifactRef(**mapping['durable_archive']))
            manifest,tensors=_manifest(snapshot)
            assert hashlib.sha256(canonical_bytes(manifest)).hexdigest()==mapping['exact_manifest_sha256']
            assert hashlib.sha256(tensors).hexdigest()==mapping['exact_tensor_container_sha256']
            assert science(snapshot)==mapping['complete_scientific_sha256']
            rows.append({'mapping':ref(path),'archive':mapping['durable_archive'],
                         'original':mapping['temporary_h07'],'all_manifest_tensor_source_context_bytes':'exact'})
    finally:Path.read_bytes=original_read
    assert len(rows)==30
    report={'status':'pass','scope':'all30 mandatory complete snapshot generations recovered with original temporary Path reads explicitly denied',
        'snapshots':rows,'project_seconds':time.perf_counter()-started,'retained_durable_bytes':retained(DURABLE),
        'retained_tmp_bytes':retained(TMP),'free_git_bytes':shutil.disk_usage(ROOT).free,
        'reconstruction':'original metadata/source identities remain unchanged; permanent refs external to temp authenticate their original paths; operational relocation/source compatibility is not inferred',
        'design':ref(DESIGN),'runner':ref(Path(__file__))}
    budget();_write(DURABLE/'archive-audit-v3.json',report);print(json.dumps(ref(DURABLE/'archive-audit-v3.json')))


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='child':child(sys.argv[2],sys.argv[3],sys.argv[4],json.loads(sys.argv[5]) if len(sys.argv)>5 else None)
    elif sys.argv[1]=='wave':wave()
    elif sys.argv[1]=='audit':audit()
    else:raise ValueError('Unknown mode')
