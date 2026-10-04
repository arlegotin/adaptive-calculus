"""Root-revised seven-step direct capacity observation with durable journaling.

64 is provisional from exact replay, never selected from the lost search.
No product source changes, geometric rerun, thermal or sustained claims.
"""
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import uuid
from datetime import datetime,timezone

DESIGN=Path('docs/harness/n01-capacity-recovery.json')
ROOT=Path('artifacts/calibration/n01-v1')
OUTPUT=ROOT/'capacity-recovery'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):return {'path':str(path),'sha256':digest(path)}


class Journal:
    """A completed action must publish/fsync its record before another starts."""
    def __init__(self,path):
        self.path=path;path.mkdir(parents=True,exist_ok=False);self.entries=[]
    def record(self,kind,payload):
        raw=(json.dumps({'kind':kind,'utc':datetime.now(timezone.utc).isoformat(),**payload},sort_keys=True,separators=(',',':'))+'\n').encode()
        if self.path.is_relative_to(ROOT):
            assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())+len(raw)<=2*2**30
            assert shutil.disk_usage(ROOT).free>=2**30+len(raw)
        target=self.path/f'{len(self.entries):03d}-{kind}.json'
        with target.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        result=ref(target);self.entries.append(result);return result
    def operation(self,label,action,failed_state=lambda:{}):
        self.record('operation-started',{'operation':label});start=time.perf_counter()
        try:value=action()
        except Exception as error:
            self.record('operation-failed',{'operation':label,'seconds':time.perf_counter()-start,'error_type':type(error).__name__,'error':str(error),'state':failed_state()})
            raise
        self.record('operation-completed',{'operation':label,'seconds':time.perf_counter()-start,'result':value})
        return value


def core():
    spec=importlib.util.spec_from_file_location('n01_frozen_core','docs/harness/n01-verification.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def preflight():
    # Actual persistence/failure behavior, no tensor or device operations.
    path=Path('/private/tmp')/('n01-capacity-journal-preflight-'+uuid.uuid4().hex)
    journal=Journal(path)
    first=journal.operation('first',lambda:{'completed_steps':1,'observed_seconds':.25})
    def fail():raise RuntimeError('injected preflight failure')
    try:journal.operation('second',fail,lambda:{'completed_steps':0})
    except RuntimeError:pass
    else:raise AssertionError('Failure must stop the sequence')
    records=[json.loads(Path(r['path']).read_bytes()) for r in journal.entries]
    assert first['completed_steps']==1
    assert [r['kind'] for r in records]==['operation-started','operation-completed','operation-started','operation-failed']
    assert records[1]['result']['completed_steps']==1 and records[-1]['state']['completed_steps']==0
    # Journal must preserve exact data and exclusively published records.
    assert all(digest(Path(r['path']))==r['sha256'] for r in journal.entries)
    print(json.dumps({'status':'PASS no GPU steps','journal':str(path),'entries':journal.entries}))


def freeze():
    c=core();original,resolved=c.design()
    initial=json.loads((ROOT/'capacity-search/initial/latest.json').read_bytes())
    path=Path(initial['path']);assert digest(path)==initial['sha256']
    frozen={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'authority':'root CORRECTED FINAL capacity ruling; original failed capacity attempt not relabeled; revised prospective engineering allowance',
            'runner':ref(Path(__file__)),'primary_design':ref(Path('docs/harness/n01-design.json')),'producer':original['producer'],'parent':original['parent'],
            'failed_attempt_source':original['runner'],'failed_process_log':ref(Path('artifacts/checks/n01-capacity.txt')),'failed_process_real_seconds':102.71,
            'lost_results':'exact search counts/timings/selection unavailable; no recorded evidence proves noOOM. Never call guessed55/7 counters actual.',
            'failed_cost_bounds':{'completed_step_lower':1,'completed_step_upper':55,'partial_failed_attempt_upper':7,'partial_attempt_target_upper_each':8192,'query_bytes':8192,
                                  'source_rationale':'normal search<=8sizes*3recipes*2steps=48; optional OOMcommon-packing remeasurement<=3recipes*2=6 before stop; warmup1 ->55. At most128→64→32→16→8→4→2→1 seven partial failing retries; successful common remeasurement cannot reduce packing again or process wouldfail before sustained-size validation.'},
            'initial_current_source_snapshot':ref(path),'sizes':[1,64],'recipes':original['recipes'],'steps_per_arm':1,'warmup_steps':1,'query_bytes':8192,
            'new_complete_steps':7,'completed_total_upper':62,'original_completed_step_allowance':64,'full_step_equivalent_total_upper_including_lost_partials':69,
            'engineering_allowance_revision':'69 full-step-equivalentUPPER expressly includes lostpartialwork; original64 not claimed to cover unknown failedattempts. No unused margin/further probes authorized.',
            'packing':'64 PROVISIONAL from numerically validated exact replay scope;1 reference only; no lost/new geometric selection or optimum claim',
            'randomization':'random.Random(808) shuffles six1-step recipe×size arms before outcomes, all strict restored sameinitialstate/cursor/RNG',
            'persistence':'fsynced start+completed/failure records peroperation before next; every arm contains all exact workevents/counts/elapsed/inputs/memory/inventory; first failure stops probe, no retry/newarm',
            'query':'single8192-byte forward evaluation from cloned initialcursor; distinctquerypass, additional_loss/query/loss_targets classify same pass, no backward/update',
            'limits':'same2GiBtasknew retained+scratch and1GiBfree; sampler originalcaps checked before/after every stage; finite180-second recoveryclock plus102.71 failed clock reported, notreset',
            'storage_forecast_bytes':80*2**20,'scope_limit':'single-step arm observation and separately retained observer-interleaved/coldstartup72study timings; no sustained/thermal/confidenceclaim. M07 remeasures longer/permethod/core forecasts and scientific budgets.',
            'preflight_log':ref(Path('artifacts/checks/n01-capacity-preflight-final.txt'))}
    c.publish(DESIGN,frozen);print(json.dumps(ref(DESIGN)))


def run():
    import torch
    from ralup.calibrate import _Sampler,_timed,_inventory,_offsets,memory_report
    from ralup.data.batches import next_logical_batch
    from ralup.data.cursor import BatchCursor
    from ralup.evaluate import score_blocks
    from ralup.records import ArtifactRef,WorkLedger
    from ralup.state import _validated,capture_snapshot,load_snapshot,restore_snapshot
    from ralup.train import select_logical_batch,train_step
    start=time.perf_counter();c=core();original,resolved=c.design()
    frozen=json.loads(DESIGN.read_bytes());assert ref(Path(__file__))==frozen['runner']
    assert original['producer']==frozen['producer']
    c.budget(frozen['storage_forecast_bytes'])
    journal=Journal(OUTPUT/'journal')
    journal.record('setup',{'design':ref(DESIGN),'source':original['producer'],'parent':original['parent'],'failed_attempt':frozen['failed_process_log']})
    snapshot=load_snapshot(ArtifactRef(**frozen['initial_current_source_snapshot']));payload=_validated(snapshot)
    assert {k:v.sha256 for k,v in payload['identities']['automatic'].items()}==original['producer']
    journal.record('authenticated-initial',{'snapshot':frozen['initial_current_source_snapshot'],'initial_inventory':_inventory(snapshot),'setup_seconds':time.perf_counter()-start})
    current=None
    def failed_state():
        if current is None:return {}
        return {'stream':current.ledger.stream,'passes':current.ledger.passes,'events':current.ledger.events,'elapsed':current.ledger.elapsed,'failed':current.failed}
    def transaction(size,recipe,label):
        nonlocal current
        sample_begin=len(sampler.samples)
        region=time.perf_counter();current=restore_snapshot(snapshot,'mps');reset_seconds=time.perf_counter()-region
        current.ledger=WorkLedger();current.microbatch=size
        current.model.configure_quantization({**current.model._recipe,'name':recipe})
        batch=select_logical_batch(current)
        result,duration=_timed(current.model,lambda:train_step(current,batch))
        assert result.target_count==8192 and current.ledger.stream['updates']==1
        region=time.perf_counter();inventory=_inventory(capture_snapshot(current));inventory_seconds=time.perf_counter()-region
        sampler.sample();memory=memory_report(sampler.samples[sample_begin:])
        return {'label':label,'size':size,'recipe':recipe,'completed_steps':1,'full_step_seconds':duration,'reset_setup_seconds':reset_seconds,
                'actual_block_duration_seconds':duration,'snapshot_inventory_seconds':inventory_seconds,'memory':memory,'input_offsets':_offsets(batch),'nll_sum':result.nll_sum,'inventory':inventory,
                'cost':{'stream':current.ledger.stream,'passes':current.ledger.passes,'elapsed':current.ledger.elapsed,'events':current.ledger.events}}
    order=[(size,recipe) for size in frozen['sizes'] for recipe in frozen['recipes']];random.Random(808).shuffle(order)
    journal.record('frozen-arm-order',{'order':order})
    arms=[]
    with _Sampler('mps',original['caps']) as sampler:
        assert not sampler.exceeded
        warmup=journal.operation('one-warmup',lambda:transaction(64,'row_fixed_max','warmup'),failed_state)
        assert not sampler.exceeded
        for size,recipe in order:
            assert not sampler.exceeded
            assert time.perf_counter()-start<180
            arm=journal.operation(f'{recipe}-{size}',lambda:transaction(size,recipe,'direct-single-step'),failed_state)
            arms.append(arm)
            assert not sampler.exceeded
        def query():
            nonlocal current
            region=time.perf_counter();current=restore_snapshot(snapshot,'mps');current.ledger=WorkLedger()
            cursor=BatchCursor(current.cursor.state);batch=next_logical_batch(cursor,8192)
            setup=time.perf_counter()-region
            result,seconds=_timed(current.model,lambda:score_blocks(current.model,batch.block_slices,evaluation_context={'partition':'target-training','seed':808}))
            assert result.target_count==8192
            return {'event_id':'n01-capacity-recovery-single-query','distinct_passes':1,'query':8192,'additional_loss':8192,'loss_targets':8192,'backward':0,'new_optimizer_updates':0,
                    'forward_context':sum(len(b.inputs) for b in batch.block_slices),'seconds':seconds,'setup_seconds':setup,'input_offsets':_offsets(batch),'nll_sum':result.nll_sum}
        query_result=journal.operation('single8192-query',query,failed_state)
        assert not sampler.exceeded
    report={'design':ref(DESIGN),'source':original['producer'],'warmup':warmup,'arms':arms,'query':query_result,'journal':journal.entries,
            'new_actual_optimizer_steps':7,'new_actual_query_bytes':8192,'provisional_packing':64,'selection':'none; validated replay packing64 provisional,1reference',
            'memory':memory_report(sampler.samples),'project_seconds':time.perf_counter()-start,'failed_attempt_expenditure':frozen['failed_cost_bounds'],
            'completed_upper_including_failed_attempt':62,'full_step_equivalent_upper_including_failed_partial_attempts':69,
            'limits':'one step per arm; no sustained/thermal/optimality confidence; everyarm exact costrecord persisted before next'}
    c.publish(OUTPUT/'report.json',report);print(json.dumps({'report':ref(OUTPUT/'report.json'),'new_actual_steps':7,'provisional_packing':64}))


if __name__=='__main__':
    if sys.argv[1]=='preflight':preflight()
    elif sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='run':run()
