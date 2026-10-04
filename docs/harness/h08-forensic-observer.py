"""Read-only forensic retention supplement; no new model work or producer edits."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROCESS_START=time.perf_counter()
spec=importlib.util.spec_from_file_location('h08_contract','docs/harness/h08-mps-contract.py')
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
from safetensors.torch import save
from ralup.quantization.formats import E2M1_VALUES,E2M1_THRESHOLDS

ROOT=contract.OUTPUT
DEST=ROOT/'forensic-retention-v1'
PARENT=int(sys.argv[1])
seen=set();entries=[];io_seconds=0.;holds=0


def children():
    text=subprocess.run(['ps','-axo','pid=,ppid=,stat='],capture_output=True,text=True,check=True).stdout
    return [(int(parts[0]),parts[2]) for line in text.splitlines() if len(parts:=line.split())>=3 and int(parts[1])==PARENT]


def harvest():
    global io_seconds
    reports=list(ROOT.glob('*/**/arm.json'))+list(ROOT.glob('*/R/replay-first.json'))
    for report_path in reports:
        try:report=json.loads(report_path.read_bytes())
        except (FileNotFoundError,json.JSONDecodeError):continue
        for step,value in report['states'].items():
            key=(report['case']['id'],report['arm'],step,value['sha256'])
            if key in seen:continue
            path=Path(value['path'])
            if not path.exists():continue  # Prior authorized disposals explicitly unavailable.
            region=time.perf_counter()
            arrays,scalars=contract.read_science(value)
            names=[key[len('state.histories.'):-len('.cached_codes')] for key in arrays if key.startswith('state.histories.') and key.endswith('.cached_codes')]
            selected={}
            for name in names:
                for owner in ('state.weights.'+name,
                    'state.model_cache._contexts.'+name+'.storage_u',
                    'state.model_cache._contexts.'+name+'.storage_values',
                    'state.model_cache._contexts.'+name+'.scales',
                    'state.model_cache._contexts.'+name+'.codes',
                    'state.histories.'+name+'.cached_codes',
                    'state.histories.'+name+'.scales'):
                    selected[owner]=arrays[owner]
            fields={k:v for k,v in scalars.items() if k.startswith('state.model_cache._contexts.') or k in ('state.schedule.step','state.model_cache._context_step','state.microbatch')}
            metadata={'fields':fields,'case':report['case'],'arm':report['arm'],'source_state':value,'source_arm_report':contract.ref(report_path),
                'input_trace':report['steps'],'producer':report['source'],'parent':report['parent'],
                'context_stage':'_contexts holds PRE-FORWARD U/Q/scales/codes at schedule.step-1; weights is POST-COMMIT W at schedule.step; cached_codes is endpoint telemetry Q(W_post on cached pre-forward scale), not active forward codes',
                'rotation':'identity per frozen config; endpoint storage_u equals saved postcommit W',
                'neighbor_values':list(E2M1_VALUES),'normalized_neighbor_midpoints':list(E2M1_THRESHOLDS),
                'numeric_design_unchanged':True}
            raw=contract.encoded(selected,metadata)
            contract.budget(len(raw))
            target=DEST/(report['case']['id']+'-'+report['arm']+'-step'+step+'.safetensors.gz')
            with gzip.open(target,'xb',compresslevel=1) as stream:stream.write(raw)
            entry={**contract.ref(target),'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'source_state':value,'case':report['case'],'arm':report['arm'],'step':int(step),'owners':list(selected),'raw_bytes':len(raw)}
            entries.append(entry);seen.add(key)
            # Publish authenticated retention receipt before allowing controller disposal.
            contract._write(DEST/(target.name+'.receipt.json'),entry)
            io_seconds+=time.perf_counter()-region


def run():
    global holds
    os.kill(PARENT,signal.SIGSTOP);holds+=1
    DEST.mkdir(exist_ok=False)
    contract._write(DEST/'supplement.json',{'schema_version':1,'date_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'authority':'root approved versioned forensic retention supplement; no extra model execution/source edit/bound change',
        'original_design':contract.ref(ROOT/'design.json'),'observer':contract.ref(Path(__file__)),'controller_pid':PARENT,
        'hold_contract':'controller alone stopped while existing child completes; authenticate/receipt all available completed child owner arrays before controller may advance/discard; children run unchanged. 0.2s controller windows launch next children; fixed eight-step workers require measured seconds, and controller is stopped again during their wait.',
        'preservation':'paired W_post, PRE-FORWARD storage_u/storage_values/scales/codes, cached endpoint code/scale, producing step/input/source/parent identities and midpoint constants, for every available retained scientific owner state',
        'limits':{'new_retained_and_scratch_bytes':2*2**30,'available_before_write_bytes':2**30},
        'historical_limit':'case03 opposite-side local W at code flat122 already unavailable after previously approved disposal; no reconstruction claimed exact; original numerical scope stays failed',
        'observer_cost':'separate read/transfer/compression/publication work and wall; wave wall includes controller hold, avoid summing overlapping clocks'})
    try:
        while not (ROOT/'wave.json').exists():
            harvest()
            active=[(pid,stat) for pid,stat in children() if not stat.startswith('Z')]
            if active:
                time.sleep(.1)
                continue
            harvest()  # Reap-boundary receipt precedes any next disposal.
            os.kill(PARENT,signal.SIGCONT)
            time.sleep(.2)
            try:os.kill(PARENT,signal.SIGSTOP);holds+=1
            except ProcessLookupError:break
        harvest()
        report={'supplement':contract.ref(DEST/'supplement.json'),'entries':entries,'read_transfer_compression_publication_seconds':io_seconds,
                'observer_wall_seconds':time.perf_counter()-PROCESS_START,'controller_hold_count':holds,
                'retained_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
                'no_new_forward_backward':True,'numeric_design_unchanged':True,'original_missing_context_not_reconstructed':True}
        ref=contract._write(DEST/'observer.json',report)
        print(json.dumps({'path':str(ref.path),'sha256':ref.sha256}))
    finally:
        try:os.kill(PARENT,signal.SIGCONT)
        except ProcessLookupError:pass


if __name__=='__main__':run()
