"""Audit already saved quantization inputs; never run a model/quantizer again."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import struct
import time

START=time.perf_counter()
spec=importlib.util.spec_from_file_location('contract','docs/harness/h08-mps-contract.py');contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
from safetensors.torch import save
from ralup.quantization.formats import E2M1_THRESHOLDS,E2M1_VALUES


def run():
    directory=contract.OUTPUT/'forensic-retention-v1'
    report=json.loads((directory/'observer.json').read_bytes())
    case='11-target_train-quest_fp4_row_reference-64'
    saved={}
    for entry in report['entries']:
        if entry['case']['id']==case and entry['step']==8:saved[entry['arm']]=(entry,*contract.read_science(entry))
    pairs=[(a,b) for a,b in itertools.combinations(('D0','D1','D2','D3'),2)]+[('U0','U1'),('U0','R')]
    rows=[];raw={};identical=0;count=0
    for first,second in pairs:
        ar,a,ma=saved[first];br,b,mb=saved[second]
        names=[key[len('state.histories.'):-len('.cached_codes')] for key in a if key.startswith('state.histories.') and key.endswith('.cached_codes')]
        for name in names:
            for stage in ('pre_forward','cached_endpoint'):
                base='state.model_cache._contexts.'+name+'.'
                code=base+'codes' if stage=='pre_forward' else 'state.histories.'+name+'.cached_codes'
                shadow=base+'storage_u' if stage=='pre_forward' else 'state.weights.'+name
                scale=base+'scales' if stage=='pre_forward' else 'state.histories.'+name+'.scales'
                points=(a[code]!=b[code]).nonzero().tolist()
                for row,col in points:
                    ua,ub=a[shadow][row,col],b[shadow][row,col]
                    sa,sb=a[scale][row,0],b[scale][row,0]
                    same_input=ua.view(__import__('torch').int32).item()==ub.view(__import__('torch').int32).item() and sa.view(__import__('torch').int32).item()==sb.view(__import__('torch').int32).item()
                    identical+=int(same_input);count+=1
                    ca,cb=int(a[code][row,col]),int(b[code][row,col])
                    ratios=[float(u)/float(s) for u,s in ((ua,sa),(ub,sb))]
                    rounded=[struct.unpack('f',struct.pack('f',value))[0] for value in ratios]
                    key=f'{first}-{second}.{stage}.{name}.{row}.{col}'
                    raw[key]=__import__('torch').stack((ua,ub,sa,sb)).contiguous()
                    rows.append({'pair':[first,second],'owner':name,'stage':stage,'completed_step':8,'context_step':ma['fields']['state.model_cache._context_step'],
                        'coordinate':[row,col],'raw_input_key':key,'codes':[ca,cb],'levels':[E2M1_VALUES[ca],E2M1_VALUES[cb]],
                        'neighbor_midpoints':list(E2M1_THRESHOLDS[min(ca,cb):max(ca,cb)]),'original_numeric_inputs_bitwise_equal':same_input,
                        'derived_double_ratios':ratios,'derived_nearest_fp32_ratios':rounded,
                        'derived_scope':'read-only arithmetic from saved normal FP32 operands; not a rerun of MPS division/quantization or a model operation'})
    destination=contract.OUTPUT/case
    data=save(raw);contract.budget(len(data));path=destination/'paired-quantization-input-witnesses.safetensors';path.write_bytes(data)
    ref=contract._write(destination/'quantization-input-audit.json',{'schema_version':1,'case':case,'observer':contract.ref(directory/'observer.json'),
        'audit_script':contract.ref(Path(__file__)),'sources':[value[0] for value in saved.values()],
        'rows':rows,'changed_code_coordinates_across_all_pairs_and_stages':count,'identical_numeric_input_changed_code_count':identical,
        'raw_paired_inputs':contract.ref(path),'audit_seconds':time.perf_counter()-START,
        'scope':'saved-input provenance/causal audit only; exact-code/ceiling scope remains failed; no threshold change/no new forward/backward/quantizer execution'})
    print(json.dumps({'path':str(ref.path),'sha256':ref.sha256,'changed_points':count,'identical_input_changed_codes':identical}))


if __name__=='__main__':run()
