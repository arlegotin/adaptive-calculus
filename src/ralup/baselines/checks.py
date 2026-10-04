"""Bounded synthetic scalar AdamW comparisons, never scientific evidence."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
import torch.optim.adam as torch_adam

from ..config import canonical_bytes
from ..optim.adamw import AdamWUpdateRule
from ..optim.base import ParameterGroup, SchedulePoint
from ..records import ArtifactRef, sha256_file
from .registry import default_registry

# Declared before target-device measurement. Same-device oracle must be exact;
# cross-device FP32 comparison permits only these bounded fixture errors.
TOLERANCES = {'cross_device_atol':2e-6,'cross_device_rtol':2e-6,'same_device_mismatches':0}
CASES = [((.9,.99),1e-8,0.),((.9,.95),1e-8,.1),((.95,.999),1e-4,.03),((0.,0.),.01,.2)]


def _compare(device):
    rows = []
    for betas,eps,decay in CASES:
        initial = torch.tensor([[-.31,0.,1.7],[.001,-4.2,.73]],dtype=torch.float32)
        weights = {name:torch.nn.Parameter(initial.clone().to(target)) for name,target in [('adapter',device),('oracle',device),('cpu','cpu')]}
        oracle = torch.optim.AdamW([weights['oracle']],lr=.001,betas=betas,eps=eps,weight_decay=decay,foreach=False,fused=False)
        cpu = torch.optim.AdamW([weights['cpu']],lr=.001,betas=betas,eps=eps,weight_decay=decay,foreach=False,fused=False)
        rule,state = AdamWUpdateRule(betas=betas,eps=eps),{}
        max_error,mismatches,state_mismatches = 0.,0,0
        for index in range(40):
            grad = torch.zeros_like(initial) if index%5 == 0 else torch.tensor([[.13,-.7,0.],[.002,1.9,-.03]])*((index%3)-1)
            lr = .001 if index<20 else .00037
            proposal = rule.propose(weights['adapter'],grad.to(device),state,SchedulePoint(lr,decay,index/40))
            for optimizer,key in [(oracle,'oracle'),(cpu,'cpu')]:
                optimizer.param_groups[0]['lr'] = lr
                weights[key].grad = grad.to(weights[key].device)
                optimizer.step()
            rule.commit(weights['adapter'],proposal)
            state = proposal.new_optimizer_state
            mismatches += int((weights['adapter'] != weights['oracle']).sum())
            state_mismatches += sum(int((value != oracle.state[weights['oracle']][name]).sum()) for name,value in state.items())
            actual,expected = weights['adapter'].detach().cpu(),weights['cpu'].detach()
            max_error = max(max_error,float((actual-expected).abs().max()))
            torch.testing.assert_close(actual,expected,atol=TOLERANCES['cross_device_atol'],rtol=TOLERANCES['cross_device_rtol'])
            for name,value in state.items():
                torch.testing.assert_close(value.cpu(),cpu.state[weights['cpu']][name],atol=TOLERANCES['cross_device_atol'],rtol=TOLERANCES['cross_device_rtol'])
        assert mismatches == state_mismatches == 0, (mismatches,state_mismatches)
        rows.append({'betas':list(betas),'eps':eps,'decay':decay,'steps':40,'zero_gradient_steps':8,
                     'parameter_mismatches':mismatches,'state_mismatches':state_mismatches,'cross_device_weight_max_error':max_error})
    return {'same_device_bitwise':True,'cases':len(rows),'fixtures':rows,'cross_device_comparison':'frozen FP32 tolerance; no general device-equivalence claim'}


def check_adamw(device: str, output: Path) -> ArtifactRef:
    if device not in ('cpu','mps'):
        raise ValueError('AdamW checks require cpu or mps')
    root = Path(__file__).resolve().parents[3]
    source_paths = ['src/ralup/optim/base.py','src/ralup/optim/adamw.py','src/ralup/baselines/registry.py','src/ralup/baselines/checks.py','src/ralup/__main__.py']
    scalar = inspect.getsource(torch_adam._single_tensor_adam)
    report = {'schema_version':1,'suite':'adamw','device':device,'dtype':'float32',
              'recorded_utc':datetime.now(timezone.utc).isoformat(),'fallback_environment':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','unset'),
              'scope':'H05 bounded fixed-gradient engineering fixture; no scientific/test-data/performance claim',
              'scientific_gate':None,'tolerances':dict(TOLERANCES),'blockers':[],
              'upstream':{'torch_version':str(torch.__version__),'torch_git_revision':torch.version.git_version,
                          'scalar_function':'torch.optim.adam._single_tensor_adam','scalar_source_sha256':hashlib.sha256(scalar.encode()).hexdigest(),
                          'settings':{'foreach':False,'fused':False,'amsgrad':False,'maximize':False,'capturable':False,'differentiable':False,'decoupled_weight_decay':True}},
              'source_hashes':{path:sha256_file(root/path) for path in source_paths}}
    if torch.__version__ != '2.14.1':
        report.update(status='blocked',blockers=['Pinned scalar oracle requires torch==2.14.1'])
    elif device == 'mps' and not torch.backends.mps.is_available():
        report.update(status='blocked',blockers=['Actual MPS unavailable; sandbox execution is not target-device evidence'])
    elif device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        report.update(status='blocked',blockers=['MPS comparison requires fallback disabled before Torch startup'])
    else:
        try:
            report['comparison'] = _compare(device)
            report['status'] = 'pass'
        except (AssertionError,RuntimeError,ValueError) as error:
            report.update(status='fail',blockers=[f'{type(error).__name__}: {error}'])
    registry = default_registry((ParameterGroup('fixture_original_W',('weight',)),),parity_status='pass' if report['status']=='pass' else report['status'],operations_status='pass' if report['status']=='pass' else 'blocked')
    method = registry.require('adamw_scalar_fp32',adaptation_label='original_W_scalar_fixture')
    report['method'] = json.loads(json.dumps(asdict(method)))
    report['claim_eligibility'] = registry.claim_eligibility(method.name)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(canonical_bytes(report)+b'\n')
    return ArtifactRef(output,sha256_file(output))
