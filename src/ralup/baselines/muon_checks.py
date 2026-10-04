"""Declared finite equation fixtures; no release or scientific comparison claim."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import math
import json
import os
from pathlib import Path

import torch

from ..config import canonical_bytes
from ..optim.base import ParameterGroup, SchedulePoint
from ..optim.muon import orthogonalize_five_step
from ..records import ArtifactRef, sha256_file
from .registry import MATRIX_NAMES, MethodRegistry, matrix_record, matrix_rule

# Frozen before device fixtures. CPU64 literal tests use 2e-13. FP32 cross-device
# polynomial gate is separate from H04 neural tolerance and exact replay gates.
TOLERANCES = {'polynomial_atol':2e-5,'polynomial_rtol':2e-5,'cpu64_literal_atol':2e-13,
              'same_device_displacement_mismatches':0,'same_device_momentum_mismatches':0}
LITERAL = [[-.6806609899651347,.825539610365519],[.7412954177537816,.25944002690644696]]


def reference_polynomial(x):
    tall = x.shape[0] > x.shape[1]
    working = x.T if tall else x
    working = working / (working.norm() + 1e-7)
    for iteration in range(5):
        gram = working @ working.T
        working = 3.4445*working - 4.775*(gram@working) + 2.0315*(gram@gram@working)
    return working.T if tall else working


def reference_normalization(x, order):
    for stage in ([] if order == 'none' else order.split('_')):
        dimension = 1 if stage == 'row' else 0
        x = x / (x.square().sum(dim=dimension,keepdim=True)+1e-8).sqrt()
    return x


def check_muon(device, output):
    if device not in ('cpu','mps'): raise ValueError('Muon checks require cpu or mps')
    root=Path(__file__).resolve().parents[3]
    report={'schema_version':1,'suite':'muon','device':device,'dtype':'float32',
        'recorded_utc':datetime.now(timezone.utc).isoformat(),'scientific_gate':None,
        'scope':'M01 independent equation fixtures; portable FP32 departures; no source release/parity or winner claim',
        'fallback_environment':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        'source_hashes':{str(p.relative_to(root)):sha256_file(p) for p in sorted(root.glob('src/ralup/**/*.py'))},
        'tolerances':dict(TOLERANCES),'status':'pass','blockers':[],'variants':[]}
    if device == 'mps' and (not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0'):
        report.update(status='blocked',blockers=['Actual MPS requires availability and fallback disabled at startup'])
    else:
        try:
            literal=orthogonalize_five_step(torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64))
            torch.testing.assert_close(literal,torch.tensor(LITERAL,dtype=torch.float64),atol=2e-13,rtol=2e-13)
            report['cpu64_literal_polynomial']=literal.tolist()
            for name in MATRIX_NAMES:
                rule=matrix_rule(name,{})
                record=replace(matrix_record(name,(ParameterGroup('internal_projection',('w',)),),rule,
                       parity_status='pass',operations_status='pass'),artifact_kind='original_fixture')
                registry=MethodRegistry();registry.register(record)
                rows=[]
                for shape,zero in [((2,2),False),((3,2),False),((2,3),False),((3,2),True)]:
                    g=torch.zeros(shape,device=device) if zero else torch.arange(1,math.prod(shape)+1,device=device,dtype=torch.float32).reshape(shape)/9
                    w=torch.ones(shape,device=device);m=torch.zeros_like(w)
                    for step in range(2):
                        old={'momentum':m}
                        proposal=rule.propose(w,g,old,SchedulePoint(.003,.02,step/2))
                        beta=rule.configuration['momentum']
                        if name == 'muon_project_fp32':
                            m=beta*m+g
                            polar_input=g+beta*m if rule.configuration['nesterov'] else m
                            scale=math.sqrt(max(1,shape[0]/shape[1]))
                        else:
                            m=(beta*m).add_(g,alpha=1-beta);polar_input=m
                            scale=math.sqrt(shape[0]/shape[1])
                        expected=reference_polynomial(polar_input)
                        cpu=reference_polynomial(polar_input.cpu())
                        torch.testing.assert_close(orthogonalize_five_step(polar_input).cpu(),cpu,atol=2e-5,rtol=2e-5)
                        if name != 'muon_project_fp32':expected=reference_normalization(expected,rule.configuration['normalization'])
                        displacement=-.003*scale*expected
                        torch.testing.assert_close(proposal.gradient_displacement,displacement,atol=0,rtol=0)
                        torch.testing.assert_close(proposal.new_optimizer_state['momentum'],m,atol=0,rtol=0)
                        rule.commit(w,proposal)
                        rows.append({'shape':list(shape),'zero':zero,'step':step+1,
                                     'input':g.cpu().tolist(),'momentum':m.cpu().tolist(),
                                     'polynomial':orthogonalize_five_step(polar_input).cpu().tolist(),
                                     'displacement':proposal.gradient_displacement.cpu().tolist(),'weight':w.cpu().tolist(),
                                     'cpu_polynomial_max_error':float((orthogonalize_five_step(polar_input).cpu()-cpu).abs().max())})
                report['variants'].append({'method':json.loads(json.dumps(asdict(record))),'fixtures':rows,'claim_eligibility':registry.claim_eligibility(name)})
        except (AssertionError,RuntimeError,ValueError) as error:
            report.update(status='fail',blockers=[f'{type(error).__name__}: {error}'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(canonical_bytes(report)+b'\n')
    return ArtifactRef(output,sha256_file(output))
