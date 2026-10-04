"""M02 finite pinned SOAP and independent scalar Madam mechanics fixtures."""
from dataclasses import asdict,replace
from datetime import datetime,timezone
import json,math,os
from pathlib import Path
import torch
from ..config import canonical_bytes
from ..records import ArtifactRef,sha256_file
from ..optim.base import ParameterGroup,SchedulePoint
from .references.soap_reference import SOAP
from .registry import SOAP_MADAM_NAMES,MethodRegistry,matrix_record,matrix_rule

SOURCE_SHA256='9e9021335ad02371584622fa256e73fe75f3eb74446a8d8a216e956534148f6b'
TOLERANCES={'cpu_reference_mismatches':0,'cpu_mps_atol':2e-5,'cpu_mps_rtol':2e-5}


def check_soap_madam(device,output):
    if device not in ('cpu','mps'):raise ValueError('SOAP/Madam fixtures require CPU or MPS')
    root=Path(__file__).resolve().parents[3]
    source=root/'src/ralup/baselines/references/soap_reference.py'
    report={'schema_version':1,'suite':'soap-madam','device':device,'dtype':'float32',
        'recorded_utc':datetime.now(timezone.utc).isoformat(),'scientific_gate':None,
        'scope':'M02 bounded pinned mechanics and independent equations; no release reproduction, refresh winner, competitiveness or universal replay claim',
        'fallback_environment':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        'source_hashes':{str(p.relative_to(root)):sha256_file(p) for p in sorted(root.glob('src/ralup/**/*.py'))},
        'tolerances':dict(TOLERANCES),'status':'pass','blockers':[],'variants':[]}
    if sha256_file(source)!=SOURCE_SHA256:
        report.update(status='blocked',blockers=['Pinned SOAP oracle identity mismatch'])
    elif device=='mps' and (not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0'):
        report.update(status='blocked',blockers=['Actual MPS requires availability and fallback disabled before startup'])
    else:
        for name in SOAP_MADAM_NAMES:
            rule=matrix_rule(name,{'precondition_frequency':4} if name.startswith('soap') else {})
            variant={'name':name,'status':'pass','operation_status':'pass','parity_status':'pass','fixtures':[]}
            try:
                if name.startswith('soap'):
                    initial=torch.tensor([[-.3,0.,1.1],[.01,-2.,.7]])
                    w=initial.to(device);other=torch.nn.Parameter(initial.clone())
                    oracle=SOAP([other],lr=.03,weight_decay=.4,precondition_frequency=4);state={};ema=torch.zeros_like(other)
                    for index in range(12):
                        g=torch.tensor([[.2,-.7,.4],[.3,.9,-.1]])*(1 if index%2 else -1)+index*.03
                        p=rule.propose(w,g.to(device),state,SchedulePoint(.03,.4,index/12));rule.commit(w,p);state=p.new_optimizer_state
                        other.grad=g;oracle.step()
                        tol={'atol':0,'rtol':0} if device=='cpu' else {'atol':2e-5,'rtol':2e-5}
                        torch.testing.assert_close(w.cpu(),other,**tol)
                        if index:ema.mul_(.95).add_(g,alpha=.05)
                        torch.testing.assert_close(rule.first_moment_in_original_coordinates(state).cpu(),ema,atol=2e-5,rtol=2e-5)
                        for key in ('exp_avg','exp_avg_sq'):
                            torch.testing.assert_close(state[key].cpu(),oracle.state[other][key],**tol)
                        variant['fixtures'].append({'observation':index+1,'effective_adam_step':state['step'],
                            'weight':w.cpu().tolist(),'reference_weight':other.detach().tolist(),
                            'original_W_first_moment':rule.first_moment_in_original_coordinates(state).cpu().tolist(),
                            'inventory':rule.state_inventory(state)})
                    variant['complete_factors']=[t.cpu().tolist() if isinstance(t,torch.Tensor) else [] for t in state['GG']]
                    variant['complete_bases']=[t.cpu().tolist() if isinstance(t,torch.Tensor) else [] for t in state['Q']]
                else:
                    initial=[-2.,0.,.5];w=torch.tensor(initial,device=device);state={};v=[0.,0.,0.];expected=initial[:];bound=3*math.sqrt(4.25/3)
                    for index,g in enumerate([[1.,0.,-2.],[0.,0.,0.],[1000.,7.,-2000.],[-5.,0.,5.]]):
                        for i,gradient in enumerate(g):
                            v[i]=.999*v[i]+.001*gradient**2
                            normalized=0. if v[i]==0 else gradient/math.sqrt(v[i]/(1-.999**(index+1)))
                            normalized=max(-10.,min(10.,normalized));sign=0 if expected[i]==0 else math.copysign(1,expected[i])
                            expected[i]=max(-bound,min(bound,expected[i]*math.exp(-.5*sign*normalized)))
                        p=rule.propose(w,torch.tensor(g,device=device),state,SchedulePoint(.5,0.,index/4));rule.commit(w,p);state=p.new_optimizer_state
                        torch.testing.assert_close(w.cpu(),torch.tensor(expected),atol=2e-6,rtol=2e-6)
                        assert w[1].item()==0 and bool((w.abs()<=state['bound']).all()) and torch.equal(w.sign().cpu(),torch.tensor(initial).sign())
                        variant['fixtures'].append({'observation':index+1,'weight':w.cpu().tolist(),'scalar_equation_weight':expected[:],
                                                    'bound':state['bound'].item(),'exp_avg_sq':state['exp_avg_sq'].cpu().tolist()})
            except AssertionError as error:
                variant.update(status='fail',parity_status='fail',error=f'{type(error).__name__}: {error}')
                report.update(status='fail');report['blockers'].append(f'{name}: fidelity failed: {error}')
            except (RuntimeError,ValueError) as error:
                variant.update(status='blocked',operation_status='blocked',parity_status='blocked',error=f'{type(error).__name__}: {error}')
                if report['status']!='fail':report['status']='blocked'
                report['blockers'].append(f'{name}: operation blocked: {error}')
            record=replace(matrix_record(name,(ParameterGroup('internal_projection',('w',)),),rule,
                    parity_status=variant['parity_status'],operations_status=variant['operation_status']),artifact_kind='original_fixture')
            registry=MethodRegistry();registry.register(record)
            variant['method']=json.loads(json.dumps(asdict(record)))
            variant['claim_eligibility']=registry.claim_eligibility(name)
            report['variants'].append(variant)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(canonical_bytes(report)+b'\n')
    return ArtifactRef(output,sha256_file(output))
