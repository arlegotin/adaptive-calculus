"""Pinned MIT SOAP mechanics on owned proposals, with explicit CPU eig/QR.

Source: nikhilvyas/SOAP@a1e553530fde97d0e6b307d7c82ac6d38b072340.
The exact source, including AdamW/GaLore provenance, lives in references;
its full MIT notice is licenses/methods/soap/LICENSE. No preconditioning change.
"""
import math
import time
from types import MappingProxyType
import torch
from ..baselines.references.soap_reference import SOAP
from .base import BaseUpdateRule, Proposal
from .adamw import AdamWUpdateRule


def _clone(value):
    if isinstance(value,torch.Tensor): return value.detach().clone()
    if isinstance(value,dict): return {k:_clone(v) for k,v in value.items()}
    if isinstance(value,list): return [_clone(v) for v in value]
    return value


TIMING_FIELDS = ('factorization_seconds','transfer_seconds')
TELEMETRY_FIELDS = (*TIMING_FIELDS,'transfer_bytes','factorization_count','eig_calls','qr_calls','float64_eig_retries')


class _PortableSOAP(SOAP):
    """Pinned eig/QR arithmetic with measured CPU transport and launch counts."""
    def _factorize(self, state_or_factors, qr=False, max_precond_dim=10000, merge_dims=False):
        factors=state_or_factors['GG'] if qr else state_or_factors
        active=next((t for t in factors if isinstance(t,torch.Tensor)),None)
        if active is None:
            return SOAP.get_orthogonal_matrix_QR(self,state_or_factors,max_precond_dim,merge_dims) if qr else SOAP.get_orthogonal_matrix(self,factors)
        device=active.device
        moved=device.type!='cpu'
        start=time.perf_counter()
        if moved:torch.mps.synchronize()
        def cpu(t):
            if isinstance(t,torch.Tensor):
                if moved:self.telemetry['transfer_bytes']+=t.numel()*t.element_size()
                return t.detach().to('cpu').clone()
            return []
        if qr:
            cpu_state={'GG':[cpu(t) for t in factors],'Q':[cpu(t) for t in state_or_factors['Q']],
                       'exp_avg_sq':cpu(state_or_factors['exp_avg_sq'])}
        else:cpu_factors=[cpu(t) for t in factors]
        self.telemetry['transfer_seconds']+=time.perf_counter()-start if moved else 0.
        start=time.perf_counter()
        # Charge launches, including a failure, before the upstream factorization.
        self.telemetry['qr_calls' if qr else 'eig_calls']+=1
        try:
            if qr:
                original_shape=cpu_state['exp_avg_sq'].shape
                channels_last=self._data_format=='channels_last' and len(original_shape)==4
                if channels_last:permuted_shape=cpu_state['exp_avg_sq'].permute(0,3,1,2).shape
                second=self.merge_dims(cpu_state['exp_avg_sq'],max_precond_dim) if merge_dims else cpu_state['exp_avg_sq']
                result=[]
                for axis,(m,q) in enumerate(zip(cpu_state['GG'],cpu_state['Q'])):
                    if not isinstance(m,torch.Tensor):result.append([]);continue
                    # Exact pinned ordering, moment-axis permutation and QR.
                    m,q=m.float(),q.float()
                    estimated=torch.diag(q.T@m@q);order=torch.argsort(estimated,descending=True)
                    second=second.index_select(axis,order);q=q[:,order]
                    power=m@q;self.telemetry['factorization_count']+=1
                    basis,_=torch.linalg.qr(power);result.append(basis)
                if merge_dims:
                    second=second.reshape(permuted_shape).permute(0,2,3,1) if channels_last else second.reshape(original_shape)
                cpu_state['exp_avg_sq']=second
            else:
                result=[]
                for m in cpu_factors:
                    if not isinstance(m,torch.Tensor):result.append([]);continue
                    self.telemetry['factorization_count']+=1
                    try:_,basis=torch.linalg.eigh(m+1e-30*torch.eye(m.shape[0],device=m.device))
                    except RuntimeError:
                        self.telemetry['factorization_count']+=1;self.telemetry['float64_eig_retries']+=1
                        _,basis=torch.linalg.eigh(m.to(torch.float64)+1e-30*torch.eye(m.shape[0],device=m.device));basis=basis.to(m.dtype)
                    result.append(torch.flip(basis,[1]))
        finally:self.telemetry['factorization_seconds']+=time.perf_counter()-start
        start=time.perf_counter()
        def back(t):
            if isinstance(t,torch.Tensor):
                if moved:self.telemetry['transfer_bytes']+=t.numel()*t.element_size()
                return t.to(device)
            return []
        result=[back(t) for t in result]
        if qr:state_or_factors['exp_avg_sq']=back(cpu_state['exp_avg_sq'])
        if moved:
            torch.mps.synchronize();self.telemetry['transfer_seconds']+=time.perf_counter()-start
        return result

    def get_orthogonal_matrix(self, factors):return self._factorize(factors)
    def get_orthogonal_matrix_QR(self,state,max_precond_dim=10000,merge_dims=False):
        return self._factorize(state,True,max_precond_dim,merge_dims)


class SoapUpdateRule(BaseUpdateRule):
    state_schema='soap_pinned_cpu_factorization.v1'
    optimizer_state_fields=('step','exp_avg','exp_avg_sq','GG','Q','precondition_frequency','shampoo_beta','basis_strides','telemetry')
    defaults={'betas':(.95,.95),'shampoo_beta':-1.,'eps':1e-8,'precondition_frequency':10,
              'max_precond_dim':10000,'merge_dims':False,'precondition_1d':False,'normalize_grads':False,
              'data_format':'channels_first','correct_bias':True,'auxiliary_lr':.001,
              'auxiliary_betas':(.9,.99),'auxiliary_eps':1e-8}

    def __init__(self,config:dict):
        if type(config) is not dict or set(config)-self.defaults.keys():raise ValueError('Unknown SOAP configuration')
        settings={**self.defaults,**config}
        settings['betas']=AdamWUpdateRule(betas=tuple(settings['betas']),eps=settings['eps']).betas
        for name in ('precondition_frequency','max_precond_dim'):
            if type(settings[name]) is not int or settings[name]<=0:raise ValueError('SOAP dimensions/frequency must be positive integers')
        for name in ('merge_dims','precondition_1d','normalize_grads','correct_bias'):
            if type(settings[name]) is not bool:raise ValueError('SOAP flags must be boolean')
        beta=settings['shampoo_beta']
        if type(beta) not in (int,float) or not math.isfinite(beta) or not (beta==-1 or 0<=beta<1):raise ValueError('Invalid Shampoo beta')
        if settings['data_format'] not in ('channels_first','channels_last'):raise ValueError('Unknown SOAP data format')
        if type(settings['auxiliary_lr']) not in (int,float) or not math.isfinite(settings['auxiliary_lr']) or settings['auxiliary_lr']<0:raise ValueError('Invalid auxiliary LR')
        aux=AdamWUpdateRule(betas=tuple(settings['auxiliary_betas']),eps=settings['auxiliary_eps']);settings['auxiliary_betas']=aux.betas
        self.config=MappingProxyType(settings)

    @property
    def configuration(self):return dict(self.config)
    @property
    def auxiliary_rule(self):return AdamWUpdateRule(betas=self.config['auxiliary_betas'],eps=self.config['auxiliary_eps'])
    def _mechanics(self,w,**extra):
        settings={k:v for k,v in self.config.items() if not k.startswith('auxiliary_')}
        return _PortableSOAP([w],**settings,**extra)

    def validate_state(self,state,w):
        if set(state)!=set(self.optimizer_state_fields):raise ValueError('Incomplete SOAP state inventory')
        if type(state['step']) is not int or state['step']<0:raise ValueError('Invalid SOAP Adam step')
        if state['precondition_frequency']!=self.config['precondition_frequency'] or state['shampoo_beta']!=(self.config['betas'][1] if self.config['shampoo_beta']<0 else self.config['shampoo_beta']):raise ValueError('SOAP preconditioner settings mismatch')
        for key in ('exp_avg','exp_avg_sq'):
            t=state[key]
            if not isinstance(t,torch.Tensor) or t.shape!=w.shape or t.dtype!=w.dtype or t.device!=w.device or not bool(torch.isfinite(t).all()):raise ValueError('SOAP moment geometry/dtype/device mismatch')
        if bool((state['exp_avg_sq']<0).any()):raise ValueError('Negative SOAP second moment')
        mechanics=self._mechanics(w)
        dimensions=mechanics.merge_dims(w,self.config['max_precond_dim']).shape if self.config['merge_dims'] and w.ndim!=1 else w.shape
        geometries=[(n,n) if n<=self.config['max_precond_dim'] and (w.ndim!=1 or self.config['precondition_1d']) else None for n in dimensions]
        for key in ('GG','Q'):
            if type(state[key]) is not list or len(state[key])!=len(geometries):raise ValueError('SOAP factor/basis axis inventory mismatch')
            for t,shape in zip(state[key],geometries):
                if shape is not None:
                    if not isinstance(t,torch.Tensor) or tuple(t.shape)!=shape or t.dtype!=w.dtype or t.device!=w.device or not bool(torch.isfinite(t).all()):raise ValueError('SOAP factor/basis geometry mismatch')
                elif type(t) is not list or t:raise ValueError('SOAP skipped axes must be empty lists')
        expected_strides=[(1,t.shape[0]) if isinstance(t,torch.Tensor) else () for t in state['Q']]
        if type(state['basis_strides']) is not list or any(type(pair) is not tuple or any(type(v) is not int for v in pair) for pair in state['basis_strides']) or state['basis_strides']!=expected_strides:
            raise ValueError('SOAP requires the pinned column-major basis layout')
        telemetry=state['telemetry']
        if type(telemetry) is not dict or set(telemetry)!=set(TELEMETRY_FIELDS):raise ValueError('Incomplete SOAP telemetry')
        for k,v in telemetry.items():
            if type(v) not in (int,float) or not math.isfinite(v) or v<0 or (k not in TIMING_FIELDS and type(v) is not int):raise ValueError('Invalid SOAP telemetry')
        active=sum(shape is not None for shape in geometries)
        eig=1 if active else 0;qr=state['step']//self.config['precondition_frequency'] if active else 0
        if telemetry['eig_calls']!=eig or telemetry['qr_calls']!=qr or telemetry['float64_eig_retries']>active or telemetry['factorization_count']!=active*(eig+qr)+telemetry['float64_eig_retries']:
            raise ValueError('SOAP factorization counters mismatch effective step/geometry')
        if state['step']==0 and (bool(state['exp_avg'].any()) or bool(state['exp_avg_sq'].any())):raise ValueError('SOAP initialization cannot observe Adam moments')

    @torch.no_grad()
    def propose(self,w,grad,state,schedule):
        if w.dtype!=torch.float32 or w.layout!=torch.strided or grad.is_sparse or not w.ndim or not all(w.shape) or grad.shape!=w.shape or grad.dtype!=w.dtype or grad.device!=w.device or w.device.type not in ('cpu','mps'):raise ValueError('SOAP requires matching dense FP32 CPU/MPS tensors')
        if not bool(torch.isfinite(w).all()) or not bool(torch.isfinite(grad).all()):raise ValueError('SOAP W/gradient must be finite')
        if state:
            self.validate_state(state,w)
            if any(isinstance(t,torch.Tensor) and tuple(t.stride())!=stride for t,stride in zip(state['Q'],state['basis_strides'])):
                raise ValueError('SOAP live basis layout differs from its declared source layout')
        scratch=torch.nn.Parameter(w.detach().clone());mechanics=self._mechanics(scratch,lr=schedule.lr,weight_decay=schedule.weight_decay)
        owned=_clone(state);previous=owned.pop('telemetry',None);owned.pop('basis_strides',None)
        mechanics.telemetry={k:0. if k in TIMING_FIELDS else 0 for k in TELEMETRY_FIELDS}
        mechanics.state[scratch]=owned
        try:
            if not state:
                owned.update(step=0,exp_avg=torch.zeros_like(grad),exp_avg_sq=torch.zeros_like(grad))
                mechanics.init_preconditioner(grad,owned,precondition_frequency=self.config['precondition_frequency'],
                    shampoo_beta=self.config['betas'][1] if self.config['shampoo_beta']<0 else self.config['shampoo_beta'],
                    max_precond_dim=self.config['max_precond_dim'],merge_dims=self.config['merge_dims'],precondition_1d=self.config['precondition_1d'])
                mechanics.update_preconditioner(grad,owned,max_precond_dim=self.config['max_precond_dim'],
                    merge_dims=self.config['merge_dims'],precondition_1d=self.config['precondition_1d'])
                displacement=torch.zeros_like(w)
            else:
                projected=mechanics.project(grad,owned,merge_dims=self.config['merge_dims'],max_precond_dim=self.config['max_precond_dim'])
                beta1,beta2=self.config['betas'];owned['step']+=1
                owned['exp_avg'].mul_(beta1).add_(projected,alpha=1-beta1)
                owned['exp_avg_sq'].mul_(beta2).add_(projected.square(),alpha=1-beta2)
                denom=owned['exp_avg_sq'].sqrt().add_(self.config['eps'])
                step_size=schedule.lr
                if self.config['correct_bias']:step_size=step_size*(1-beta2**owned['step'])**.5/(1-beta1**owned['step'])
                direction=mechanics.project_back(owned['exp_avg']/denom,owned,merge_dims=self.config['merge_dims'],max_precond_dim=self.config['max_precond_dim'])
                if self.config['normalize_grads']:direction=direction/(1e-30+torch.mean(direction**2)**.5)
                scratch.add_(direction,alpha=-step_size)
                if schedule.weight_decay>0:scratch.add_(scratch,alpha=-schedule.lr*schedule.weight_decay)
                displacement=-step_size*direction*(1-schedule.lr*schedule.weight_decay)
                mechanics.update_preconditioner(grad,owned,max_precond_dim=self.config['max_precond_dim'],
                    merge_dims=self.config['merge_dims'],precondition_1d=self.config['precondition_1d'])
        except Exception as error:
            error.optimizer_work=dict(mechanics.telemetry);raise
        new=mechanics.state[scratch]
        new['basis_strides']=[tuple(t.stride()) if isinstance(t,torch.Tensor) else () for t in new['Q']]
        new['telemetry']={k:mechanics.telemetry[k]+(previous[k] if previous else 0) for k in TELEMETRY_FIELDS}
        reference=w.detach().clone()
        if state and schedule.weight_decay>0:reference.add_(reference,alpha=-schedule.lr*schedule.weight_decay)
        # Candidate is the native add_(direction,alpha), then add_(self,alpha) result.
        candidate=scratch.detach().clone()
        return Proposal(reference,displacement,new,{'candidate':candidate})

    @torch.no_grad()
    def commit(self,w,proposal):w.copy_(proposal.authoritative_update['candidate'])

    def restore_basis_layout(self,state):
        # Safetensors is canonical row-major. Rebuild only the local source's
        # fixed square column-major layout, never arbitrary artifact strides.
        self.validate_state(state,state['exp_avg'])
        state['Q']=[t.T.contiguous().T if isinstance(t,torch.Tensor) else [] for t in state['Q']]

    def first_moment_in_original_coordinates(self,state:dict):
        self.validate_state(state,state['exp_avg'])
        return self._mechanics(state['exp_avg']).project_back(state['exp_avg'],state,merge_dims=self.config['merge_dims'],max_precond_dim=self.config['max_precond_dim']).detach().clone()

    def state_inventory(self,state:dict):
        self.validate_state(state,state['exp_avg'])
        size=lambda arrays:sum(t.numel()*t.element_size() for t in arrays if isinstance(t,torch.Tensor))
        factor=size(state['GG']);basis=size(state['Q']);moments=size([state['exp_avg'],state['exp_avg_sq']])
        return {'tensor_bytes':factor+basis+moments,'factor_bytes':factor,'basis_bytes':basis,'moment_bytes':moments,
                'basis_strides':[list(pair) for pair in state['basis_strides']],'factorization_location':'cpu','tensor_device':str(state['exp_avg'].device),**state['telemetry']}
