"""S19/S20 finite controllers. Deployment records contain arrays, never code.

Both learned families consume the same normalized fourteen causal scalars.
Training/rollout selection remains external (M11); no query value, matrix ID or
coverage field is read here. Feature removals mask only after the common scan.
"""
from dataclasses import dataclass
import json
import math
import torch
from torch import nn
from .features import FeatureVector
from ..config import canonical_bytes
from ..records import ArtifactRef, sha256_file

AFFINE_FAMILIES=('affine45','initialized','additive_only','multiplicative_only','assistance_only','no_history','no_cells','norm_matched')
FAMILIES={**{name:45 for name in AFFINE_FAMILIES},'learned_lr49':49,'learned_lr49_shared_range':49,'static3':3}


@dataclass(frozen=True)
class PolicyAction:
    alpha: float
    h: float
    tau: float

    def __post_init__(self):
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (self.alpha,self.h,self.tau)) or not 0<=self.alpha<=1 or not .25<=self.h<=32 or not .125<=self.tau<=1:
            raise ValueError('Finite bounded policy action required')

    @property
    def identity(self):return (self.alpha,self.h,self.tau)==(0.,1.,1.)


def _values(features):
    if type(features) is not FeatureVector:raise ValueError('Only fourteen causal FeatureVector scalars accepted')
    return features.values.detach().to(device='cpu',dtype=torch.float32)


def _action(raw):
    a,b,c=(float(v) for v in raw.detach())
    if not all(math.isfinite(v) for v in (a,b,c)):raise ValueError('Nonfinite controller output')
    return PolicyAction(max(0.,min(1.,a)),math.exp(max(math.log(.25),min(math.log(4),b))),max(.125,min(1.,c)))


class Affine45(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(14,3)
        with torch.no_grad():self.linear.weight.zero_();self.linear.bias.copy_(torch.tensor([0.,0.,1.]))

    def forward(self,features):return _action(self.linear(_values(features)))


class LearnedLR49(nn.Module):
    def __init__(self,*,max_multiplier=32):
        super().__init__()
        if max_multiplier not in (4,32):raise ValueError('Declared LR or shared-range sensitivity required')
        self.max_multiplier=max_multiplier;self.hidden=nn.Linear(14,3);self.output=nn.Linear(3,1)
        with torch.no_grad():self.output.weight.zero_();self.output.bias.zero_()

    def forward(self,features):
        value=float(self.output(self.hidden(_values(features)).tanh()).detach().item())
        if not math.isfinite(value):raise ValueError('Nonfinite LR output')
        return math.exp(max(math.log(.25),min(math.log(self.max_multiplier),value)))


class Static3(nn.Module):
    """Three future-loss optimized constants; no training objective is executed here."""
    def __init__(self):
        super().__init__();self.constants=nn.Parameter(torch.tensor([0.,0.,1.]))

    def forward(self,features):
        _values(features)  # Same information boundary; scans are still charged.
        return _action(self.constants)


def count_parameters(controller):return sum(p.numel() for p in controller.parameters() if p.requires_grad)


@dataclass(frozen=True)
class ControllerState:
    family: str
    parameters: torch.Tensor

    def __post_init__(self):
        if self.family not in FAMILIES or not isinstance(self.parameters,torch.Tensor) or self.parameters.dtype!=torch.float32 or self.parameters.shape!=(FAMILIES[self.family],) or not bool(torch.isfinite(self.parameters).all()):
            raise ValueError('Registered finite controller family/parameter array required')
        object.__setattr__(self,'parameters',self.parameters.detach().clone())
        if self.family=='initialized' and not torch.equal(self.parameters.cpu(),torch.tensor([0.]*42+[0.,0.,1.])):
            raise ValueError('Candidate initialization is frozen at exact identity')


def freeze_controller(controller,*,family=None):
    types={Affine45:'affine45',LearnedLR49:'learned_lr49',Static3:'static3'}
    if type(controller) not in types:raise ValueError('Only finite trusted controller constructors accepted')
    expected=types[type(controller)]
    base_fields=set(vars(nn.Module()))
    allowed={'affine45':set(),'learned_lr49':{'max_multiplier'},'static3':set()}[expected]
    if set(vars(controller))!=base_fields|allowed:raise ValueError('Unknown controller attributes or executable owner')
    if expected=='learned_lr49' and controller.max_multiplier not in (4,32):raise ValueError('Invalid LR range')
    family=family or ('learned_lr49_shared_range' if type(controller) is LearnedLR49 and controller.max_multiplier==4 else expected)
    if ((expected=='affine45' and family not in AFFINE_FAMILIES) or
        (expected=='learned_lr49' and family not in ('learned_lr49','learned_lr49_shared_range')) or
        (expected=='static3' and family!='static3')):raise ValueError('Controller family/capacity mismatch')
    # No executable hooks or added buffers/children may cross the deployment boundary.
    names={'affine45':('linear.weight','linear.bias'),'learned_lr49':('hidden.weight','hidden.bias','output.weight','output.bias'),'static3':('constants',)}[expected]
    if tuple(dict(controller.named_parameters()))!=names or list(controller.buffers()):raise ValueError('Unknown controller owners')
    for module in controller.modules():
        permitted=base_fields|({'in_features','out_features'} if type(module) is nn.Linear else allowed)
        if set(vars(module))!=permitted:raise ValueError('Unknown controller module fields')
        if type(module) not in (type(controller),nn.Linear) or any(getattr(module,n) for n in ('_forward_hooks','_forward_pre_hooks','_backward_hooks')) or 'forward' in vars(module):raise ValueError('Executable controller hooks forbidden')
    shapes={'linear.weight':(3,14),'linear.bias':(3,),'hidden.weight':(3,14),'hidden.bias':(3,),'output.weight':(1,3),'output.bias':(1,),'constants':(3,)}
    for name,param in controller.named_parameters():
        if param.dtype!=torch.float32 or tuple(param.shape)!=shapes[name] or param._backward_hooks or param._post_accumulate_grad_hooks:raise ValueError('Invalid controller parameter shape/dtype/hooks')
    p=torch.cat([v.detach().cpu().flatten() for v in controller.parameters()])
    return ControllerState(family,p)


@torch.no_grad()
def evaluate_controller(state,features):
    if type(state) is not ControllerState:raise ValueError('Registered controller state required')
    # Validate live tensors too; frozen dataclasses do not make tensor bytes immutable.
    ControllerState(state.family,state.parameters)
    x=_values(features).clone();p=state.parameters.cpu()
    if state.family=='no_history':x[8:12]=0
    if state.family=='no_cells':x[4:8]=0
    if state.family.startswith('learned_lr49'):
        hidden=torch.nn.functional.linear(x,p[:42].reshape(3,14),p[42:45]).tanh()
        raw=float(torch.nn.functional.linear(hidden,p[45:48].reshape(1,3),p[48:49]).item())
        maximum=4 if state.family=='learned_lr49_shared_range' else 32
        if not math.isfinite(raw):raise ValueError('Nonfinite LR output')
        return PolicyAction(0.,math.exp(max(math.log(.25),min(math.log(maximum),raw))),1.)
    action=_action(p if state.family=='static3' else torch.nn.functional.linear(x,p[:42].reshape(3,14),p[42:45]))
    if state.family=='additive_only':return PolicyAction(0.,action.h,1.)
    if state.family=='multiplicative_only':return PolicyAction(1.,action.h,1.)
    if state.family=='assistance_only':return PolicyAction(0.,1.,action.tau)
    return action


def authenticate(ref):
    if type(ref) is not ArtifactRef or sha256_file(ref.path)!=ref.sha256:raise ValueError('Authenticated selected controller provenance required')
    return json.loads(ref.path.read_bytes())


@dataclass(frozen=True)
class ControllerArtifact:
    family: str
    parameters: torch.Tensor
    normalization: ArtifactRef
    B0: ArtifactRef
    data: ArtifactRef
    training: ArtifactRef

    def __post_init__(self):
        state=ControllerState(self.family,self.parameters)
        object.__setattr__(self,'parameters',state.parameters)
        for ref in (self.normalization,self.B0,self.data,self.training):authenticate(ref)

    def to_json(self):
        # Reauthentication ensures a stale/mutated deployment cannot be exported.
        ControllerArtifact(self.family,self.parameters,self.normalization,self.B0,self.data,self.training)
        ref=lambda r:{'path':str(r.path.resolve()),'sha256':r.sha256}
        return {'schema':'ralup.controller_artifact.v1','family':self.family,'parameters':self.parameters.cpu().tolist(),
                'normalization':ref(self.normalization),'B0':ref(self.B0),'data':ref(self.data),'training':ref(self.training),
                'privileged_candidate_rms':self.family=='norm_matched','source_information':'normalized_causal14_no_matrix_id',
                'h_range':[.25,32 if self.family=='learned_lr49' else 4],
                'training_objective':'future_loss; external M11 owner'}

    @classmethod
    def from_json(cls,value):
        keys={'schema','family','parameters','normalization','B0','data','training','privileged_candidate_rms','source_information','h_range','training_objective'}
        if type(value) is not dict or set(value)!=keys or value['schema']!='ralup.controller_artifact.v1':raise ValueError('Unknown controller artifact fields')
        if type(value['parameters']) is not list or any(type(v) not in (int,float) for v in value['parameters']):raise ValueError('Plain finite parameter list required')
        obj=cls(value['family'],torch.tensor(value['parameters'],dtype=torch.float32),*(ArtifactRef(**value[k]) for k in ('normalization','B0','data','training')))
        if canonical_bytes(obj.to_json())!=canonical_bytes(value):raise ValueError('Controller artifact contract mismatch')
        return obj
