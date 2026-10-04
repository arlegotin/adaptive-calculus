"""S07 proposal boundary and ordinary hooks; H06 owns the step transaction."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Mapping, Protocol

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SchedulePoint:
    lr: float
    weight_decay: float
    progress: float

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.lr,self.weight_decay,self.progress)):
            raise ValueError('Schedule values must be finite')
        if self.lr < 0 or self.weight_decay < 0 or not 0 <= self.progress <= 1:
            raise ValueError('Schedule requires nonnegative LR/decay and progress in [0,1]')


@dataclass(frozen=True)
class Proposal:
    reference_after_decay: Tensor
    gradient_displacement: Tensor
    new_optimizer_state: dict[str, Tensor]
    # Exact original-coordinate bypass operands. Telemetry W0+dW need not
    # reproduce rounding in an upstream in-place addcdiv_ update.
    authoritative_update: Mapping[str, Tensor | float] = field(default_factory=dict)


class BaseUpdateRule(ABC):
    state_schema: str

    @property
    @abstractmethod
    def configuration(self) -> Mapping:
        """Declared immutable update settings, separately from mutable moments."""

    @abstractmethod
    def propose(self, w: Tensor, grad: Tensor, state: Mapping[str, Tensor], schedule: SchedulePoint) -> Proposal:
        """Return detached owned tensors without changing W, grad, or old state."""

    @abstractmethod
    def commit(self, w: Tensor, proposal: Proposal) -> None:
        """Authoritative original-coordinate bypass, preserving W identity."""


@dataclass(frozen=True)
class ParameterGroup:
    name: str
    parameter_names: tuple[str, ...]
    # None uses the shared schedule decay; 0 explicitly excludes auxiliaries.
    weight_decay: float | None = None

    def __post_init__(self):
        if not self.name or not self.parameter_names or len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError('Parameter group requires a name and unique ordered parameter names')
        if self.weight_decay is not None and (not math.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError('Group decay must be finite and nonnegative')


def default_parameter_groups(model: nn.Module) -> tuple[ParameterGroup, ...]:
    """H04's explicit six-projection manifest decays; all auxiliaries do not.

    Auxiliary names are lexically ordered and recorded, including norms,
    input/position embeddings and the output head. No shape heuristic grants
    decay to an embedding/head just because it is a matrix.
    """
    internal = tuple(item['name'] for item in model.quantized_matrix_manifest())
    names = set(dict(model.named_parameters()))
    if not internal or len(set(internal)) != len(internal) or not set(internal) <= names:
        raise ValueError('Invalid internal projection manifest')
    auxiliary = tuple(sorted(names-set(internal)))
    groups = (ParameterGroup('internal_projection',internal),)
    return groups + ((ParameterGroup('auxiliary_no_decay',auxiliary,0.),) if auxiliary else ())


class BackwardState(Protocol):
    def summed_nll(self, batch_slice) -> Tensor:
        """Shared scorer's scalar SUM, on exactly the supplied slice."""


def _finite(value) -> bool:
    if isinstance(value,Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value,Mapping):
        return all(_finite(item) for item in value.values())
    return math.isfinite(value)


class MethodAdapter:
    """Data-neutral ordinary adapter with named, snapshot-visible state.

    Caller clears gradients, prepares quantized contexts, selects/feeds slices,
    normalizes gradients once and clips once. This adapter never selects data,
    clears/normalizes/clips gradients, advances a schedule, or writes ledgers.
    Exceptional adapters override hooks and extend schemas/extra_passes.
    collect_proposals/commit_proposals separate M09 policy actions from bypass.
    H06 supplies rollback for errors during actual commit or later operations.
    """
    extra_passes: tuple[str, ...] = ()

    def __init__(self, parameters: Mapping[str, nn.Parameter], groups: tuple[ParameterGroup, ...], rule: BaseUpdateRule):
        ordered = tuple(name for group in groups for name in group.parameter_names)
        if len(set(ordered)) != len(ordered) or set(ordered) != set(parameters):
            raise ValueError('Ordered groups must cover each named parameter exactly once')
        if len({group.name for group in groups}) != len(groups):
            raise ValueError('Group names must be unique')
        if len({id(parameters[name]) for name in ordered}) != len(ordered):
            raise ValueError('Parameters must have unique identities in the manifest')
        self.parameters = {name:parameters[name] for name in ordered}
        self.groups = groups
        self.rule = rule
        self.optimizer_states: dict[str, dict[str,Tensor]] = {name:{} for name in ordered}
        self.proposals: dict[str,Proposal] = {}
        self.snapshot_schema = {'optimizer_states':f'named.original_W.{rule.state_schema}',
                                'proposals':'named.proposal.v1:reference_after_decay,gradient_displacement,new_optimizer_state,authoritative_update',
                                'adapter_mutable_state':'none.v1'}

    @property
    def metadata(self) -> dict:
        return {'ordered_parameter_groups':self.groups,'update_rule_settings':dict(self.rule.configuration),
                'schedule_fields':('lr','weight_decay','progress'),'extra_passes':self.extra_passes,
                'state_schema':dict(self.snapshot_schema)}

    def prepare_step(self, training_state, logical_batch) -> None:
        if self.proposals:
            raise ValueError('Uncommitted proposals remain from a previous step')

    def backward_microbatch(self, training_state: BackwardState, batch_slice) -> dict:
        loss = training_state.summed_nll(batch_slice)
        if loss.ndim != 0:
            raise ValueError('Shared scorer must return scalar summed NLL')
        loss.backward()
        return {'summed_nll':float(loss.detach()),'forward_passes':1,'backward_passes':1,
                'extra_forward_passes':0,'extra_backward_passes':0}

    def rule_for_parameter(self, name: str) -> BaseUpdateRule:
        return self.rule

    def schedule_for_group(self, group: ParameterGroup, schedule: SchedulePoint) -> SchedulePoint:
        return SchedulePoint(schedule.lr,schedule.weight_decay if group.weight_decay is None else group.weight_decay,schedule.progress)

    def collect_proposals(self, schedule: SchedulePoint) -> dict[str,Proposal]:
        if self.proposals:
            raise ValueError('Uncommitted proposals must be resolved before another collection')
        proposals = {}
        for group in self.groups:
            point = self.schedule_for_group(group,schedule)
            for name in group.parameter_names:
                parameter = self.parameters[name]
                if parameter.grad is not None:
                    if not _finite(parameter.grad):
                        raise ValueError(f'Gradient must be finite: {name}')
                    proposals[name] = self.rule_for_parameter(name).propose(parameter,parameter.grad,self.optimizer_states[name],point)
        self.proposals = proposals
        return proposals

    @torch.no_grad()
    def commit_proposals(self, proposals: dict[str,Proposal]) -> dict:
        if proposals is not self.proposals:
            raise ValueError('Commit requires the current uncommitted proposal collection')
        # Validate all actual rounded candidates before the first real commit.
        # These validation tensor updates are reported rather than hidden.
        for name,proposal in proposals.items():
            if not all(_finite(value) for value in (proposal.reference_after_decay,proposal.gradient_displacement,
                                                     proposal.new_optimizer_state,proposal.authoritative_update)):
                raise ValueError(f'Proposal and optimizer state must be finite: {name}')
            candidate = self.parameters[name].detach().clone()
            self.rule_for_parameter(name).commit(candidate,proposal)
            if not _finite(candidate):
                raise ValueError(f'Committed candidate must be finite: {name}')
        for name,proposal in proposals.items():
            self.rule_for_parameter(name).commit(self.parameters[name],proposal)
        # Publish new moments deliberately, only after every parameter commit.
        for name,proposal in proposals.items():
            self.optimizer_states[name] = proposal.new_optimizer_state
        self.proposals = {}
        return {'optimizer_updates':len(proposals),'candidate_validation_updates':len(proposals),
                'extra_forward_passes':0,'extra_backward_passes':0}

    def finish_step(self, training_state, schedule: SchedulePoint) -> dict:
        return self.commit_proposals(self.collect_proposals(schedule))
