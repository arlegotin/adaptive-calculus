"""Explicit matrix manifest routing, never an embedding/head shape heuristic."""
from .base import MethodAdapter, SchedulePoint
from .muon import MuonUpdateRule
from .muon_plus import MuonPlusUpdateRule
from .soap import SoapUpdateRule
from .madam import MadamUpdateRule


class MatrixMethodAdapter(MethodAdapter):
    def __init__(self, parameters, groups, rule):
        if type(rule) not in (MuonUpdateRule,MuonPlusUpdateRule,SoapUpdateRule,MadamUpdateRule):
            raise ValueError('Matrix adapter requires a declared matrix rule')
        if tuple(g.name for g in groups) not in (('internal_projection',),('internal_projection','auxiliary_no_decay')):
            raise ValueError('Matrix adapter requires explicit internal and auxiliary groups')
        if groups[0].weight_decay is not None or (len(groups)>1 and groups[1].weight_decay != 0.):
            raise ValueError('Matrix decay follows schedule; auxiliary AdamW must explicitly exclude decay')
        if any(parameters[n].ndim != 2 for n in groups[0].parameter_names):
            raise ValueError('Internal projection manifest must contain matrices')
        super().__init__(parameters,groups,rule)
        self.snapshot_schema['optimizer_states'] = f'named.original_W.{rule.state_schema}+auxiliary.adamw.v1'

    def rule_for_parameter(self, name):
        return self.rule if name in self.groups[0].parameter_names else self.rule.auxiliary_rule

    def schedule_for_group(self, group, schedule):
        if group.name == 'auxiliary_no_decay':
            return SchedulePoint(self.rule.configuration['auxiliary_lr'],0.,schedule.progress)
        if type(self.rule) is MadamUpdateRule:
            return SchedulePoint(schedule.lr,0.,schedule.progress)
        return super().schedule_for_group(group,schedule)

    def finish_step(self, training_state, schedule):
        return self.commit_proposals(self.collect_proposals(schedule,work_recorder=training_state.record_work))
