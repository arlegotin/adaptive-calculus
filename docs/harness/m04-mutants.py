"""Test-local negative controls; product source is never changed."""
import inspect,sys
import pytest
from ralup.baselines import trs
kind=sys.argv[1]
if kind=='additive':
    source=inspect.getsource(trs.trs_effective_lr)
    source=source.replace('r=mu*r+(1-mu)*r*target_rate/(K+1e-8)','r=max(0.,r+(1-mu)*(target_rate-K))')
    assert source!=inspect.getsource(trs.trs_effective_lr)
    exec(compile(source,'<M04 additive-feedback mutant>','exec'),trs.__dict__)
    selection='tests/unit/test_trs.py::test_released_first_rate_ignored_and_multiplicative_fixed_values'
elif kind=='microbatch':
    original=trs.TrsBackboneAdapter.backward_microbatch
    def count_microbatch(self,state,batch):
        for name in self.groups[0].parameter_names:
            _,new,_=self._feedback(name,state.schedule['point'])
            self.trs_states[name]=new
        return original(self,state,batch)
    trs.TrsBackboneAdapter.backward_microbatch=count_microbatch
    selection='tests/integration/test_trs_accumulation.py::test_one_feedback_commit_per_logical_batch_and_validation_read_only'
else:raise ValueError('Unknown mutant')
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider',selection,*sys.argv[2:]]))
