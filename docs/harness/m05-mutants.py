"""Test-local contract violations; no product edits and no scientific claims."""
import inspect,sys
import pytest
from ralup.baselines import qar,rounding
kind=sys.argv[1]
if kind=='unmasked':
    source=inspect.getsource(qar._routing)
    mutant=source.replace('a=torch.where(feasible,signal.abs(),0.)','a=signal.abs()')
    assert mutant!=source;exec(compile(mutant,'<M05 unmasked-power mutant>','exec'),qar.__dict__)
    selection='tests/unit/test_endpoint_routing.py::test_mask_before_power_normalization_and_grouping'
elif kind=='shadow':
    original=qar.QarAdjacentAdapter.finish_step
    def retain_shadow(self,state,schedule):
        result=original(self,state,schedule)
        self.shadow_residual={n:p.detach().clone() for n,p in self.parameters.items()}
        return result
    qar.QarAdjacentAdapter.finish_step=retain_shadow
    selection='tests/integration/test_stochastic_context.py::test_one_realization_microbatch_cache_and_deterministic_evaluation[qar_fp4_adjacent]'
elif kind=='microbatch':
    original=rounding.RoundingAdapter.backward_microbatch
    def redraw(self,state,batch):
        state.model._context_step=None
        self.prepare_contexts(state,state.schedule['step'])
        return original(self,state,batch)
    rounding.RoundingAdapter.backward_microbatch=redraw
    selection='tests/integration/test_stochastic_context.py::test_one_realization_microbatch_cache_and_deterministic_evaluation[stochastic_shadow_forward]'
else:raise ValueError('Unknown M05 negative control')
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider',selection]))
