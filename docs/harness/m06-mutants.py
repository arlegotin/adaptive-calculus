"""Test-local contract violations; never product edits or scientific claims."""
import inspect,sys
import pytest
from ralup.baselines import learned_scale,noise
kind=sys.argv[1]
if kind=='finite-difference':
    # Away from transitions true hard-quantizer d(s*C)/ds=C. This incorrect
    # target removes both round-surrogate residual and LSQ normalization.
    source=inspect.getsource(learned_scale._LearnedScale.forward)
    source=source.replace('    @staticmethod\n','',1)
    import textwrap
    source=textwrap.dedent(source)
    mutant=source.replace("local=torch.where(x<=contract['minimum'],float(contract['minimum']),torch.where(x>=contract['maximum'],float(contract['maximum']),code-x))",'local=code').replace('ctx.factor=factor','ctx.factor=1.')
    assert mutant!=source
    namespace=dict(learned_scale.__dict__);exec(compile(mutant,'<M06 wrong hard-finite-difference derivative>','exec'),namespace)
    learned_scale._LearnedScale.forward=staticmethod(namespace['forward'])
    selection='tests/unit/test_learned_scale.py::test_analytic_surrogate_inside_outside_log_chain_normalizer'
elif kind=='alignment-probability':
    source=inspect.getsource(noise.mask_displacement)
    mutant=source.replace('accepted=uniform<.5','accepted=uniform<alignment')
    assert mutant!=source;exec(compile(mutant,'<M06 wrong alignment-dependent survival>','exec'),noise.__dict__)
    selection='tests/unit/test_noise_controls.py::test_simple_doubles_and_magma_alignment_never_changes_probability'
else:raise ValueError('Unknown M06 negative control')
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider',selection]))
