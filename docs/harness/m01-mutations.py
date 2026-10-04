"""Fresh-process negative controls; never change product files on disk."""
import dataclasses
import inspect
import math
import sys
import pytest
from ralup.optim.muon import MuonUpdateRule
from ralup.optim.muon_plus import MuonPlusUpdateRule

if sys.argv[1] == 'multiplier':
    original = MuonPlusUpdateRule.propose
    def swapped(self,w,grad,state,schedule):
        p=original(self,w,grad,state,schedule)
        factor=math.sqrt(max(1,w.shape[0]/w.shape[1]))/math.sqrt(w.shape[0]/w.shape[1])
        return dataclasses.replace(p,gradient_displacement=p.gradient_displacement*factor)
    MuonPlusUpdateRule.propose=swapped
elif sys.argv[1] == 'momentum':
    import textwrap
    source=textwrap.dedent(inspect.getsource(MuonPlusUpdateRule.propose))
    source=source.replace('.add_(grad,alpha=1-beta)', '.add_(grad)')
    scope=dict(inspect.unwrap(MuonPlusUpdateRule.propose).__globals__)
    exec(source,scope)
    MuonPlusUpdateRule.propose=scope['propose']
else: raise ValueError('Unknown negative control')
raise SystemExit(pytest.main(['-q','tests/unit/test_muon_variants.py','-k','distinct_momentum_and_original_rectangular_displacements']))
