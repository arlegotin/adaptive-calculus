import sys
sys.path.insert(0,'tests/unit')
from test_muon_variants import scalar_polynomial
import torch
print(repr(scalar_polynomial(torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)).tolist()))
