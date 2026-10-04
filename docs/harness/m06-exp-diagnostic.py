"""FP32 CPU/MPS exp owner validation diagnostic; no loss/data pass."""
import json,os
from pathlib import Path
import torch
assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0' and torch.backends.mps.is_available()
x=torch.linspace(-20.,10.,1024,dtype=torch.float32)
cpu=x.exp();mps=x.to('mps').exp().cpu()
delta=(mps.view(torch.int32).long()-cpu.view(torch.int32).long()).abs()
relative=((mps-cpu).abs()/cpu).max().item()
value={'operation':'FP32 exp','scope':'1024 explicit samples in default log-scale range; no universal libm claim or tolerance proof',
       'maximum_observed_ulp_difference':int(delta.max()),'maximum_observed_relative_difference':relative,
       'loss_passes':0,'backward_passes':0,'observed_different_elements':int((cpu!=mps).sum())}
path=Path(os.environ['M06_ATTEMPT_DIR'])/'diagnostic.json';path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
print(json.dumps(value,sort_keys=True))
