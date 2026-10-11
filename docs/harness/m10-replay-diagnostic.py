"""Captured, temporary diagnostic at H07 owner boundary; no numerical repair."""
import sys
import torch
sys.path[:0]=['tests/unit','tests/integration']
from test_metaquant_adapter import fixture
from ralup.train import train_step
from ralup.state import capture_snapshot,restore_snapshot,_validated
a,batch=fixture();train_step(a,batch);b=restore_snapshot(capture_snapshot(a),'cpu')
train_step(a,batch);train_step(b,batch)
def compare(x,y,path=''):
    if isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor):
        if not torch.equal(x,y):print(path,tuple(x.shape),'maxdiff',float((x-y).abs().max()))
    elif isinstance(x,dict) and isinstance(y,dict):
        for k in x:compare(x[k],y[k],path+'.'+k)
    elif isinstance(x,(tuple,list)) and isinstance(y,(tuple,list)):
        for i,(v,w) in enumerate(zip(x,y)):compare(v,w,path+f'[{i}]')
print('original gprev order',list(a.method.gprev),'restored',list(b.method.gprev))
compare(_validated(capture_snapshot(a)),_validated(capture_snapshot(b)))
