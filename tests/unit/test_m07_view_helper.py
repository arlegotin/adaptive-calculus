"""Actual model view transitions used by the M07 query helper."""
import importlib.util
from pathlib import Path
import torch
from ralup.models.transformer import ByteTransformer,ModelConfig


def test_fp32_view_clears_quantized_cache_without_replacing_weights():
    path=Path('docs/harness/m07-prioritized-views-v2.py')
    spec=importlib.util.spec_from_file_location('m07_view_helper',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    model=ByteTransformer(ModelConfig(1,8,2));tokens=torch.tensor([[256,1,2]]);positions=torch.tensor([[0,1,2]])
    expected=model(tokens,positions).detach();ids=[id(p) for p in model.parameters()]
    weights=[p.detach().clone() for p in model.parameters()]
    recipe={'name':'row_dynamic_max','rotation':'identity','backward':'ste'}
    model.configure_quantization(recipe);model.prepare_quantized_views(0,{'seed':1})
    old=[getattr(model,k) for k in ('_recipe','_runtime','_contexts','_context_step')]
    assert model.quantization_contexts
    try:
        helper.configure_view(model,None)
        assert model._recipe is None and not model.quantization_contexts
        assert torch.equal(model(tokens,positions).detach(),expected)
        assert [id(p) for p in model.parameters()]==ids
        assert all(torch.equal(p,w) for p,w in zip(model.parameters(),weights))
    finally:
        for key,value in zip(('_recipe','_runtime','_contexts','_context_step'),old):setattr(model,key,value)
    assert all(getattr(model,k) is v for k,v in zip(('_recipe','_runtime','_contexts','_context_step'),old))
    helper.configure_view(model,recipe)
    assert model._recipe==recipe and not model.quantization_contexts
