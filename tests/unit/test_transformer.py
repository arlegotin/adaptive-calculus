"""Catch count drift, fused projections, noncausal attention and stale graphs."""
import math

import pytest
import torch
import torch.nn.functional as F

from ralup.models import ByteTransformer, ModelConfig
from ralup.models.linear import QuantizedLinear
from ralup.quantization.context import prepare_context
from ralup.quantization.oracle import e2m1_oracle


@pytest.mark.parametrize('shape,total,internal', [((2,128,4),493056,393216), ((4,256,4),3347456,3145728), ((6,512,8),19281920,18874368)])
def test_architecture_counts_and_separate_matrix_manifest(shape, total, internal):
    model = ByteTransformer(ModelConfig(*shape))
    assert sum(p.numel() for p in model.parameters()) == total
    manifest = model.quantized_matrix_manifest()
    assert sum(math.prod(item['shape']) for item in manifest) == internal
    assert [item['name'] for item in manifest] == [f'blocks.{i}.{name}.weight' for i in range(shape[0]) for name in ('q','k','v','attention_output','up','down')]
    parameters = dict(model.named_parameters())
    assert all(parameters[item['name']].shape == tuple(item['shape']) for item in manifest)
    assert len({id(parameters[item['name']]) for item in manifest}) == shape[0]*6
    assert model.embedding.weight is not model.output.weight


def test_initialization_gains_biases_and_exact_nonlinearity():
    torch.manual_seed(41)
    model = ByteTransformer(ModelConfig(4,256,4))
    for name, parameter in model.named_parameters():
        if parameter.ndim == 1:
            torch.testing.assert_close(parameter, torch.zeros_like(parameter) if name.endswith('bias') else torch.ones_like(parameter))
        else:
            expected_std = .02/math.sqrt(8) if name.endswith(('attention_output.weight','down.weight')) else .02
            assert abs(float(parameter.detach().mean())) < expected_std*.025
            assert abs(float(parameter.detach().std())/expected_std-1) < .025
    assert all(m.eps == 1e-5 for m in model.modules() if isinstance(m, torch.nn.LayerNorm))
    assert all(m.approximate == 'none' for m in model.modules() if isinstance(m, torch.nn.GELU))
    assert all(m.bias is None for m in model.modules() if isinstance(m, (QuantizedLinear,torch.nn.Linear)))
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())


def test_future_tokens_cannot_change_earlier_logits():
    torch.manual_seed(73)
    model = ByteTransformer(ModelConfig(2,32,4))
    inputs = torch.tensor([[256,3,7,11,17,19,23]])
    positions = torch.arange(7)[None,:]
    before = model(inputs, positions)
    changed = inputs.clone()
    changed[:,3:] = torch.tensor([101,102,103,104])
    torch.testing.assert_close(before[:,:3], model(changed, positions)[:,:3], atol=0, rtol=0)


@pytest.mark.parametrize('rotation',['identity','full_hadamard'])
def test_quantized_projection_matches_independent_oracle_and_preserves_identity(rotation):
    layer = QuantizedLinear(4,2)
    original = layer.weight
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.,2.,-3.,4.],[0.,1.,3.,-2.]]))
    x = torch.tensor([[1.,2.,0.,-1.]])
    dense = torch.tensor([[1.,1.,1.,1.],[1.,-1.,1.,-1.],[1.,1.,-1.,-1.],[1.,-1.,-1.,1.]])/2
    r = torch.eye(4) if rotation == 'identity' else dense
    torch.testing.assert_close((x@r)@(layer.weight@r).T, F.linear(x,layer.weight), atol=2e-6,rtol=2e-6)
    context = prepare_context(layer.weight, {'name':'row_dynamic_max','rotation':rotation},0,None)
    layer.context = context
    # Oracle directly searches all fifteen FP64 codebook distances.
    values = e2m1_oracle((layer.weight.detach()@r).double().numpy(), context.scales.double().numpy())
    torch.testing.assert_close(layer(x), (x@r)@torch.tensor(values,dtype=torch.float32).T,atol=8e-6,rtol=2e-6)
    levels = torch.tensor([-6.,-4.,-3.,-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.,3.,4.,6.])
    expected_codes = (torch.tensor(values)/context.scales).unsqueeze(-1).sub(levels).abs().argmin(-1).to(torch.uint8)
    assert torch.equal(context.codes, expected_codes)
    layer(x).sum().backward()
    layer(x*2).sum().backward()
    torch.testing.assert_close(layer.weight.grad,(x*3).expand_as(layer.weight),atol=8e-6,rtol=2e-6)
    assert layer.weight is original


def test_context_step_lifetime_fixed_scales_and_fresh_model_microbatch_graphs():
    model = ByteTransformer(ModelConfig(1,16,4))
    identities = {name:id(p) for name,p in model.named_parameters()}
    model.configure_quantization({'name':'row_fixed_max','rotation':'identity'})
    first = model.prepare_quantized_views(0,None)
    assert model.prepare_quantized_views(0,None) is first
    inputs = torch.tensor([[256,1,2]])
    positions = torch.tensor([[0,1,2]])
    for _ in range(2):
        model(inputs,positions).square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    runtime = {name:{'previous_scales':c.scales,'previous_zero_groups':c.scale_zero_groups} for name,c in first.items()}
    with torch.no_grad():
        for item in model.quantized_matrix_manifest():
            dict(model.named_parameters())[item['name']].mul_(3)
    model.clear_quantized_views()
    model.configure_quantization({'name':'row_fixed_max','rotation':'identity'},runtime=runtime)
    second = model.prepare_quantized_views(1,None)
    for name in first:
        torch.testing.assert_close(first[name].scales,second[name].scales,atol=0,rtol=0)
        assert second[name].metadata['scale_source'] == 'previous_fixed_scales'
        assert all(t.grad_fn is None and not t.requires_grad for t in second[name].numeric_tensors())
    assert identities == {name:id(p) for name,p in model.named_parameters()}


def test_padding_including_empty_row_has_finite_scored_gradients():
    model = ByteTransformer(ModelConfig(2,16,4))
    from ralup.data.blocks import make_block
    from ralup.data.batches import LogicalBatch
    blocks = [make_block('one',0,1,b'',bytes([3])),make_block('short',0,3,b'',bytes([1,2,3]))]
    batch = LogicalBatch(blocks,4)
    # Bounded tensor collation preserves each block's original tuple fields.
    inputs = torch.zeros((3,3),dtype=torch.int64)
    positions = torch.zeros_like(inputs)
    labels = torch.zeros_like(inputs)
    padding = torch.ones_like(inputs,dtype=torch.bool)
    selected = torch.zeros_like(padding)
    for row,block in enumerate(batch.block_slices):
        end = len(block.inputs)
        inputs[row,:end] = torch.tensor(block.inputs)
        positions[row,:end] = torch.tensor(block.positions)
        labels[row,:end] = torch.tensor(block.labels)
        padding[row,:end] = False
        selected[row,:end] = torch.tensor(block.score_mask)
    assert int(selected.sum()) == batch.target_count
    logits = model(inputs,positions,padding)
    loss = F.cross_entropy(logits[selected],labels[selected],reduction='sum')
    loss.backward()
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    torch.testing.assert_close(logits[0,:1],model(inputs[:1,:1],positions[:1,:1])[0],atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('config',[(0,16,4),(1,0,4),(1,16,3),(True,16,4)])
def test_invalid_model_geometry_rejected(config):
    with pytest.raises((ValueError,TypeError)):
        ModelConfig(*config)


@pytest.mark.parametrize('shape',[(0,3),(1,0),(1,257)])
def test_empty_or_overlength_forward_input_rejected(shape):
    model = ByteTransformer(ModelConfig(1,16,4))
    with pytest.raises(ValueError):
        model(torch.empty(shape,dtype=torch.int64),torch.empty(shape,dtype=torch.int64))


def test_evaluation_context_override_preserves_training_cache_and_rng():
    model = ByteTransformer(ModelConfig(1,16,4))
    model.configure_quantization({'name':'row_dynamic_max'})
    train = model.prepare_quantized_views(2,None)
    x = torch.tensor([[256,1,2]])
    positions = torch.tensor([[0,1,2]])
    before = model(x,positions).detach().clone()
    with torch.no_grad():
        for item in model.quantized_matrix_manifest():
            dict(model.named_parameters())[item['name']].mul_(2)
    rng_before = torch.get_rng_state().clone()
    final = {name:prepare_context(dict(model.named_parameters())[name],{'name':'row_dynamic_max'},3,None) for name in train}
    evaluated = model(x,positions,contexts=final)
    assert not torch.equal(before,evaluated)
    assert all(model.quantization_contexts[name] is train[name] for name in train)
    assert torch.equal(rng_before,torch.get_rng_state())
    torch.testing.assert_close(before,model(x,positions),atol=0,rtol=0)


def test_model_harness_records_counts_and_staged_remaining_work(tmp_path):
    from ralup.__main__ import main
    import json
    output = tmp_path/'model.json'
    assert main(['test-harness','--suite','model','--output',str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['status'] == 'pass'
    assert report['checks']['parameter_counts'] == [493056,3347456,19281920]
    assert report['scientific_gate'] is None
    assert report['remaining_harness'] == ['H05-H08; complete WP02 pending']


def test_model_parameters_remain_fp32_under_external_default_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = ByteTransformer(ModelConfig(1,16,4))
        assert {p.dtype for p in model.parameters()} == {torch.float32}
    finally:
        torch.set_default_dtype(previous)
