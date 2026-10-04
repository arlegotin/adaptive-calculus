"""Shared gather with differentiable selector reduction; independent sum fixtures."""
import pytest
import torch
from ralup.models.transformer import ByteTransformer, ModelConfig


def test_both_model_embeddings_use_shared_ordered_owner():
    model = ByteTransformer(ModelConfig(1,8,2))
    assert type(model.embedding).__name__ == 'OrderedEmbedding'
    assert type(model.position) is type(model.embedding)


def test_exact_gather_boundary_and_repeated_analytic_gradient():
    from ralup.models.embedding import OrderedEmbedding
    module = OrderedEmbedding(257,2,dtype=torch.float32)
    with torch.no_grad(): module.weight.copy_(torch.arange(514).reshape(257,2))
    indices = torch.tensor([[256,0,256,2]])
    output = module(indices)
    assert torch.equal(output, module.weight.detach()[indices])
    output.backward(torch.tensor([[[1.,2.],[3.,4.],[5.,6.],[7.,8.]]]))
    expected = torch.zeros(257,2)
    expected[256] = torch.tensor([6.,8.]); expected[0] = torch.tensor([3.,4.]); expected[2] = torch.tensor([7.,8.])
    assert torch.equal(module.weight.grad,expected)
    with pytest.raises((IndexError,RuntimeError)): module(torch.tensor([257]))
    with pytest.raises((IndexError,RuntimeError)): module(torch.tensor([-1]))


def test_higher_derivatives_finite_difference_and_functional_gradients():
    from ralup.models.embedding import ordered_embedding
    indices = torch.tensor([0,2,0])
    weight = torch.tensor([[.2,.4],[.6,.8],[1.,1.2]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda w: ordered_embedding(indices,w),(weight,))
    assert torch.autograd.gradgradcheck(lambda w: ordered_embedding(indices,w).square(),(weight,))
    loss = lambda w: ordered_embedding(indices,w).square().sum()
    gradient = torch.func.grad(loss)(weight)
    assert torch.allclose(gradient,torch.tensor([[.8,1.6],[0.,0.],[2.,2.4]],dtype=torch.float64))
    hessian = torch.func.jacrev(torch.func.grad(loss))(weight)
    assert torch.equal(hessian[0,0,0,0],torch.tensor(4.,dtype=torch.float64))
    assert torch.equal(hessian[1,0,1,0],torch.tensor(0.,dtype=torch.float64))
    # Actual wrong derivative: halve the upstream contribution to repeated
    # index0. The numerical derivative oracle must reject this same forward.
    class WrongDerivative(torch.autograd.Function):
        @staticmethod
        def forward(ctx,w):
            ctx.save_for_backward(indices)
            return w[indices]
        @staticmethod
        def backward(ctx,upstream):
            result=torch.stack(((upstream[0]+upstream[2])/2,torch.zeros_like(upstream[0]),upstream[1]))
            return result
    with pytest.raises(torch.autograd.gradcheck.__globals__['GradcheckError']):
        torch.autograd.gradcheck(WrongDerivative.apply,(weight,))



def test_functional_model_gradient_and_owner_rejection():
    from ralup.state import _validate_model_owners, IntegrityError
    model = ByteTransformer(ModelConfig(1,8,2))
    inputs = torch.tensor([[256,0,0]])
    positions = torch.tensor([[0,1,2]])
    parameters = dict(model.named_parameters())
    gradients = torch.func.grad(lambda p: torch.func.functional_call(model,p,(inputs,positions)).square().sum())(parameters)
    assert torch.isfinite(gradients['embedding.weight']).all()
    assert torch.equal(gradients['embedding.weight'][1],torch.zeros(8))
    _validate_model_owners(model)
    model.position = torch.nn.Embedding(256,8,dtype=torch.float32)
    with pytest.raises(IntegrityError,match='owner'): _validate_model_owners(model)


@pytest.mark.parametrize('device',['cpu','mps'])
@pytest.mark.parametrize('vocabulary',[257,256])
def test_full_shape_repeated_gradient_exact_same_device_and_parity(device,vocabulary):
    import os
    if device=='mps':
        if not torch.backends.mps.is_available(): pytest.skip('Actual MPS unavailable; skip is not acceptance')
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    from ralup.models.embedding import ordered_embedding
    generator=torch.Generator().manual_seed(901)
    weight=torch.randn(vocabulary,128,generator=generator)
    indices=torch.randint(vocabulary,(64,256),generator=generator)
    upstream=torch.randn(64,256,128,generator=generator)
    indices[0,0]=vocabulary-1
    cpu=weight.clone().requires_grad_()
    expected=torch.autograd.grad(ordered_embedding(indices,cpu),cpu,upstream)[0]
    gradients=[]
    for _ in range(12):
        local=weight.to(device).clone().requires_grad_()
        output=ordered_embedding(indices.to(device),local)
        assert torch.equal(output.detach().cpu(),weight[indices])
        gradient=torch.autograd.grad(output,local,upstream.to(device))[0].cpu()
        gradients.append(gradient)
    assert all(torch.equal(gradients[0],g) for g in gradients[1:])
    assert torch.allclose(gradients[0],expected,atol=1e-4,rtol=1e-4)
