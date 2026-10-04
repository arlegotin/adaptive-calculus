"""N01 shared gather and selector-GEMM derivative, on the weight's backend.

The numerical realization is identified by these source bytes in strict producer
archives. No checkpoint callback or method-specific derivative is accepted.
Selector allocation belongs to every backward: N*V elements in gradient dtype,
plus the equality selector. Same-device repeatability is empirically scoped.
"""
import torch
from torch import nn
from torch.nn import functional as F

PRIMITIVE_ID = 'gather-selector-transpose-gemm-v1'


class _OrderedGather(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(indices, weight):
        return F.embedding(indices, weight)

    @staticmethod
    def setup_context(ctx, inputs, output):
        indices, weight = inputs
        ctx.save_for_backward(indices)
        ctx.vocabulary = weight.shape[0]
        ctx.width = weight.shape[1]

    @staticmethod
    def backward(ctx, upstream):
        (indices,) = ctx.saved_tensors
        # Constant selector, constructed on the same backend as the gathered
        # tensor. Keep upstream attached: higher derivatives must see this GEMM.
        selector = (indices.reshape(-1,1) == torch.arange(ctx.vocabulary,device=indices.device)).to(upstream.dtype)
        return None, selector.transpose(0,1) @ upstream.reshape(-1,ctx.width)


def ordered_embedding(indices, weight):
    """Exact native gather forward; analytic indexed sum via selector GEMM."""
    return _OrderedGather.apply(indices,weight)


class OrderedEmbedding(nn.Embedding):
    """Constructor-owned dense embedding with the shared N01 derivative.

    ByteTransformer owns the supported dense/default settings; strict module
    validation rejects modified fields or a substituted native embedding.
    """
    def forward(self, indices):
        return ordered_embedding(indices,self.weight)
