"""S11 reference byte decoder and explicit S15 logical-step context boundary."""
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor, nn

from ..quantization.context import QuantizationContext, prepare_context
from .linear import QuantizedLinear

MATRIX_NAMES = ('q', 'k', 'v', 'attention_output', 'up', 'down')


@dataclass(frozen=True)
class ModelConfig:
    layers: int
    width: int
    heads: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.layers, self.width, self.heads)):
            raise ValueError('Model geometry requires positive integers')
        if self.width % self.heads:
            raise ValueError('Model width must be divisible by head count')


class _Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.attention_norm = nn.LayerNorm(config.width, eps=1e-5, dtype=torch.float32)
        self.mlp_norm = nn.LayerNorm(config.width, eps=1e-5, dtype=torch.float32)
        for name in MATRIX_NAMES[:4]:
            setattr(self, name, QuantizedLinear(config.width, config.width))
        self.up = QuantizedLinear(config.width, 4*config.width)
        self.down = QuantizedLinear(4*config.width, config.width)
        self.gelu = nn.GELU(approximate='none')

    def forward(self, x: Tensor, allowed: Tensor, contexts: Mapping[str, QuantizationContext], prefix: str) -> Tensor:
        def project(name, value):
            return getattr(self, name)(value, contexts.get(f'{prefix}.{name}.weight'))
        batch, time, width = x.shape
        normalized = self.attention_norm(x)
        q, k, v = (project(name, normalized).reshape(batch,time,self.heads,width//self.heads).transpose(1,2)
                   for name in ('q','k','v'))
        scores = (q@k.transpose(-2,-1))/math.sqrt(width//self.heads)
        probabilities = scores.masked_fill(~allowed, float('-inf')).softmax(dim=-1)
        attended = (probabilities@v).transpose(1,2).reshape(batch,time,width)
        x = x + project('attention_output', attended)
        return x + project('down', self.gelu(project('up',self.mlp_norm(x))))


class ByteTransformer(nn.Module):
    """Full-precision by default; configure then prepare for quantized forwards.

    The model owns only the current numeric context cache, never histories or
    optimizer moments. Runtime tensor handoffs are separate from ModelConfig.
    Forward's optional context mapping provides read-only evaluation/functional
    parameter contexts without changing the training cache or RNG streams.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(257, config.width, dtype=torch.float32)
        self.position = nn.Embedding(256, config.width, dtype=torch.float32)
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width, eps=1e-5, dtype=torch.float32)
        self.output = nn.Linear(config.width,257,bias=False,dtype=torch.float32)
        for name, parameter in self.named_parameters():
            if parameter.ndim == 2:
                std = .02/math.sqrt(2*config.layers) if name.endswith(('attention_output.weight','down.weight')) else .02
                nn.init.normal_(parameter,std=std)
        self._recipe: dict | None = None
        self._runtime: Mapping[str, dict] = {}
        self._contexts: dict[str, QuantizationContext] = {}
        self._context_step: int | None = None

    @property
    def quantization_contexts(self) -> Mapping[str, QuantizationContext]:
        return MappingProxyType(self._contexts)

    def quantized_matrix_manifest(self) -> list[dict]:
        return [{'name':f'blocks.{index}.{name}.weight',
                 'shape':list(getattr(block,name).weight.shape),
                 'group':'internal_projection', 'coordinates':'original_W'}
                for index,block in enumerate(self.blocks) for name in MATRIX_NAMES]

    def configure_quantization(self, recipe: dict, *, runtime: Mapping[str, dict] | None = None) -> None:
        if not isinstance(recipe,dict) or set(recipe)-{'name','rotation','backward'} or 'name' not in recipe:
            raise ValueError('Quantization recipe requires name and only scalar rotation/backward settings')
        runtime = {} if runtime is None else runtime
        names = {item['name'] for item in self.quantized_matrix_manifest()}
        if set(runtime)-names:
            raise ValueError('Runtime handoff contains unknown matrix names')
        for state in runtime.values():
            if set(state)-{'previous_scales','previous_zero_groups','warm_start','codes'}:
                raise ValueError('Unknown runtime quantizer state')
        self._recipe = dict(recipe)
        self._runtime = {name:dict(state) for name,state in runtime.items()}
        self.clear_quantized_views()

    def prepare_quantized_views(self, step: int, rng: object) -> dict[str, QuantizationContext]:
        if type(step) is not int or step < 0:
            raise ValueError('Logical step must be a nonnegative integer')
        if self._recipe is None:
            raise ValueError('Configure quantization before preparing views')
        if self._context_step == step:
            return self._contexts
        parameters = dict(self.named_parameters())
        # Publish atomically: failed preparation cannot leave a partial cache.
        contexts = {item['name']:prepare_context(parameters[item['name']],
                    {**self._recipe, **self._runtime.get(item['name'],{})}, step, rng)
                    for item in self.quantized_matrix_manifest()}
        self._contexts = contexts
        self._context_step = step
        return contexts

    def clear_quantized_views(self) -> None:
        """Release current numeric cache when a transaction/parameter device ends."""
        self._contexts = {}
        self._context_step = None

    def forward(self, inputs: Tensor, positions: Tensor, padding_mask: Tensor | None = None,
                *, contexts: Mapping[str, QuantizationContext] | None = None) -> Tensor:
        """Return [batch,time,257]; padding_mask True denotes padded tokens.

        Padded query rows attend to their own diagonal to avoid an all-masked
        softmax. Real queries retain the exact causal/key-padding reference mask.
        Score masks remain a caller responsibility, including optimizer empties.
        """
        if inputs.ndim != 2 or not inputs.shape[0] or not 0 < inputs.shape[1] <= 256:
            raise ValueError('Inputs require nonempty [batch,time] with time <= 256')
        if positions.shape != inputs.shape or inputs.dtype != torch.int64 or positions.dtype != torch.int64:
            raise ValueError('Inputs/positions must be matching int64 tensors')
        if inputs.device != positions.device:
            raise ValueError('Inputs/positions require the same device')
        if contexts is None:
            if self._recipe is not None and self._context_step is None:
                raise ValueError('Prepare numeric views before a quantized forward')
            contexts = self._contexts
        if contexts and set(contexts) != {item['name'] for item in self.quantized_matrix_manifest()}:
            raise ValueError('Context override requires every internal matrix exactly once')
        time = inputs.shape[1]
        causal = torch.ones((time,time),dtype=torch.bool,device=inputs.device).tril()
        allowed = causal[None,None,:,:]
        if padding_mask is not None:
            if padding_mask.shape != inputs.shape or padding_mask.dtype != torch.bool or padding_mask.device != inputs.device:
                raise ValueError('Padding mask requires matching boolean shape/device')
            allowed = allowed & ~padding_mask[:,None,None,:]
            # Safe diagonal is used only for padded queries, never scored ones.
            diagonal = torch.eye(time,dtype=torch.bool,device=inputs.device)[None,None,:,:]
            allowed = torch.where(padding_mask[:,None,:,None], diagonal, allowed)
        x = self.embedding(inputs) + self.position(positions)
        for index,block in enumerate(self.blocks):
            x = block(x,allowed,contexts,f'blocks.{index}')
        return self.output(self.final_norm(x))
