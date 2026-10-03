"""Detached FP32 group scales and independently checked S14 Gaussian refit.

Analytic moments and bracketed log-scale search implement the project math;
no upstream quantizer code is copied. Floating fitting/integration is a CPU
float64 initialization oracle, never parameter-by-parameter host arithmetic.
"""
from functools import lru_cache
import math

import torch
from torch import Tensor

from .formats import E2M1_VALUES, _preserved_ratio
from .fp32 import bits as fp32_bits, divide_six, multiply_rescaled

RECIPES = ('row_fixed_max', 'row_dynamic_max', 'tensor_fixed_max',
           'tensor_dynamic_max', 'quest_fp4_row_reference', 'quest_fp4_tensor_reference')


def _cells(a):
    if not math.isfinite(a) or a <= 0:
        raise ValueError('Gaussian maximum level must be finite and positive')
    levels = [a*c/6 for c in E2M1_VALUES]
    bounds = [-math.inf] + [(x+y)/2 for x, y in zip(levels, levels[1:])] + [math.inf]
    return zip(levels, bounds[:-1], bounds[1:])


def _moments(lower, upper):
    def phi(z):
        return math.exp(-z*z/2)/math.sqrt(2*math.pi)
    def cdf(z):
        return .5*math.erfc(-z/math.sqrt(2))
    # Avoid subtracting two numbers rounded close to one in the positive tail.
    m0 = cdf(-lower)-cdf(-upper) if lower >= 0 else cdf(upper)-cdf(lower)
    p_lower, p_upper = phi(lower), phi(upper)
    m1 = p_lower-p_upper
    m2 = m0 + (lower*p_lower if math.isfinite(lower) else 0.) - (upper*p_upper if math.isfinite(upper) else 0.)
    return m0, m1, m2


def gaussian_e2m1_mse(a: float) -> float:
    """E[(Z-nearest(a*C/6))²], Z standard normal, including infinite tails."""
    terms = []
    for q, lower, upper in _cells(a):
        m0, m1, m2 = _moments(lower, upper)
        terms.append(m2-2*q*m1+q*q*m0)
    return math.fsum(terms)


def _derivative(a):
    terms = []
    for c, (q, lower, upper) in zip(E2M1_VALUES, _cells(a)):
        m0, m1, _ = _moments(lower, upper)
        terms.append(2*(c/6)*(q*m0-m1))
    return math.fsum(terms)


def fit_gaussian_e2m1() -> dict:
    """Deterministic expanding bracket + Brent minimization in log(a).

    A stationary-root refinement in the same bracket removes flat-objective
    minimizer position noise; deterministic direct cell quadrature independently
    checks the resulting objective. Returns fresh metadata, with no tensor state.
    """
    from scipy.integrate import quad
    from scipy.optimize import brentq, minimize_scalar

    bracket = [math.log(.5), 0., math.log(2.)]
    history = []
    for _ in range(64):
        values = [gaussian_e2m1_mse(math.exp(x)) for x in bracket]
        history.append({'a': [math.exp(x) for x in bracket], 'objectives': values})
        if values[1] < values[0] and values[1] < values[2]:
            break
        if values[2] < values[1]:
            bracket = [bracket[1], bracket[2], bracket[2]+math.log(2.)]
        elif values[0] < values[1]:
            bracket = [bracket[0]-math.log(2.), bracket[0], bracket[1]]
        else:
            raise ValueError('Gaussian fit could not obtain a strict bracket')
    else:
        raise ValueError('Gaussian fit exhausted bracket expansion')
    optimum = minimize_scalar(lambda x: gaussian_e2m1_mse(math.exp(x)), bracket=tuple(bracket),
                              method='brent', options={'xtol': 1e-13, 'maxiter': 300})
    if not optimum.success or not bracket[0] < optimum.x < bracket[2]:
        raise ValueError('Gaussian bracketed minimization failed')
    if not _derivative(math.exp(bracket[0])) < 0 < _derivative(math.exp(bracket[2])):
        raise ValueError('Gaussian stationary root is not bracketed')
    root = brentq(lambda x: _derivative(math.exp(x)), bracket[0], bracket[2], xtol=5e-15, rtol=1e-14)
    a = math.exp(root)
    objective = gaussian_e2m1_mse(a)
    integrals = [quad(lambda z: (z-q)**2*math.exp(-z*z/2)/math.sqrt(2*math.pi), lower, upper,
                      epsabs=2e-14, epsrel=2e-13, limit=300)
                 for q, lower, upper in _cells(a)]
    integrated = math.fsum(item[0] for item in integrals)
    residual = abs(objective-integrated)
    if residual > 2e-12 or abs(_derivative(a)) > 2e-9:
        raise ValueError('Gaussian fit failed independent integration or stationarity')
    return {'a': a, 'scale_constant': a/6, 'objective': objective,
            'derivative_residual': _derivative(a), 'integration_objective': integrated,
            'integration_residual': residual, 'integration_error_estimate': math.fsum(item[1] for item in integrals),
            'bracket_history': history, 'bracket_log_a': bracket, 'bracket_objectives': values,
            'minimizer_a': math.exp(optimum.x), 'minimizer_objective': float(optimum.fun),
            'minimizer_iterations': int(optimum.nit),
            'conventions': {'dtype': 'CPU float64', 'distribution': 'standard normal',
                            'cells': 'adjacent midpoints with infinite exterior tails; one numerical zero',
                            'moments': 'M0=Phi(u)-Phi(l); M1=phi(l)-phi(u); M2=M0+l*phi(l)-u*phi(u)',
                            'tails': 'positive intervals use survival-CDF differences; infinite z*phi(z)=0',
                            'search': 'expanding strict bracket; Brent log(a) minimization; bracketed stationary refinement',
                            'integration': 'independent direct deterministic QUADPACK cell squared-error',
                            'quad_epsabs': 2e-14, 'quad_epsrel': 2e-13, 'quad_limit': 300}}


@lru_cache(maxsize=1)
def _gaussian_scale_constant():
    return fit_gaussian_e2m1()['scale_constant']


def grouping_for(recipe: str) -> str:
    if recipe not in RECIPES:
        raise ValueError(f'Unknown scale recipe: {recipe}')
    return 'whole_tensor' if 'tensor' in recipe else 'last_axis_row'


def group_zero_flags(u: Tensor, recipe: str) -> Tensor:
    dims = tuple(range(u.ndim)) if grouping_for(recipe) == 'whole_tensor' else (-1,)
    bits = u.detach().contiguous().view(torch.int32) & 0x7fffffff
    return bits.amax(dim=dims, keepdim=True) == 0


@torch.no_grad()
def compute_scales(u: Tensor, recipe: str, previous: Tensor | None) -> Tensor:
    """Caller owns fixed-scale lifetime; pass the prior scale on later steps.

    With no previous scale, fixed max initializes from this supplied warm-start
    U. Dynamic/Gaussian recipes reject previous scales to prevent accidental
    freezing. Zero groups alone use sentinel one. Stable RMS first divides by
    group maximum to avoid squaring overflow/underflow. A scale not representable
    as a positive FP32 number raises explicitly; representable subnormals
    use device integer bits and normal-range rescaling.
    """
    grouping = grouping_for(recipe)
    if u.dtype != torch.float32:
        raise TypeError('Scale inputs must be float32')
    if u.ndim < 1 or u.numel() == 0:
        raise ValueError('Scale inputs require nonempty groups')
    dims = tuple(range(u.ndim)) if grouping == 'whole_tensor' else (-1,)
    bits = u.detach().contiguous().view(torch.int32) & 0x7fffffff
    if not bool((bits < 0x7f800000).all()):
        raise ValueError('Scale inputs must be finite')
    maximum_bits = bits.amax(dim=dims, keepdim=True)
    zero = maximum_bits == 0
    maximum = maximum_bits.contiguous().view(torch.float32)
    if previous is not None:
        if 'fixed' not in recipe:
            raise ValueError('Previous scales are only valid for fixed recipes')
        if previous.dtype != torch.float32 or previous.device != u.device:
            raise ValueError('Previous scales must be float32 on the input device')
        if previous.shape != maximum.shape:
            raise ValueError('Previous scale shape must match the declared grouping')
        previous_bits = previous.detach().contiguous().view(torch.int32)
        if not bool(((previous_bits > 0) & (previous_bits < 0x7f800000)).all()):
            raise ValueError('Previous scales must be finite and positive')
        return previous.detach().clone()
    safe_maximum = torch.where(zero, torch.ones_like(maximum), maximum)
    if recipe.startswith('quest'):
        normalized = (_preserved_ratio(fp32_bits(u), fp32_bits(safe_maximum))
                      if bool(((bits > 0) & (bits < 0x800000)).any()) else u / safe_maximum)
        rms_normalized = normalized.square().mean(dim=dims, keepdim=True).sqrt()
        factor = rms_normalized * _gaussian_scale_constant()
        # Sparsity can make this product subnormal for any normal maximum.
        # Preserve the computed product directly; input-magnitude cutoffs cannot
        # determine its representability across arbitrary group sizes.
        scale = multiply_rescaled(safe_maximum, factor)
    else:
        scale = divide_six(safe_maximum)
    scale = torch.where(zero, torch.ones_like(scale), scale)
    scale_bits = scale.contiguous().view(torch.int32)
    if bool(((~zero) & (scale_bits == 0)).any()):
        raise ValueError('Nonzero group scale underflow in float32 storage')
    if not bool(((scale_bits > 0) & (scale_bits < 0x7f800000)).all()):
        raise ValueError('Group scale overflow in float32 storage')
    return scale.detach().clone()
