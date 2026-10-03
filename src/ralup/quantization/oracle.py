"""Independent CPU float64 all-distance E2M1 reference."""
import numpy as np

from .formats import E2M1_EVEN, E2M1_VALUES


def e2m1_oracle(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Nearest E2M1 value in float64 scaled coordinates, then reconstruct.

    Compare all 15 distances; ties use actual significand parity. This CPU
    reference deliberately allocates [*broadcast_shape,15]. It does not emulate
    rounded FP32 division. Finite scaled overflow saturates; physical float64
    reconstruction overflow is a named storage error.
    """
    x = np.asarray(x, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('E2M1 rejects nonfinite original inputs')
    if not (np.isfinite(scale) & (scale > 0)).all():
        raise ValueError('E2M1 scales must be finite and positive')
    with np.errstate(over='ignore', under='ignore'):
        # Clipping outside the finite codebook preserves nearest saturation and
        # avoids meaningless distance overflow for very large finite inputs.
        scaled = np.clip(x / scale, -6., 6.)
    levels = np.asarray(E2M1_VALUES, dtype=np.float64)
    distances = np.abs(scaled[..., None] - levels)
    nearest = distances == distances.min(axis=-1, keepdims=True)
    even_nearest = nearest & np.asarray(E2M1_EVEN)
    indices = np.where(even_nearest.any(axis=-1), even_nearest.argmax(axis=-1), distances.argmin(axis=-1))
    with np.errstate(over='ignore', under='ignore'):
        values = levels[indices] * scale
    if not np.isfinite(values).all():
        raise ValueError('E2M1 reconstruction overflow in float64 storage')
    return np.where(values == 0, 0., values)
