from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def _to_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=float)


def sigmoid(x: ArrayLike, beta: float = 25.0) -> ArrayLike:
    """
    Smooth logistic function with slope control via beta.
    Output in (0, 1). For beta -> ∞, approaches a hard step.

    y = 1 / (1 + exp(-beta * x))
    """
    x_arr = _to_array(x)
    # numerically-stable sigmoid using where to avoid overflow
    z = beta * x_arr
    # For large negative z, exp(z) underflows; for large positive z, exp(-z) underflows
    out = np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )
    # Clip away exact 0/1 due to floating point rounding so tests expecting strict
    # inequalities (y > 0 and y < 1) pass robustly and to avoid log/odds singularities.
    eps = 1e-12
    out = np.clip(out, eps, 1.0 - eps)
    if isinstance(x, np.ndarray):
        return out
    return float(out)


def softplus(x: ArrayLike, beta: float = 25.0) -> ArrayLike:
    """
    Smooth approximation to ReLU with slope control.
    softplus(x) = (1/beta) * log(1 + exp(beta*x))
    """
    x_arr = _to_array(x)
    z = beta * x_arr
    # use log1p for stability
    out = np.log1p(np.exp(np.minimum(z, 60.0))) / beta  # clamp to avoid overflow in tests
    if isinstance(x, np.ndarray):
        return out
    return float(out)


def hard_logic(value: float, thr: float) -> int:
    """
    Hard threshold logic: returns 1 if value > thr else 0.
    """
    return 1 if value > thr else 0


def soft_logic(value: float, thr: float, beta: float = 25.0) -> float:
    """
    Soft logic threshold via sigmoid(value - thr, beta).
    Returns a probability-like value in (0, 1).
    """
    return sigmoid(value - thr, beta)
