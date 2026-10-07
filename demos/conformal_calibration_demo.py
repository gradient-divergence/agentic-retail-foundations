"""
Conformal calibration demo for forecast intervals.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import ceil

import numpy as np


def to_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def conformal_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_new: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    if y_true.shape != y_pred.shape:
        raise ValueError("Calibration arrays must have the same shape")
    if y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("Calibration arrays must be non-empty and one-dimensional")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    residuals = np.abs(y_true - y_pred)
    rank = ceil((len(residuals) + 1) * (1 - alpha))
    q = np.partition(residuals, rank - 1)[rank - 1] if rank <= len(residuals) else np.inf
    lower = y_pred_new - q
    upper = y_pred_new + q
    return lower, upper


def run_demo() -> None:
    y_true = to_array([100, 98, 104, 97, 110, 108, 105, 99, 103])
    y_pred = to_array([101, 99, 103, 98, 109, 110, 104, 100, 104])

    y_pred_new = to_array([102, 101, 106])
    lower, upper = conformal_interval(y_true, y_pred, y_pred_new, alpha=0.1)

    print("Point forecasts:", y_pred_new.tolist())
    print("Lower bounds:", lower.round(2).tolist())
    print("Upper bounds:", upper.round(2).tolist())


if __name__ == "__main__":
    run_demo()
