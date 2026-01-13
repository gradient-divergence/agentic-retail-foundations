"""
Conformal calibration demo for forecast intervals.
"""

from __future__ import annotations

from collections.abc import Iterable

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
    residuals = np.abs(y_true - y_pred)
    q = np.quantile(residuals, 1.0 - alpha)
    lower = y_pred_new - q
    upper = y_pred_new + q
    return lower, upper


def run_demo() -> None:
    y_true = to_array([100, 98, 104, 97, 110, 108, 105, 99])
    y_pred = to_array([101, 99, 103, 98, 109, 110, 104, 100])

    y_pred_new = to_array([102, 101, 106])
    lower, upper = conformal_interval(y_true, y_pred, y_pred_new, alpha=0.1)

    print("Point forecasts:", y_pred_new.tolist())
    print("Lower bounds:", lower.round(2).tolist())
    print("Upper bounds:", upper.round(2).tolist())


if __name__ == "__main__":
    run_demo()
