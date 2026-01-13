"""
Hybrid forecasting ensemble demo.

Blends a tree model forecast with a neural forecast and evaluates MAE.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def to_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def weighted_ensemble(tree_pred: np.ndarray, neural_pred: np.ndarray, weight: float) -> np.ndarray:
    if tree_pred.shape != neural_pred.shape:
        raise ValueError("Predictions must have the same shape")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Weight must be between 0 and 1")
    return weight * tree_pred + (1.0 - weight) * neural_pred


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def run_demo() -> None:
    actuals = to_array([102, 98, 110, 105, 97, 120, 115])

    # Simulated outputs from a tree model and a neural model
    tree_pred = to_array([101, 100, 108, 104, 99, 118, 113])
    neural_pred = to_array([104, 96, 112, 106, 95, 121, 116])

    blended = weighted_ensemble(tree_pred, neural_pred, weight=0.6)

    tree_mae = mean_absolute_error(actuals, tree_pred)
    neural_mae = mean_absolute_error(actuals, neural_pred)
    blended_mae = mean_absolute_error(actuals, blended)

    print("Tree MAE:", round(tree_mae, 2))
    print("Neural MAE:", round(neural_mae, 2))
    print("Blended MAE:", round(blended_mae, 2))
    print("Blended forecast:", blended.round(2).tolist())


if __name__ == "__main__":
    run_demo()
