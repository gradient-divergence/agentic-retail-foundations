"""
Constrained pricing update demo with a simple Lagrangian penalty.
"""

from __future__ import annotations

import numpy as np


def constrained_update(
    price: float,
    gradient: float,
    cost: float,
    cost_limit: float,
    lagrange: float,
    cost_gradient: float,
    step_size: float = 0.05,
    lagrange_lr: float = 0.1,
    price_min: float = 10.0,
    price_max: float = 40.0,
) -> tuple[float, float]:
    lagrange = max(0.0, lagrange + lagrange_lr * (cost - cost_limit))
    adjusted_grad = gradient - lagrange * cost_gradient
    updated = price + step_size * adjusted_grad
    updated = float(np.clip(updated, price_min, price_max))
    return updated, lagrange


def run_demo() -> None:
    price = 25.0
    lagrange = 0.0

    gradients = [200.0, -160.0, 0.0, 0.0]  # Synthetic stress steps: clipping, active cost, then recovery.
    cost_limit = 0.6

    for step, grad in enumerate(gradients, start=1):
        cost = ((price - 24.0) / 2.0) ** 2
        cost_gradient = (price - 24.0) / 2.0
        price, lagrange = constrained_update(
            price=price,
            gradient=grad,
            cost=cost,
            cost_gradient=cost_gradient,
            cost_limit=cost_limit,
            lagrange=lagrange,
            price_min=18.0,
            price_max=32.0,
        )
        evaluated_cost = ((price - 24.0) / 2.0) ** 2
        print(f"Step {step}: price={price:.2f}, cost={evaluated_cost:.2f}, lambda={lagrange:.2f}")


if __name__ == "__main__":
    run_demo()
