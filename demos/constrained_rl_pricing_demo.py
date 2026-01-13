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
    step_size: float = 0.05,
    lagrange_lr: float = 0.1,
    price_min: float = 10.0,
    price_max: float = 40.0,
) -> tuple[float, float]:
    if cost > cost_limit:
        lagrange += lagrange_lr * (cost - cost_limit)

    adjusted_grad = gradient - lagrange
    updated = price + step_size * adjusted_grad
    updated = float(np.clip(updated, price_min, price_max))
    return updated, lagrange


def run_demo() -> None:
    price = 25.0
    lagrange = 0.0

    gradients = [0.8, 0.6, 0.4, 0.2]
    costs = [0.3, 0.5, 0.8, 0.4]
    cost_limit = 0.6

    for step, (grad, cost) in enumerate(zip(gradients, costs, strict=True), start=1):
        price, lagrange = constrained_update(
            price=price,
            gradient=grad,
            cost=cost,
            cost_limit=cost_limit,
            lagrange=lagrange,
            price_min=18.0,
            price_max=32.0,
        )
        print(f"Step {step}: price={price:.2f}, cost={cost:.2f}, lambda={lagrange:.2f}")


if __name__ == "__main__":
    run_demo()
