from __future__ import annotations

import random
from collections.abc import Callable, Iterable

from pydantic import BaseModel, Field


class InventoryRiskSimConfig(BaseModel):
    horizon_weeks: int = Field(default=12, ge=1)
    initial_inventory: int = Field(default=120, ge=0)
    reorder_point: int = Field(default=40, ge=0)
    reorder_qty: int = Field(default=60, ge=1)
    lead_time_weeks: int = Field(default=2, ge=0)
    base_demand: int = Field(default=10, ge=0)
    promo_weeks: list[int] = Field(default_factory=lambda: [4, 8])
    promo_spike_multiplier: float = Field(default=1.8, gt=0)
    disruption_probability: float = Field(default=0.2, ge=0, le=1)
    disruption_delay_weeks: int = Field(default=2, ge=0)
    unit_margin: float = Field(default=5.0, ge=0)
    stockout_penalty: float = Field(default=8.0, ge=0)


class InventoryState(BaseModel):
    week: int
    on_hand: int
    pipeline: int


class SimulationMetrics(BaseModel):
    service_level: float
    total_margin: float
    stockout_weeks: int
    total_demand: int
    lost_sales: int


PolicyFn = Callable[[InventoryState], int]


class InventoryRiskSimulator:
    def __init__(self, config: InventoryRiskSimConfig, seed: int = 7) -> None:
        self.config = config
        self.rng = random.Random(seed)

    def _demand_for_week(self, week: int) -> int:
        base = self.config.base_demand
        if week in self.config.promo_weeks:
            base = int(round(base * self.config.promo_spike_multiplier))
        noise = self.rng.randint(-2, 2)
        return max(0, base + noise)

    def _arrival_week(self, week: int) -> int:
        delay = self.config.lead_time_weeks
        if self.rng.random() < self.config.disruption_probability:
            delay += self.config.disruption_delay_weeks
        return week + delay

    def run(self, policy: PolicyFn | None = None) -> SimulationMetrics:
        on_hand = self.config.initial_inventory
        pipeline: list[tuple[int, int]] = []
        total_demand = 0
        lost_sales = 0
        stockout_weeks = 0
        total_margin = 0.0

        def default_policy(state: InventoryState) -> int:
            return self.config.reorder_qty if state.on_hand <= self.config.reorder_point else 0

        policy_fn = policy or default_policy

        for week in range(1, self.config.horizon_weeks + 1):
            arrivals = [qty for arrival_week, qty in pipeline if arrival_week <= week]
            on_hand += sum(arrivals)
            pipeline = [(w, q) for (w, q) in pipeline if w > week]

            demand = self._demand_for_week(week)
            total_demand += demand
            sales = min(on_hand, demand)
            on_hand -= sales

            if demand > sales:
                lost_sales += demand - sales
                stockout_weeks += 1

            total_margin += sales * self.config.unit_margin
            if demand > sales:
                total_margin -= (demand - sales) * self.config.stockout_penalty

            state = InventoryState(week=week, on_hand=on_hand, pipeline=sum(q for _, q in pipeline))
            order_qty = policy_fn(state)
            if order_qty > 0:
                pipeline.append((self._arrival_week(week), order_qty))

        service_level = 1.0 if total_demand == 0 else 1 - (lost_sales / total_demand)
        return SimulationMetrics(
            service_level=round(service_level, 4),
            total_margin=round(total_margin, 2),
            stockout_weeks=stockout_weeks,
            total_demand=total_demand,
            lost_sales=lost_sales,
        )


def evaluate_policies(
    config: InventoryRiskSimConfig, policies: Iterable[PolicyFn]
) -> list[SimulationMetrics]:
    return [InventoryRiskSimulator(config).run(policy) for policy in policies]
