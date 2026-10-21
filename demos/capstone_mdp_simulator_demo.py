"""Synthetic weekly simulator; margins, costs, and rewards are in USD."""

# region book:capstone-mdp-imports
import random
from enum import Enum

from pydantic import BaseModel


class SupplierStatus(str, Enum):
    STABLE = "stable"
    DELAYED = "delayed"


class Action(str, Enum):
    WAIT = "wait"
    REORDER = "reorder"
    MESSAGE_CUSTOMERS = "message_customers"
    PRICE_PROTECT = "price_protect"


class InventoryState(BaseModel):
    week: int
    inventory_on_hand: int
    pipeline: list[tuple[int, int]]  # (remaining lead time in weeks, quantity)
    risk_level: str
    supplier_status: SupplierStatus
    promo_active: bool


class StepInfo(BaseModel):
    demand: float
    sales: float
    stockouts: float
    contribution: float
    holding_cost: float
    penalty: float
    action_cost: float


# endregion book:capstone-mdp-imports


# region book:capstone-mdp-env-init
class InventoryRiskMDP:
    """Minimal MDP simulator for the capstone inventory risk workflow."""

    def __init__(
        self,
        horizon_weeks: int = 8,
        initial_inventory: int = 120,
        reorder_qty: int = 80,
        reorder_lead_time: int = 2,
        base_demand: int = 30,
        promo_lift: float = 0.25,
        margin_per_unit: float = 8.0,
        holding_cost_per_unit: float = 0.3,
        stockout_penalty: float = 6.0,
        price_protect_cost: float = 1.5,
        message_demand_reduction: float = 0.1,
        promo_weeks: tuple[int, ...] = (3, 4),
    ):
        self.horizon_weeks = horizon_weeks
        self.initial_inventory = initial_inventory
        self.reorder_qty = reorder_qty
        self.reorder_lead_time = reorder_lead_time
        self.base_demand = base_demand
        self.promo_lift = promo_lift
        self.margin_per_unit = margin_per_unit
        self.holding_cost_per_unit = holding_cost_per_unit
        self.stockout_penalty = stockout_penalty
        self.price_protect_cost = price_protect_cost
        self.message_demand_reduction = message_demand_reduction
        self.promo_weeks = set(promo_weeks)
        self.state = self.reset()

    def reset(self) -> InventoryState:
        self.state = InventoryState(
            week=0,
            inventory_on_hand=self.initial_inventory,
            pipeline=[],
            risk_level=self._compute_risk(self.initial_inventory, SupplierStatus.STABLE),
            supplier_status=SupplierStatus.STABLE,
            promo_active=0 in self.promo_weeks,
        )
        return self.state

    # endregion book:capstone-mdp-env-init

    # region book:capstone-mdp-step-dynamics
    def step(self, action: Action) -> tuple[InventoryState, float, bool, StepInfo]:
        if self.state.week >= self.horizon_weeks:
            raise RuntimeError("Episode is complete; call reset() before stepping again.")
        # Deliver any inbound inventory
        delivered = 0
        new_pipeline = []
        for eta, qty in self.state.pipeline:
            if eta <= 1:
                delivered += qty
            else:
                new_pipeline.append((eta - 1, qty))
        inventory = self.state.inventory_on_hand + delivered

        # Update supplier status stochastically
        if random.random() < 0.1:
            supplier_status = SupplierStatus.DELAYED
        elif random.random() < 0.2:
            supplier_status = SupplierStatus.STABLE
        else:
            supplier_status = self.state.supplier_status

        # Apply promo calendar
        promo_active = self.state.week in self.promo_weeks

        # Action effects
        action_cost = 0.0
        margin = self.margin_per_unit
        demand_multiplier = 1.0
        stockout_penalty_multiplier = 1.0

        if action == Action.REORDER:
            lead_time = self.reorder_lead_time + (self.state.supplier_status == SupplierStatus.DELAYED)
            new_pipeline.append((lead_time, self.reorder_qty))
            action_cost = 20.0
        elif action == Action.PRICE_PROTECT:
            margin -= self.price_protect_cost
            demand_multiplier += 0.1
        elif action == Action.MESSAGE_CUSTOMERS:
            demand_multiplier -= self.message_demand_reduction
            stockout_penalty_multiplier = 0.8

        # endregion book:capstone-mdp-step-dynamics

        # region book:capstone-mdp-step-reward
        # Demand and sales
        promo_multiplier = 1.0 + (self.promo_lift if promo_active else 0.0)
        demand_mean = self.base_demand * promo_multiplier * demand_multiplier
        demand = max(0, int(round(random.gauss(demand_mean, demand_mean * 0.15))))
        sales = min(inventory, demand)
        stockouts = max(0, demand - sales)

        # Reward components
        contribution = sales * margin
        holding_cost = (inventory - sales) * self.holding_cost_per_unit
        penalty = stockouts * self.stockout_penalty * stockout_penalty_multiplier
        reward = contribution - holding_cost - penalty - action_cost

        # Advance time
        next_week = self.state.week + 1
        done = next_week >= self.horizon_weeks
        next_inventory = inventory - sales
        risk_level = self._compute_risk(next_inventory, supplier_status)

        self.state = InventoryState(
            week=next_week,
            inventory_on_hand=next_inventory,
            pipeline=new_pipeline,
            risk_level=risk_level,
            supplier_status=supplier_status,
            promo_active=next_week in self.promo_weeks,
        )

        info = StepInfo(
            demand=float(demand),
            sales=float(sales),
            stockouts=float(stockouts),
            contribution=contribution,
            holding_cost=holding_cost,
            penalty=penalty,
            action_cost=action_cost,
        )
        return self.state, reward, done, info

    def _compute_risk(self, inventory: int, supplier_status: SupplierStatus) -> str:
        if inventory < 30 or supplier_status == SupplierStatus.DELAYED:
            return "high"
        if inventory < 60:
            return "medium"
        return "low"

    # endregion book:capstone-mdp-step-reward


# region book:capstone-mdp-baseline
def baseline_action(env: InventoryRiskMDP, state: InventoryState) -> Action:
    """Reorder for a projected shortage over the supply horizon, including dated receipts."""
    horizon = env.reorder_lead_time + (state.supplier_status == SupplierStatus.DELAYED)
    if state.week + horizon >= env.horizon_weeks:
        return Action.WAIT  # No saleable receipt before the episode ends.
    available = state.inventory_on_hand
    for week in range(horizon + 1):
        available += sum(qty for eta, qty in state.pipeline if eta == week + 1)
        demand = env.base_demand * (1 + (env.promo_lift if state.week + week in env.promo_weeks else 0))
        available -= demand
        if available < 0:
            return Action.REORDER
    return Action.WAIT


# endregion book:capstone-mdp-baseline


# region book:capstone-mdp-run
if __name__ == "__main__":
    random.seed(7)
    env = InventoryRiskMDP()
    state = env.reset()

    for _ in range(env.horizon_weeks):
        action = baseline_action(env, state)

        state, reward, done, info = env.step(action)
        print(
            f"week={state.week} action={action.value} inventory={state.inventory_on_hand} "
            f"risk={state.risk_level} reward={reward:.1f} stockouts={int(info.stockouts)}"
        )

        if done:
            break
# endregion book:capstone-mdp-run
