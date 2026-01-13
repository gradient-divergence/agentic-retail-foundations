"""Compact decide-phase excerpt for OODA pricing."""

# region book:ooda-pricing-decide-excerpt
from datetime import datetime

from pydantic import BaseModel


class OrientationSignal(BaseModel):
    inventory_status: str | None = None
    sales_assessment: str | None = None
    price_diff_pct: float = 0.0


class DecisionWeights(BaseModel):
    inventory: float = 0.4
    competitor: float = 0.3
    sales: float = 0.3


class PriceDecision(BaseModel):
    timestamp: datetime
    old_price: float
    new_price: float
    capped_change_pct: float


def decide_price(
    current_price: float,
    orientation: OrientationSignal,
    weights: DecisionWeights,
    min_price: float,
    max_price: float,
    max_change_pct: float,
) -> PriceDecision:
    inv_status = orientation.inventory_status
    sales_assess = orientation.sales_assessment
    price_diff_pct = orientation.price_diff_pct

    inv_component = 2.0 if inv_status == "low" else -3.0 if inv_status == "high" else 0.0
    comp_component = -(price_diff_pct / 3.0) if abs(price_diff_pct) > 5 else 0.0
    sales_component = (
        2.5 if sales_assess == "risk_of_stockout" else -2.5 if sales_assess == "slow_moving" else 0.0
    )

    total_change = (
        inv_component * weights.inventory
        + comp_component * weights.competitor
        + sales_component * weights.sales
    )
    capped_change = max(-max_change_pct, min(max_change_pct, total_change))
    new_price = current_price * (1 + capped_change / 100)
    new_price = max(min_price, min(max_price, new_price))

    return PriceDecision(
        timestamp=datetime.now(),
        old_price=current_price,
        new_price=round(new_price, 2),
        capped_change_pct=capped_change,
    )


# endregion book:ooda-pricing-decide-excerpt
