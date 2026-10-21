"""Compact decide-phase excerpt for OODA pricing."""

from datetime import datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

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


def _bounded_price(current_price, proposed_price, min_price, max_price, max_change_pct):
    price = Decimal(str(current_price))
    cap = Decimal(str(max_change_pct)) / 100
    cent = Decimal("0.01")
    lower = max(Decimal(str(min_price)), price * (1 - cap)).quantize(cent, ROUND_CEILING)
    upper = min(Decimal(str(max_price)), price * (1 + cap)).quantize(cent, ROUND_FLOOR)
    if price <= 0 or cap < 0 or lower > upper:
        raise ValueError("No feasible price within bounds and change cap")
    final = max(lower, min(upper, Decimal(str(proposed_price)).quantize(cent)))
    return float(final), float((final - price) / price * 100)


# region book:ooda-pricing-decide-excerpt
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
        2.5
        if sales_assess == "risk_of_stockout"
        else -2.5
        if sales_assess == "slow_moving"
        else -4.0
        if sales_assess == "stagnant"
        else 0.0
    )

    total_change = (
        inv_component * weights.inventory
        + comp_component * weights.competitor
        + sales_component * weights.sales
    )
    capped_change = max(-max_change_pct, min(max_change_pct, total_change))
    new_price = current_price * (1 + capped_change / 100)
    new_price, capped_change = _bounded_price(
        current_price, new_price, min_price, max_price, max_change_pct
    )

    return PriceDecision(
        timestamp=datetime.now(),
        old_price=current_price,
        new_price=round(new_price, 2),
        capped_change_pct=capped_change,
    )


# endregion book:ooda-pricing-decide-excerpt
