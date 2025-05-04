"""
Pricing-related data models for agentic-retail-foundations.
Includes the PricingProduct dataclass for OODA pricing agent demonstration.
"""

from dataclasses import dataclass, field


@dataclass
class PricingProduct:
    """
    Data model for a product used in dynamic pricing (OODA agent).
    """

    product_id: str
    name: str
    category: str
    cost: float
    current_price: float
    min_price: float
    max_price: float
    inventory: int = 0
    target_profit_margin: float = 0.3
    competitor_prices: dict[str, float] = field(default_factory=dict)
    sales_last_7_days: list[int] = field(default_factory=list)
