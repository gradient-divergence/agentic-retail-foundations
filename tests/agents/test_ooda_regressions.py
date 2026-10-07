import pytest

from agents.ooda import OODAPricingAgent
from models.pricing import PricingProduct


@pytest.mark.parametrize(
    "price,min_price,max_price,cap",
    [(10, 8, 15, 5), (10, 10, 10, 5), (10, 8, 15, 0), (8.5, 8.9, 12, 5), (19.99, 10, 30, 0)],
)
def test_psychological_rounding_respects_price_and_change_limits(price, min_price, max_price, cap):
    agent = OODAPricingAgent(max_price_change_pct=cap)
    product = PricingProduct("A", "Tee", "apparel", 5, price, min_price, max_price)
    agent.update_products({"A": product})
    decision = agent.decide("A", {"inventory_status": "optimal", "sales_assessment": "normal"})
    assert min_price <= decision["new_price"] <= max_price
    assert abs(decision["new_price"] / price - 1) <= cap / 100 + 1e-12
    assert decision["new_price"] * 100 == pytest.approx(round(decision["new_price"] * 100))
