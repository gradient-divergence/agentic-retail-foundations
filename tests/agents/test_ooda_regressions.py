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


@pytest.mark.parametrize("old_price,new_price", [(29.28, 29.29), (29.29, 29.28)])
def test_one_cent_change_executes(old_price, new_price):
    agent = OODAPricingAgent()
    product = PricingProduct("A", "Tee", "apparel", 5, old_price, 10, 40)
    agent.update_products({"A": product})
    assert agent.act("A", {"old_price": old_price, "new_price": new_price})
    assert product.current_price == new_price
