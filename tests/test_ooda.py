from agents.ooda import OODAPricingAgent
from models.pricing import PricingProduct


def test_ooda_agent_initialization():
    """Test that OODAPricingAgent initializes with default values."""
    agent = OODAPricingAgent()
    assert isinstance(agent.products, dict)
    assert isinstance(agent.action_history, list)


def test_ooda_agent_update_products():
    """Test updating products with minimal valid data."""
    agent = OODAPricingAgent()
    products = {
        "P001": PricingProduct(
            product_id="P001",
            name="Test Product",
            category="TestCat",
            cost=5.0,
            current_price=10.0,
            min_price=8.0,
            max_price=15.0,
            inventory=10,
            target_profit_margin=0.3,
            sales_last_7_days=[1, 2, 3, 4, 5, 6, 7],
        )
    }
    agent.update_products(products)
    assert agent.products["P001"].name == "Test Product"
    assert agent.products["P001"].current_price == 10.0


def test_ooda_agent_run_cycle_for_product():
    """Test running a full OODA cycle for a product with minimal data."""
    agent = OODAPricingAgent()
    products = {
        "P001": PricingProduct(
            product_id="P001",
            name="Test Product",
            category="TestCat",
            cost=5.0,
            current_price=10.0,
            min_price=8.0,
            max_price=15.0,
            inventory=10,
            target_profit_margin=0.3,
            sales_last_7_days=[1, 2, 3, 4, 5, 6, 7],
        )
    }
    agent.update_products(products)
    result = agent.run_cycle_for_product("P001")
    assert isinstance(result, bool)
