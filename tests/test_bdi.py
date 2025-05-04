from agents.bdi import InventoryBDIAgent
from models.inventory import ProductInfo, InventoryItem, SalesData
from datetime import datetime


def test_bdi_agent_initialization():
    """Test that InventoryBDIAgent initializes with default values."""
    agent = InventoryBDIAgent()
    assert isinstance(agent.products, dict)
    assert isinstance(agent.inventory, dict)
    assert isinstance(agent.sales_data, dict)
    assert isinstance(agent.goals, dict)
    assert isinstance(agent.active_intentions, list)


def test_bdi_agent_update_beliefs():
    """Test updating beliefs with minimal valid data."""
    agent = InventoryBDIAgent()
    products = {
        "P001": ProductInfo(
            product_id="P001",
            name="Test Product",
            category="TestCat",
            price=10.0,
            cost=5.0,
            lead_time_days=2,
            shelf_life_days=10,
            supplier_id="S1",
            min_order_quantity=1,
        )
    }
    inventory = {
        "P001": InventoryItem(
            product_id="P001",
            current_stock=10,
            reorder_point=5,
            optimal_stock=20,
        )
    }
    sales = {
        "P001": SalesData(
            product_id="P001",
            daily_sales=[1, 2, 3, 4, 5, 6, 7],
        )
    }
    now = datetime.now()
    agent.update_beliefs(
        new_products=products, new_inventory=inventory, new_sales=sales, new_date=now
    )
    assert agent.products["P001"].name == "Test Product"
    assert agent.inventory["P001"].current_stock == 10
    assert agent.sales_data["P001"].daily_sales[-1] == 7
    assert agent.current_date == now


def test_bdi_agent_run_cycle():
    """Test running a full BDI cycle with minimal data."""
    agent = InventoryBDIAgent()
    products = {
        "P001": ProductInfo(
            product_id="P001",
            name="Test Product",
            category="TestCat",
            price=10.0,
            cost=5.0,
            lead_time_days=2,
            shelf_life_days=10,
            supplier_id="S1",
            min_order_quantity=1,
        )
    }
    inventory = {
        "P001": InventoryItem(
            product_id="P001",
            current_stock=10,
            reorder_point=5,
            optimal_stock=20,
        )
    }
    sales = {
        "P001": SalesData(
            product_id="P001",
            daily_sales=[1, 2, 3, 4, 5, 6, 7],
        )
    }
    now = datetime.now()
    agent.update_beliefs(
        new_products=products, new_inventory=inventory, new_sales=sales, new_date=now
    )
    actions = agent.run_cycle()
    assert isinstance(actions, list)
