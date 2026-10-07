from agents.bdi import InventoryBDIAgent
from models.inventory import InventoryItem, ProductInfo, SalesData


def make_agent(stock=0, optimal=40, shelf_life=180):
    agent = InventoryBDIAgent()
    agent.update_beliefs(
        new_products={"A": ProductInfo("A", "Coffee", "beverage", 12, 6, 5, shelf_life)},
        new_inventory={"A": InventoryItem("A", stock, 15, optimal)},
        new_sales={"A": SalesData("A", [10] * 14)},
    )
    return agent


def test_exhausted_inventory_triggers_reordering_in_default_cycle():
    agent = make_agent()
    actions = agent.run_cycle()
    assert [action["action"] for action in actions] == ["reorder"]
    assert agent.inventory["A"].pending_order_quantity == 40


def test_reorder_does_not_exceed_target_when_supply_is_already_sufficient():
    agent = make_agent(stock=40)
    assert agent.run_cycle(["minimize_stockouts"]) == []
    assert agent.inventory["A"].pending_order_quantity == 0


def test_freshness_planning_does_not_discount_products_that_will_sell_in_time():
    agent = make_agent(stock=40)
    assert agent.run_cycle(["ensure_fresh_products"]) == []
    assert agent.products["A"].current_price == 12
