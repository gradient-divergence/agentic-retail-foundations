from agents.bdi import InventoryBDIAgent
from demos.bdi_planning_excerpt import plan_for_goals
from models.inventory import InventoryItem, ProductInfo, SalesData


def test_excerpt_preserves_canonical_goal_priority_and_clears_old_intentions():
    agent = InventoryBDIAgent()
    agent.update_beliefs(
        new_products={"A": ProductInfo("A", "Bread", "bakery", 10, 2, 1, 2)},
        new_inventory={"A": InventoryItem("A", 100, 10, 20)},
        new_sales={"A": SalesData("A", [1] * 14)},
    )
    goals = ["ensure_fresh_products", "minimize_excess_inventory"]
    agent.generate_intentions(goals)
    expected = agent.active_intentions.copy()
    agent.active_intentions = [{"action": "stale"}]
    result = plan_for_goals(agent, goals)
    assert len(result) == 1
    assert result[0].action == "discount_perishable"
    assert agent.active_intentions == expected
