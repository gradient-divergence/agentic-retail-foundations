from demos.inventory_agent_unit_test_demo import DemandForecast, InventoryAgent, InventorySnapshot


def test_forecast_for_new_sku_generates_order():
    plan = InventoryAgent(safety_stock=10).evaluate_restock(
        InventorySnapshot(stock={"existing": 40}), DemandForecast(demand={"new": 15})
    )
    assert plan.orders == {"existing": 0, "new": 25}
