from environments.inventory_risk_sim import InventoryRiskSimConfig, InventoryRiskSimulator, evaluate_policies


def test_zero_lead_time_orders_arrive_by_next_demand_period(monkeypatch):
    config = InventoryRiskSimConfig(
        horizon_weeks=3,
        initial_inventory=0,
        base_demand=5,
        reorder_qty=10,
        lead_time_weeks=0,
        disruption_probability=0,
    )
    simulator = InventoryRiskSimulator(config)
    monkeypatch.setattr(simulator, "_demand_for_week", lambda _: 5)
    metrics = simulator.run()
    assert metrics.lost_sales == 5
    assert metrics.total_margin == 10


def test_policies_are_compared_on_the_same_seeded_scenario():
    config = InventoryRiskSimConfig()
    results = evaluate_policies(config, [lambda _: 0, lambda _: 0])
    assert results[0] == results[1]
