import subprocess
import sys

from demos.store_fulfillment_book_demo import Associate, FulfillmentPlanner, Item, Order, StoreLayout


def test_cumulative_batch_capacity_and_replanning_keep_assignments_consistent():
    planner = FulfillmentPlanner(StoreLayout(1, 1))
    orders = [Order(str(i), [Item(str(i), "Item", "grocery", (0, 0), handling_time=2)] * 6) for i in range(2)]
    associate = Associate("A", "Alex", shift_end_time=15)
    planner.add_associate(associate)
    for order in orders:
        planner.add_order(order)
    assert planner.plan()["unassigned"] == [orders[1]]
    assert associate.estimate_time_to_complete(planner.assignments["A"]) == 12
    associate.shift_end_time = 30
    assert planner.plan()["unassigned"] == []
    assert [order.assigned_to for order in orders] == ["A", "A"]
    associate.shift_end_time = 0
    assert planner.plan()["unassigned"] == orders
    assert all(order.assigned_to is None for order in orders)


def test_no_zone_authorization_means_no_assignment():
    associate = Associate("A", "Alex", authorized_zones=[])
    assert not associate.can_handle_order(Order("O", [Item("I", "Item", "grocery", (0, 0))]))


def test_picking_path_walks_around_obstacles_and_rejects_unreachable_orders():
    layout = StoreLayout(3, 2)
    layout.add_obstacle(1, 0)
    path = layout.optimize_path([(2, 0)], (0, 0))
    assert path[0] == (0, 0) and path[-1] == (2, 0)
    assert all(point not in layout.obstacles for point in path)
    assert all(layout.distance(a, b) == 1 for a, b in zip(path, path[1:], strict=False))
    planner = FulfillmentPlanner(layout)
    order = Order("O", [Item("I", "Blocked", "grocery", (1, 0))])
    planner.add_order(order)
    planner.add_associate(Associate("A", "Alex"))
    assert planner.plan()["unassigned"] == [order]


def test_module_runs_the_fulfillment_demo():
    result = subprocess.run(
        [sys.executable, "-m", "demos.store_fulfillment_book_demo"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Fulfillment Plan Summary:" in result.stdout
