import pytest

from models.fulfillment import Associate, Order, OrderLineItem
from utils import planning


def test_quantities_and_shared_locations_are_counted_once():
    planner = planning.FulfillmentPlanner(planning.StoreLayout(2, 1))
    order = Order("order", [OrderLineItem("A", 2, 1.0), OrderLineItem("B", 3, 1.0)], "customer")
    details = [
        {"location": (1, 0), "handling_time": 2.0},
        {"location": (1, 0), "handling_time": 4.0},
    ]
    path, minutes = planner._estimate_order_time(order, details, (0, 0), efficiency=2.0)
    assert path == [(0, 0), (1, 0), (0, 0)]
    assert minutes == pytest.approx((2 * 2 + 3 * 4 + 0.2) / 2)


def test_shift_capacity_counts_every_unit_across_orders(monkeypatch):
    monkeypatch.setattr(
        planning,
        "get_mock_item_details",
        lambda _: {"location": (0, 0), "handling_time": 1.0, "temperature_zone": "ambient"},
    )
    planner = planning.FulfillmentPlanner(planning.StoreLayout(1, 1))
    associate = Associate("picker", "Picker", shift_end_time=15)
    planner.add_associate(associate)
    for index in range(2):
        planner.add_order(Order(str(index), [OrderLineItem("A", 12, 1.0)], "customer"))
    planner.plan()
    assigned = planner.assignments[associate.associate_id]
    assert len(assigned) == 1
    assert sum(item.quantity for order in assigned for item in order.items) == 12
    assert planner.estimated_times[associate.associate_id] == 12


def test_current_task_reserves_shift_minutes(monkeypatch):
    monkeypatch.setattr(
        planning,
        "get_mock_item_details",
        lambda _: {"location": (0, 0), "handling_time": 1.0, "temperature_zone": "ambient"},
    )
    planner = planning.FulfillmentPlanner(planning.StoreLayout(1, 1))
    planner.add_associate(Associate("picker", "Picker", shift_end_time=15, current_task_completion_time=5))
    planner.add_order(Order("new", [OrderLineItem("A", 12, 1.0)], "customer"))
    planner.plan()
    assert planner.assignments["picker"] == []


@pytest.mark.parametrize("start,end", [((0, 0), (0, 0)), ((-1, 0), (1, 0)), ((1, 0), (0, 0))])
def test_path_rejects_blocked_or_out_of_bounds_endpoints(start, end):
    layout = planning.StoreLayout(2, 1)
    layout.add_obstacle(0, 0)
    assert layout.shortest_path(start, end) is None
