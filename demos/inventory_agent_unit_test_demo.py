from __future__ import annotations

from pydantic import BaseModel, Field


class InventorySnapshot(BaseModel):
    stock: dict[str, int]
    on_order: dict[str, int] = Field(default_factory=dict)
    season_open: bool = True


class DemandForecast(BaseModel):
    demand: dict[str, int]


class RestockPlan(BaseModel):
    orders: dict[str, int]


# region book:inventory-agent-unit-test-demo
class InventoryAgent:
    def __init__(self, safety_stock: int):
        self.safety_stock = safety_stock

    def evaluate_restock(self, snapshot: InventorySnapshot, forecast: DemandForecast) -> RestockPlan:
        orders: dict[str, int] = {}
        for item in snapshot.stock.keys() | forecast.demand.keys():
            stock = snapshot.stock.get(item, 0) + snapshot.on_order.get(item, 0)
            demand = forecast.demand.get(item, 0)
            if snapshot.season_open and stock < demand + self.safety_stock:
                order_qty = (demand + self.safety_stock) - stock
                orders[item] = max(order_qty, 0)
            else:
                orders[item] = 0
        return RestockPlan(orders=orders)


def test_inventory_restock_logic():
    agent = InventoryAgent(safety_stock=10)
    snapshot = InventorySnapshot(stock={"SKU-123": 5, "SKU-456": 20})
    forecast = DemandForecast(demand={"SKU-123": 15, "SKU-456": 5})
    orders = agent.evaluate_restock(snapshot, forecast)
    assert orders.orders["SKU-123"] == 20
    assert orders.orders["SKU-456"] == 0

    snapshot = InventorySnapshot(stock={"SKU-789": 50})
    forecast = DemandForecast(demand={"SKU-789": 30})
    orders = agent.evaluate_restock(snapshot, forecast)
    assert orders.orders["SKU-789"] == 0
    assert agent.evaluate_restock(InventorySnapshot(stock={"SKU-789": 40}), forecast).orders["SKU-789"] == 0


# endregion book:inventory-agent-unit-test-demo
