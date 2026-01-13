from __future__ import annotations

from pydantic import BaseModel


class InventorySnapshot(BaseModel):
    stock: dict[str, int]


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
        for item, stock in snapshot.stock.items():
            demand = forecast.demand.get(item, 0)
            if stock < demand + self.safety_stock:
                order_qty = (demand + self.safety_stock) - stock
                orders[item] = max(order_qty, 0)
            else:
                orders[item] = 0
        return RestockPlan(orders=orders)


def test_inventory_restock_logic():
    agent = InventoryAgent(safety_stock=10)
    snapshot = InventorySnapshot(stock={"Jeans": 5, "T-Shirt": 20})
    forecast = DemandForecast(demand={"Jeans": 15, "T-Shirt": 5})
    orders = agent.evaluate_restock(snapshot, forecast)
    assert orders.orders["Jeans"] >= 20
    assert orders.orders["T-Shirt"] == 0

    snapshot = InventorySnapshot(stock={"Dress": 50})
    forecast = DemandForecast(demand={"Dress": 30})
    orders = agent.evaluate_restock(snapshot, forecast)
    assert orders.orders["Dress"] == 0


# endregion book:inventory-agent-unit-test-demo
