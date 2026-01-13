"""
Short demo for the BDI inventory agent (print-friendly).
"""

# region book:bdi-inventory-demo-short
from datetime import datetime

from agents.bdi import InventoryBDIAgent
from models.inventory import InventoryItem, ProductInfo, SalesData


def run_short_demo() -> None:
    agent = InventoryBDIAgent()

    products = {
        "SKU-001": ProductInfo(
            product_id="SKU-001",
            name="Everyday Coffee",
            category="Beverage",
            price=12.0,
            cost=6.0,
            lead_time_days=5,
            shelf_life_days=180,
            supplier_id="SUP-COFFEE",
        )
    }
    inventory = {
        "SKU-001": InventoryItem(
            product_id="SKU-001",
            current_stock=12,
            reorder_point=15,
            optimal_stock=40,
        )
    }
    sales = {"SKU-001": SalesData(product_id="SKU-001", daily_sales=[5, 6, 7, 8, 6])}

    agent.update_beliefs(
        new_products=products,
        new_inventory=inventory,
        new_sales=sales,
        new_date=datetime(2025, 1, 15),
    )

    actions = agent.run_cycle(prioritized_goals=["minimize_stockouts"])
    print(f"Actions executed: {actions}")


if __name__ == "__main__":
    run_short_demo()

# endregion book:bdi-inventory-demo-short
