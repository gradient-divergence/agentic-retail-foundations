"""
Demonstration: run the BDI inventory agent on a small synthetic scenario.

This file is used by the book via region-tagged snippet extraction to keep
the manuscript and runnable code aligned.
"""

# region book:bdi-inventory-demo
import logging
from datetime import datetime, timedelta

from agents.bdi import InventoryBDIAgent
from models.inventory import InventoryItem, ProductInfo, SalesData


def demonstrate_bdi_agent() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    agent = InventoryBDIAgent()

    products = {
        "P001": ProductInfo(
            product_id="P001",
            name="Organic Apples",
            category="Produce",
            price=2.99,
            cost=1.50,
            lead_time_days=2,
            shelf_life_days=14,
            supplier_id="S1",
            min_order_quantity=10,
        ),
        "P002": ProductInfo(
            product_id="P002",
            name="Whole Grain Bread",
            category="Bakery",
            price=3.49,
            cost=1.25,
            lead_time_days=1,
            shelf_life_days=5,
            supplier_id="S2",
            min_order_quantity=20,
        ),
        "P003": ProductInfo(
            product_id="P003",
            name="Premium Coffee",
            category="Beverages",
            price=12.99,
            cost=6.50,
            lead_time_days=5,
            supplier_id="S3",
            min_order_quantity=5,
        ),
    }

    inventory = {
        "P001": InventoryItem(
            product_id="P001",
            current_stock=25,
            reorder_point=20,
            optimal_stock=50,
        ),
        "P002": InventoryItem(
            product_id="P002",
            current_stock=5,
            reorder_point=10,
            optimal_stock=30,
        ),
        "P003": InventoryItem(
            product_id="P003",
            current_stock=60,
            reorder_point=15,
            optimal_stock=40,
        ),
    }

    sales = {
        "P001": SalesData(
            product_id="P001",
            daily_sales=[
                8,
                7,
                9,
                8,
                10,
                12,
                9,
                8,
                7,
                6,
                8,
                9,
                10,
                11,
                9,
                8,
                9,
                10,
                11,
                12,
                13,
                11,
                10,
                12,
                13,
                14,
                15,
                13,
                12,
                11,
            ],
        ),
        "P002": SalesData(
            product_id="P002",
            daily_sales=[
                6,
                5,
                7,
                8,
                6,
                5,
                4,
                6,
                7,
                8,
                6,
                5,
                4,
                5,
                6,
                7,
                8,
                9,
                7,
                6,
                5,
                6,
                7,
                8,
                9,
                10,
                8,
                7,
                6,
                7,
            ],
        ),
        "P003": SalesData(
            product_id="P003",
            daily_sales=[
                2,
                1,
                3,
                2,
                1,
                2,
                3,
                2,
                1,
                0,
                2,
                3,
                2,
                1,
                3,
                2,
                1,
                2,
                3,
                4,
                2,
                1,
                2,
                3,
                2,
                1,
                2,
                1,
                2,
                3,
            ],
        ),
    }

    agent.update_beliefs(
        new_products=products,
        new_inventory=inventory,
        new_sales=sales,
        new_date=datetime(2025, 1, 15),
    )

    print("\n[Day 1] BEFORE Cycle:")
    for pid, item in sorted(agent.inventory.items()):
        print(
            f"  {pid} stock={item.current_stock:<3} pending={item.pending_order_quantity:<3}"
            f" (optimal={item.optimal_stock}, reorder={item.reorder_point})"
        )

    goals = ["minimize_stockouts", "maximize_profit_margin"]
    print("\n[Day 1] Running BDI Cycle...")
    actions_day_1 = agent.run_cycle(prioritized_goals=goals)

    print("\n[Day 1] Actions:")
    for i, action in enumerate(actions_day_1, 1):
        print(f"  {i}. {action}")

    print("\nSimulating one day passing...")
    new_inventory = {
        "P001": InventoryItem(
            product_id="P001",
            current_stock=max(0, agent.inventory["P001"].current_stock - 13),
            reorder_point=agent.inventory["P001"].reorder_point,
            optimal_stock=agent.inventory["P001"].optimal_stock,
            pending_order_quantity=agent.inventory["P001"].pending_order_quantity,
            expected_delivery_date=agent.inventory["P001"].expected_delivery_date,
            last_reorder_date=agent.inventory["P001"].last_reorder_date,
        ),
        "P002": InventoryItem(
            product_id="P002",
            current_stock=max(0, agent.inventory["P002"].current_stock - 7),
            reorder_point=agent.inventory["P002"].reorder_point,
            optimal_stock=agent.inventory["P002"].optimal_stock,
            pending_order_quantity=agent.inventory["P002"].pending_order_quantity,
            expected_delivery_date=agent.inventory["P002"].expected_delivery_date,
            last_reorder_date=agent.inventory["P002"].last_reorder_date,
        ),
        "P003": InventoryItem(
            product_id="P003",
            current_stock=max(0, agent.inventory["P003"].current_stock - 2),
            reorder_point=agent.inventory["P003"].reorder_point,
            optimal_stock=agent.inventory["P003"].optimal_stock,
            pending_order_quantity=agent.inventory["P003"].pending_order_quantity,
            expected_delivery_date=agent.inventory["P003"].expected_delivery_date,
            last_reorder_date=agent.inventory["P003"].last_reorder_date,
        ),
    }

    agent.update_beliefs(
        new_inventory=new_inventory,
        new_date=agent.current_date + timedelta(days=1),
    )

    print("\n[Day 2] Running BDI Cycle...")
    actions_day_2 = agent.run_cycle(prioritized_goals=goals)

    print("\n[Day 2] Updated state:")
    print(f"  Bread (P002): {agent.inventory['P002'].current_stock} units")
    print(f"  Coffee (P003): {agent.inventory['P003'].current_stock} units")
    print(f"  Apples (P001): {agent.inventory['P001'].current_stock} units")

    print("\n[Day 2] Actions:")
    for i, action in enumerate(actions_day_2, 1):
        print(f"  {i}. {action}")


if __name__ == "__main__":
    demonstrate_bdi_agent()
# endregion book:bdi-inventory-demo
