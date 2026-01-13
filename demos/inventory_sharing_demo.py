"""
Demonstrates cooperative inventory sharing between stores.

This demo runs two collaboration cycles:
1) Identify transfer opportunities and execute beneficial transfers.
2) Simulate a demand shift and repeat.
"""

# region book:inventory-sharing-demo

import asyncio
import logging
import random

from pydantic import BaseModel

from agents.protocols.inventory_sharing import InventoryCollaborationNetwork
from models.store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TransferOpportunity(BaseModel):
    sender_id: str
    receiver_id: str
    product_id: str
    quantity: int
    net_value: float


def _print_opportunities(
    network: InventoryCollaborationNetwork, opportunities: list[TransferOpportunity]
) -> None:
    if not opportunities:
        print("No transfer opportunities found.")
        return

    print(f"Identified {len(opportunities)} potential transfers:")
    for i, opp in enumerate(opportunities, start=1):
        sender = network.stores[opp.sender_id].name
        receiver = network.stores[opp.receiver_id].name
        product_id = opp.product_id
        quantity = opp.quantity
        net_value = opp.net_value
        print(f"{i}. {sender} -> {receiver}: {quantity} of {product_id} (Value: {net_value:.2f})")


def _print_final_inventory(stores: list[Store]) -> None:
    print("\n=== Final Inventory Status ===")
    for store in stores:
        print(f"\n{store.name}:")
        for product_id, pos in store.inventory.items():
            status = pos.get_status()
            print(
                f"  {product_id}: {pos.current_stock} units, "
                f"{pos.days_of_supply():.1f} days supply, Status: {status.value}"
            )

        out_transfers = len([t for t in store.transfer_history if t["direction"] == "out"])
        in_transfers = len([t for t in store.transfer_history if t["direction"] == "in"])
        print(f"  Transfers out: {out_transfers}, in: {in_transfers}")
        print(f"  Cooperation Score: {store.cooperation_score:.2f}")


async def demo_collaborative_inventory_sharing() -> None:
    logger.info("Initializing Collaborative Inventory Sharing Demo...")

    random.seed(7)
    network = InventoryCollaborationNetwork(max_transfer_distance=2.0)

    stores = [
        Store(
            store_id="store1",
            name="Downtown Store",
            location="City Center",
            initial_cooperation_score=1.2,
        ),
        Store(
            store_id="store2",
            name="Suburban Store",
            location="Westfield",
            initial_cooperation_score=1.0,
        ),
        Store(
            store_id="store3",
            name="Mall Store",
            location="Eastland Mall",
            initial_cooperation_score=0.8,
        ),
        Store(
            store_id="store4",
            name="Express Store",
            location="North Station",
            initial_cooperation_score=1.5,
        ),
        Store(
            store_id="store5",
            name="Flagship Store",
            location="Main Street",
            initial_cooperation_score=0.9,
        ),
    ]
    for store in stores:
        network.register_store(store)

    # Add sample products
    for store in stores:
        store.add_product(
            "P1001",
            current_stock=100,
            target_stock=80,
            sales_rate_per_day=10,
        )

    stores[0].add_product("P1002", current_stock=150, target_stock=80, sales_rate_per_day=8)
    stores[1].add_product("P1002", current_stock=120, target_stock=80, sales_rate_per_day=7)
    stores[2].add_product("P1002", current_stock=30, target_stock=60, sales_rate_per_day=12)
    stores[3].add_product("P1002", current_stock=40, target_stock=70, sales_rate_per_day=10)
    stores[4].add_product("P1002", current_stock=90, target_stock=80, sales_rate_per_day=9)

    stores[0].add_product("P1003", current_stock=20, target_stock=40, sales_rate_per_day=15)
    stores[1].add_product("P1003", current_stock=30, target_stock=40, sales_rate_per_day=5)
    stores[2].add_product("P1003", current_stock=10, target_stock=30, sales_rate_per_day=8)
    stores[3].add_product("P1003", current_stock=80, target_stock=40, sales_rate_per_day=3)
    stores[4].add_product("P1003", current_stock=25, target_stock=40, sales_rate_per_day=12)

    print("\n=== First Collaboration Cycle ===")
    opportunities_raw = await network.identify_transfer_opportunities()
    opportunities = [TransferOpportunity(**opp) for opp in opportunities_raw]
    _print_opportunities(network, opportunities)

    approved = [o for o in opportunities if o.net_value > 0]
    results = await network.execute_transfers([op.model_dump() for op in approved])
    print(f"\nExecuted {len(results)} transfers")

    print("\n=== Simulating changed conditions ===")
    stores[3].update_sales_rate("P1002", new_rate=18)
    print("Express Store had a sales spike for P1002")
    stores[0].update_sales_rate("P1003", new_rate=8)
    print("Downtown Store had a sales slowdown for P1003")

    print("\n=== Second Collaboration Cycle ===")
    opportunities2_raw = await network.identify_transfer_opportunities()
    opportunities2 = [TransferOpportunity(**opp) for opp in opportunities2_raw]
    _print_opportunities(network, opportunities2)

    approved2 = [o for o in opportunities2 if o.net_value > 0]
    results2 = await network.execute_transfers([op.model_dump() for op in approved2])
    print(f"\nExecuted {len(results2)} transfers")

    _print_final_inventory(stores)


if __name__ == "__main__":
    asyncio.run(demo_collaborative_inventory_sharing())

# endregion book:inventory-sharing-demo
