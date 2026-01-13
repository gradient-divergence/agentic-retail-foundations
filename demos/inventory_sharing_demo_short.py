"""Short, print-friendly cooperative inventory sharing demo."""

# region book:inventory-sharing-demo-short
import asyncio

from agents.protocols.inventory_sharing import InventoryCollaborationNetwork
from models.store import Store


async def demo_collaborative_inventory_sharing() -> None:
    network = InventoryCollaborationNetwork(max_transfer_distance=2.0)

    stores = [
        Store(
            store_id="store-a",
            name="Downtown Store",
            location="City Center",
            initial_cooperation_score=1.2,
        ),
        Store(
            store_id="store-b",
            name="Suburban Store",
            location="Westfield",
            initial_cooperation_score=1.0,
        ),
        Store(
            store_id="store-c",
            name="Mall Store",
            location="Eastland Mall",
            initial_cooperation_score=0.9,
        ),
    ]
    for store in stores:
        network.register_store(store)

    product_id = "P1002"
    stores[0].add_product(product_id, current_stock=150, target_stock=80, sales_rate_per_day=8)
    stores[1].add_product(product_id, current_stock=40, target_stock=70, sales_rate_per_day=12)
    stores[2].add_product(product_id, current_stock=25, target_stock=60, sales_rate_per_day=10)

    opportunities = await network.identify_transfer_opportunities()
    approved = [o for o in opportunities if o["net_value"] > 0]
    results = await network.execute_transfers(approved)

    print(f"Transfers executed: {len(results)}")
    for transfer in results:
        print(
            f"{transfer['sender_id']} -> {transfer['receiver_id']}: "
            f"{transfer['quantity']} of {transfer['product_id']}"
        )


if __name__ == "__main__":
    asyncio.run(demo_collaborative_inventory_sharing())

# endregion book:inventory-sharing-demo-short
