"""
Demonstration of the event-driven order orchestration framework.
"""

import asyncio

from agents.fulfillment import FulfillmentAgent
from agents.inventory_orchestration import InventoryAgent
from agents.orchestrator import MasterOrchestrator
from models.enums import AgentType
from models.events import RetailEvent
from utils.event_bus import EventBus


async def run_orchestration_simulation() -> list[str]:
    """Synthetic adapters complete an order trace, with one allocation failure and replay."""
    from models.fulfillment import Order, OrderLineItem

    bus = EventBus()
    trace = []
    events = [
        "order.created",
        "order.validated",
        "order.exception",
        "inventory.reallocation_requested",
        "order.allocated",
        "payment.request_processing",
        "order.payment_processed",
        "fulfillment.requested",
        "order.status_update_request",
        "fulfillment.picked",
        "fulfillment.packed",
        "fulfillment.shipped",
        "order.delivered",
        "order.completed",
    ]

    async def record(event):
        trace.append(event.event_type)

    for event_type in events:
        bus.subscribe(event_type, record)
    inventory = InventoryAgent("inventory", bus)
    fulfillment = FulfillmentAgent("fulfillment", bus)
    master = MasterOrchestrator("master", bus)
    order = Order("ORD-SIM-001", [OrderLineItem("PROD-001", 2, 50)], "CUST-SIM-101")

    async def lookup(order_id):
        return order if order_id == order.order_id else None

    inventory._get_order = fulfillment._get_order = lookup
    allocate = inventory.allocate_inventory
    attempts = 0

    async def allocate_with_failure(value):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("synthetic unavailable allocation")
        await allocate(value)

    inventory.allocate_inventory = allocate_with_failure
    await bus.publish(
        RetailEvent(
            event_type="order.created",
            payload={"order_id": order.order_id, "policy": {"allow_substitutions": False}},
            source=AgentType.CUSTOMER,
        )
    )
    await bus.publish(
        RetailEvent(
            event_type="order.validated", payload={"order_id": order.order_id}, source=AgentType.FINANCIAL
        )
    )
    corrected = RetailEvent(
        event_type="order.validated", payload={"order_id": order.order_id}, source=AgentType.FINANCIAL
    )
    await bus.publish(corrected)
    await bus.publish(corrected)  # Replay must not allocate or charge twice.
    await bus.publish(
        RetailEvent(
            event_type="order.payment_processed",
            payload={"order_id": order.order_id},
            source=AgentType.PAYMENT,
        )
    )
    for event_type in (
        "fulfillment.picked",
        "fulfillment.packed",
        "fulfillment.shipped",
        "order.delivered",
        "order.completed",
    ):
        await bus.publish(
            RetailEvent(
                event_type=event_type, payload={"order_id": order.order_id}, source=AgentType.FULFILLMENT
            )
        )
    assert trace.count("order.allocated") == 1 and attempts == 2
    assert master.orders[order.order_id]["current_status"] == "completed"
    print("Synthetic success/recovery trace:", " -> ".join(trace))
    return trace


# region book:order-orchestration-demo
if __name__ == "__main__":
    import asyncio

    from demos.order_orchestration_demo import run_orchestration_simulation

    asyncio.run(run_orchestration_simulation())
# endregion book:order-orchestration-demo
