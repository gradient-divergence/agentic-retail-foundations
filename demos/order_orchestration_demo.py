"""
Demonstration of the event-driven order orchestration framework.
"""

# region book:order-orchestration-demo
import asyncio

from agents.fulfillment import FulfillmentAgent
from agents.inventory_orchestration import InventoryAgent
from agents.orchestrator import MasterOrchestrator
from models.enums import AgentType
from models.events import RetailEvent
from utils.event_bus import EventBus


async def run_orchestration_simulation():
    """Run a short simulation of the orchestration framework."""
    event_bus = EventBus()

    _inventory_agent = InventoryAgent("inventory-agent-1", event_bus)
    _fulfillment_agent = FulfillmentAgent("fulfillment-agent-1", event_bus)
    _master_orchestrator = MasterOrchestrator("master-orchestrator-1", event_bus)

    order_id = "ORD-SIM-001"
    customer_id = "CUST-SIM-101"

    await event_bus.publish(
        RetailEvent(
            event_type="order.created",
            payload={
                "order_id": order_id,
                "customer_id": customer_id,
                "items": [
                    {"product_id": "PROD-001", "quantity": 2, "price": 50.0},
                    {"product_id": "PROD-007", "quantity": 1, "price": 120.0},
                ],
                "total_amount": 220.0,
            },
            source=AgentType.CUSTOMER,
        )
    )
    await asyncio.sleep(0.1)

    await event_bus.publish(
        RetailEvent(
            event_type="order.validated",
            payload={"order_id": order_id, "validation_status": "passed"},
            source=AgentType.FINANCIAL,
        )
    )
    await asyncio.sleep(0.2)

    await event_bus.publish(
        RetailEvent(
            event_type="order.payment_processed",
            payload={"order_id": order_id, "transaction_id": "TXN12345"},
            source=AgentType.PAYMENT,
        )
    )
    await asyncio.sleep(0.3)


if __name__ == "__main__":
    asyncio.run(run_orchestration_simulation())

# endregion book:order-orchestration-demo
