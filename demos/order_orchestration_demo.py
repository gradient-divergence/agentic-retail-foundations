"""
Demonstration of the event-driven order orchestration framework.
"""

import asyncio
import logging

# Import refactored components
from utils.event_bus import EventBus

# Use the specific event-driven InventoryAgent
from agents.inventory_orchestration import InventoryAgent
from agents.fulfillment import FulfillmentAgent
from agents.orchestrator import MasterOrchestrator
from models.events import RetailEvent
from models.enums import AgentType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_orchestration_simulation():
    """Run a simple simulation of the orchestration framework (Extracted from notebook)"""
    logger.info("--- Starting Order Orchestration Simulation ---")
    # Create event bus
    event_bus = EventBus()
    logger.info("EventBus created.")

    # Create agents, passing the event bus
    # Make sure agent IDs are unique
    inventory_agent = InventoryAgent("inventory-agent-1", event_bus)
    fulfillment_agent = FulfillmentAgent("fulfillment-agent-1", event_bus)
    master_orchestrator = MasterOrchestrator("master-orchestrator-1", event_bus)
    logger.info("Agents created: Inventory, Fulfillment, MasterOrchestrator")

    # --- Simulate Order Flow ---

    order_id = "ORD-SIM-001"
    customer_id = "CUST-SIM-101"
    logger.info(f"Simulating creation of Order: {order_id}")

    # 1. Simulate order creation event (e.g., from a frontend or API)
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
                "total_amount": 220.0,  # Example total
                # Include other necessary order details if needed by validation
            },
            source=AgentType.CUSTOMER,  # Or perhaps an API_GATEWAY source
        )
    )
    await asyncio.sleep(0.1)  # Allow event processing

    # 2. Simulate validation completed (e.g., fraud check, basic validation passed)
    logger.info(f"Simulating validation success for Order: {order_id}")
    await event_bus.publish(
        RetailEvent(
            event_type="order.validated",
            payload={"order_id": order_id, "validation_status": "passed"},
            source=AgentType.FINANCIAL,  # Example source
        )
    )
    await asyncio.sleep(0.2)  # Allow inventory allocation + payment request

    # 3. Simulate successful payment processing (e.g., from a Payment Agent)
    logger.info(f"Simulating payment success for Order: {order_id}")
    await event_bus.publish(
        RetailEvent(
            event_type="order.payment_processed",
            payload={"order_id": order_id, "transaction_id": "TXN12345"},
            source=AgentType.PAYMENT,
        )
    )
    await asyncio.sleep(0.3)  # Allow fulfillment initiation

    # --- Add more steps as needed ---
    # e.g., Simulate fulfillment.picked, fulfillment.packed, fulfillment.shipped, order.delivered

    logger.info("--- Order Orchestration Simulation Finished --- ")
    # Print final state from orchestrator (optional)
    if (
        hasattr(master_orchestrator, "orders")
        and order_id in master_orchestrator.orders
    ):
        print(f"\nFinal tracked state for Order {order_id}:")
        import json

        print(json.dumps(master_orchestrator.orders[order_id], indent=2, default=str))
    else:
        print(f"\nFinal state for Order {order_id} not tracked by orchestrator.")


if __name__ == "__main__":
    asyncio.run(run_orchestration_simulation())
