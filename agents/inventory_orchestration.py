"""
Inventory Agent for event-driven updates (orchestration context).

This agent listens for validated orders and allocates inventory.
Distinguish from agents/inventory.py which implements (s, S) policy.
"""

import logging
from typing import Any

# Import Base Agent, Models, and Utilities
from .base import BaseAgent
from models.enums import AgentType, FulfillmentMethod
from models.events import RetailEvent
from models.fulfillment import Order  # Assuming Order might be fetched
from utils.event_bus import EventBus
import asyncio

logger = logging.getLogger(__name__)


class InventoryAgent(BaseAgent):
    """Agent responsible for inventory allocation (Event-driven version from notebook)"""

    def __init__(self, agent_id: str, event_bus: EventBus):
        super().__init__(agent_id, AgentType.INVENTORY, event_bus)
        # In a real implementation, this would connect to inventory systems/DB
        # self.inventory_db = ...
        self.register_event_handlers()  # Register handlers on init

    def register_event_handlers(self) -> None:
        """Register for events this agent cares about."""
        self.event_bus.subscribe("order.validated", self.handle_order_validated)
        # Could also listen for inventory updates, transfers, etc.

    async def handle_order_validated(self, event: RetailEvent) -> None:
        """Handle validated order by performing inventory allocation."""
        order_id = event.payload.get("order_id")
        if not order_id:
            logger.error("Missing order_id in validated order event")
            return

        # In a real implementation, fetch order details to know items, preferences etc.
        order = await self._get_order(order_id)
        if not order:
            logger.error(f"Order {order_id} not found when handling validation.")
            # Publish an exception event?
            return

        try:
            logger.info(f"Allocating inventory for validated order {order_id}...")
            await self.allocate_inventory(order)
            # Publish allocation success event (could include allocation details)
            await self.publish_event(
                "order.allocated",
                {
                    "order_id": order.order_id,
                    "allocation_details": self._get_allocation_details(order),
                },
            )
            logger.info(f"Inventory allocated successfully for order {order_id}.")

        except Exception as e:
            logger.error(f"Error allocating inventory for order {order_id}: {e}")
            # Use the base class exception handler
            await self.handle_exception(
                e, {"stage": "inventory_allocation", "order_id": order_id}, order=order
            )

    async def _get_order(self, order_id: str) -> Order | None:
        """Mock implementation to get order details."""
        # Fetch from a database/order service in reality
        logger.warning(
            f"_get_order is a mock. Returning None for order {order_id}. Implement real fetching."
        )
        return None

    def _get_allocation_details(self, order: Order) -> dict[str, Any]:
        """Helper to extract allocation details from the order object for the event payload."""
        # Assuming allocate_inventory modifies order.items directly
        if not hasattr(order, "items"):
            return {}
        return {
            "items": [
                {
                    "product_id": item.product_id,
                    "quantity": item.quantity,
                    "fulfillment_method": item.fulfillment_method.value
                    if item.fulfillment_method
                    else None,
                    "location_id": item.fulfillment_location_id,
                }
                for item in order.items
                if hasattr(item, "product_id")  # Basic check
            ]
        }

    async def allocate_inventory(self, order: Order) -> None:
        """
        Allocate inventory for an order.
        This is a placeholder. Real logic would check inventory levels,
        apply rules, reserve stock, and update order items.
        """
        logger.debug(
            f"Simulating inventory allocation logic for order {order.order_id}..."
        )
        # Logic to determine optimal fulfillment locations & methods based on:
        # - Current stock levels across locations (stores, warehouses)
        # - Order preferences (pickup, ship, delivery address)
        # - Business rules (cost, speed, store capacity)
        # - Real-time availability
        # - Reserving inventory atomistically

        # ---- Placeholder Logic ----
        if not hasattr(order, "items") or not isinstance(order.items, list):
            logger.error(f"Cannot allocate: Order {order.order_id} has invalid items.")
            raise ValueError(f"Invalid items in order {order.order_id}")

        preferred_method = order.preferred_fulfillment_method
        pickup_location = order.pickup_store_id
        delivery_available = order.delivery_address is not None

        for item in order.items:
            # Simple mock logic:
            # Prefer pickup if specified, else ship from warehouse, else ship from store A
            if (
                preferred_method == FulfillmentMethod.PICKUP_IN_STORE
                and pickup_location
            ):
                item.fulfillment_method = FulfillmentMethod.PICKUP_IN_STORE
                item.fulfillment_location_id = pickup_location
            elif item.product_id == "PROD-001":  # Assume warehouse only has PROD-001
                item.fulfillment_method = FulfillmentMethod.SHIP_FROM_WAREHOUSE
                item.fulfillment_location_id = "WAREHOUSE_01"
            else:  # Default to shipping from Store A (if available)
                item.fulfillment_method = FulfillmentMethod.SHIP_FROM_STORE
                item.fulfillment_location_id = (
                    "STORE_A"  # Need to verify STORE_A exists/has stock
                )
            logger.debug(
                f"  Allocated item {item.product_id} to {item.fulfillment_method.value} from {item.fulfillment_location_id}"
            )
        # ---- End Placeholder ----

        # In reality, update inventory DB/Cache to reserve stock here
        # update_inventory_reservations(order)

        # # Update order status directly (Alternative: publish event)
        # order.update_status(OrderStatus.ALLOCATED, self.agent_type, {"allocation_strategy": "mock_default"})

        await asyncio.sleep(0.05)  # Simulate allocation time
