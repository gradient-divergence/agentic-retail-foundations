"""
Fulfillment Agent responsible for orchestrating order fulfillment steps.
"""

import logging

# Import Base Agent, Models, and Utilities
from .base import BaseAgent
from models.enums import AgentType, FulfillmentMethod, OrderStatus
from models.events import RetailEvent
from models.fulfillment import Order, OrderLineItem
from utils.event_bus import EventBus

logger = logging.getLogger(__name__)


class FulfillmentAgent(BaseAgent):
    """Agent responsible for order fulfillment orchestration (Extracted from notebook)"""

    def __init__(self, agent_id: str, event_bus: EventBus):
        super().__init__(agent_id, AgentType.FULFILLMENT, event_bus)
        # Add any specific state needed for the fulfillment agent
        self.register_event_handlers()  # Register handlers on init

    def register_event_handlers(self) -> None:
        """Register for events this agent cares about."""
        self.event_bus.subscribe("order.allocated", self.handle_order_allocated)
        self.event_bus.subscribe(
            "order.payment_processed", self.handle_payment_processed
        )
        # Add handlers for fulfillment updates (e.g., picked, packed, shipped)
        # self.event_bus.subscribe("fulfillment.picked", self.handle_fulfillment_picked)
        # self.event_bus.subscribe("fulfillment.packed", self.handle_fulfillment_packed)
        # self.event_bus.subscribe("fulfillment.shipped", self.handle_fulfillment_shipped)

    async def handle_order_allocated(self, event: RetailEvent) -> None:
        """Process order after inventory allocation by requesting payment."""
        order_id = event.payload.get("order_id")
        if not order_id:
            logger.error("Missing order_id in allocated order event")
            return

        # In a real system, we might fetch the order to get payment details
        # order = await self._get_order(order_id)
        # if not order:
        #     logger.error(f"Order {order_id} not found when handling allocation.")
        #     return

        try:
            logger.info(f"Order {order_id} allocated. Requesting payment processing.")
            # Initiate payment processing (payload might need total amount, customer id etc.)
            await self.publish_event(
                "payment.request_processing",  # More specific event type
                {
                    "order_id": order_id,
                    "amount": event.payload.get("order_total", 0),
                },  # Assuming total is passed or fetchable
            )
        except Exception as e:
            logger.error(f"Error initiating payment for order {order_id}: {e}")
            # Use the base class exception handler with correct arguments
            await self.handle_exception(
                exception=e,
                context={"stage": "payment_initiation", "order_id": order_id},
                order=None,
            )  # Pass None for order if not fetched

    async def handle_payment_processed(self, event: RetailEvent) -> None:
        """Handle successful payment processing by initiating fulfillment steps."""
        order_id = event.payload.get("order_id")
        if not order_id:
            logger.error("Missing order_id in payment processed event")
            return

        # Fetch the full order details now that payment is confirmed
        order = await self._get_order(order_id)
        if not order:
            logger.error(f"Order {order_id} not found when handling payment processed.")
            # Potentially publish an error event or escalate
            return

        try:
            logger.info(
                f"Payment processed for order {order_id}. Initiating fulfillment."
            )
            fulfillment_groups = self._group_items_by_fulfillment(order)
            logger.debug(
                f"Order {order_id} split into {len(fulfillment_groups)} fulfillment groups."
            )

            # Initiate fulfillment for each group
            for method, location, items in fulfillment_groups:
                await self._initiate_fulfillment_request(order, method, location, items)

            # Update order status to Picking (or appropriate starting state)
            # Ensure OrderStatus is imported or defined
            # order.update_status(OrderStatus.PICKING, self.agent_type, {"fulfillment_groups": len(fulfillment_groups)})
            # Instead of updating directly, publish an event for the status change
            await self.publish_event(
                "order.status_update_request",
                {
                    "order_id": order_id,
                    "new_status": OrderStatus.PICKING.value,
                    "details": {"fulfillment_groups": len(fulfillment_groups)},
                },
            )

        except Exception as e:
            logger.error(f"Error initiating fulfillment for order {order_id}: {e}")
            # Pass the actual order object if available
            await self.handle_exception(
                exception=e, context={"stage": "fulfillment_initiation"}, order=order
            )

    # --- Helper Methods ---

    async def _get_order(self, order_id: str) -> Order | None:
        """Mock implementation to get order details."""
        # In a real implementation, this would fetch from a database/order service
        # Returning None simulates order not found
        logger.warning(
            f"_get_order is a mock. Returning None for order {order_id}. Implement real fetching."
        )
        return None

    def _group_items_by_fulfillment(
        self, order: Order
    ) -> list[tuple[FulfillmentMethod, str, list[OrderLineItem]]]:
        """Group order items by fulfillment method and location."""
        groups: dict[tuple[FulfillmentMethod, str], list[OrderLineItem]] = {}
        # Use order.items which should be List[OrderLineItem]
        if not hasattr(order, "items") or not isinstance(order.items, list):
            logger.error(f"Order {order.order_id} has invalid items attribute.")
            return []

        for item in order.items:
            if not isinstance(item, OrderLineItem):
                logger.warning(
                    f"Skipping invalid item type in order {order.order_id}: {type(item)}"
                )
                continue

            if not item.fulfillment_method or not item.fulfillment_location_id:
                # Log error or raise exception if fulfillment details are mandatory at this stage
                logger.error(
                    f"Item {item.product_id} in order {order.order_id} missing fulfillment details."
                )
                # Optionally raise ValueError or handle gracefully depending on requirements
                continue

            key = (item.fulfillment_method, item.fulfillment_location_id)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        # Ensure type hints match the return value
        return [
            (method, location, items) for (method, location), items in groups.items()
        ]

    async def _initiate_fulfillment_request(
        self,
        order: Order,
        method: FulfillmentMethod,
        location: str,
        items: list[OrderLineItem],
    ) -> None:
        """Determine target agent and publish fulfillment request event."""

        target_agent_type: AgentType | None = None
        if method in [
            FulfillmentMethod.SHIP_FROM_STORE,
            FulfillmentMethod.PICKUP_IN_STORE,
            FulfillmentMethod.DELIVERY_FROM_STORE,
        ]:
            target_agent_type = AgentType.STORE_OPS
        elif method == FulfillmentMethod.SHIP_FROM_WAREHOUSE:
            target_agent_type = AgentType.WAREHOUSE
        # Extend with other methods like DROPSHIP -> VENDOR_AGENT etc.
        else:
            logger.error(
                f"Cannot determine target agent for unknown fulfillment method: {method} in order {order.order_id}"
            )
            # Optionally raise an error or publish an exception event
            return

        # Create fulfillment request payload
        item_details = [
            {
                "product_id": item.product_id,
                "quantity": item.quantity,
                "item_id": getattr(item, "id", None),
            }  # Add item ID if available
            for item in items
        ]

        payload = {
            "order_id": order.order_id,
            "fulfillment_group_id": f"{order.order_id}-{method.value}-{location}",  # Unique ID for this sub-task
            "fulfillment_method": method.value,
            "location_id": location,
            "items": item_details,
            "customer_id": order.customer_id,
            "delivery_address": order.delivery_address,  # Include if needed for delivery
            "target_agent_type": target_agent_type.value,  # Indicate who should handle this
        }

        logger.info(
            f"Publishing fulfillment.requested for order {order.order_id}, group {payload['fulfillment_group_id']}"
        )
        await self.publish_event("fulfillment.requested", payload)

    # Add handlers for fulfillment status updates (e.g., picked, packed)
    # async def handle_fulfillment_picked(self, event: RetailEvent): ...
    # async def handle_fulfillment_packed(self, event: RetailEvent): ...
    # async def handle_fulfillment_shipped(self, event: RetailEvent): ...
