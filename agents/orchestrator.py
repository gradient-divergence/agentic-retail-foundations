"""
Master Orchestrator Agent for coordinating complex retail processes.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from models.enums import AgentType, OrderStatus  # OrderStatus needed?
from models.events import RetailEvent
from utils.event_bus import EventBus

# Import base class, models, and utilities
from .base import BaseAgent

logger = logging.getLogger(__name__)


class MasterOrchestrator(BaseAgent):
    """Centralized orchestrator for end-to-end order process (Extracted from notebook)"""

    def __init__(self, agent_id: str, event_bus: EventBus):
        super().__init__(agent_id, AgentType.MASTER, event_bus)
        # Track all orders and their current state
        self.orders: dict[str, dict[str, Any]] = {}
        self.register_event_handlers()  # Register handlers on init

    def register_event_handlers(self) -> None:
        """Register for all order-related events for monitoring"""
        event_types = [
            "order.created",
            "order.validated",
            "order.allocated",
            "order.payment_processed",
            "order.exception",  # Listen for exceptions published by other agents
            "fulfillment.requested",
            "fulfillment.picked",
            "fulfillment.packed",
            "fulfillment.shipped",
            "order.delivered",
            "order.completed",
            "order.cancelled",
            "payment.retry_requested",  # Listen for specific recovery events
            "payment.alternative_requested",
            "inventory.reallocation_requested",
            "support.ticket_created",
            "notification.sent",
            # Add any other relevant system events
        ]

        for event_type in event_types:
            self.event_bus.subscribe(event_type, self.handle_order_event)

        # Special handling for exceptions (might be redundant if handle_order_event covers it)
        # self.event_bus.subscribe("order.exception", self.handle_exception_event)

    async def handle_order_event(self, event: RetailEvent) -> None:
        """Track all order events to maintain global state and potentially trigger actions."""
        # Basic tracking logic from notebook
        order_id = event.payload.get("order_id")
        if not order_id:
            # Log warning only if the event type *usually* has an order_id
            if event.event_type.startswith("order.") or event.event_type.startswith("fulfillment."):
                logger.warning(f"Event missing order_id: {event.event_type} from {event.source.value}")
            return

        # Update tracking state
        if order_id not in self.orders:
            self.orders[order_id] = {
                "events": [],
                "last_update": None,
                "current_status": None,
            }

        # Add event to history
        self.orders[order_id]["events"].append(
            {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "source": event.source.value,
                "payload": event.payload,  # Store payload for context
            }
        )
        self.orders[order_id]["last_update"] = event.timestamp

        # Update status based on event type heuristics
        if event.event_type.startswith("order."):
            status = event.event_type.split(".", 1)[1]
            # Map event type to OrderStatus enum if possible, otherwise keep as string
            try:
                order_status_enum = OrderStatus(status)
                self.orders[order_id]["current_status"] = order_status_enum.value
            except ValueError:
                self.orders[order_id]["current_status"] = status  # Store raw status if not in enum
        elif event.event_type == "fulfillment.shipped":
            self.orders[order_id]["current_status"] = OrderStatus.SHIPPED.value
        # Add other status mappings as needed...

        # Log for monitoring
        logger.info(f"Order {order_id} - Event: {event.event_type} from {event.source.value}. Status: {self.orders[order_id].get('current_status')}")

        # Handle specific events that require orchestrator action
        if event.event_type == "order.exception":
            await self.handle_exception_event(event)
        elif event.event_type == "order.stalled":  # Assuming another process detects stall
            await self._apply_recovery_strategy(order_id, event)  # Or specific stall handling

        # Periodic check for stalled orders (could run as a separate task)
        # await self._check_for_stalled_orders()

    async def handle_exception_event(self, event: RetailEvent) -> None:
        """Handle exception events with specific logic."""
        order_id = event.payload.get("order_id")
        if not order_id:
            logger.error("Exception event missing order_id")
            return

        error_details = event.payload.get("error_details", {})
        error_type = error_details.get("error_type", "unknown")
        source_agent = event.source.value if event.source else "unknown_source"

        # Update tracking state (if handle_order_event didn't already)
        if order_id in self.orders:
            self.orders[order_id]["current_status"] = "exception"
            self.orders[order_id]["exception_details"] = error_details

        # Log the exception
        logger.error(f"Order {order_id} - Exception: {error_type} from {source_agent}")

        # Apply recovery strategy based on exception type and context
        await self._apply_recovery_strategy(order_id, event)

    # Note: This check might be better implemented as a separate periodic task
    #       rather than being called on every single event.
    async def _check_for_stalled_orders(self) -> None:
        """Identify and potentially resolve stalled orders."""
        now = datetime.now()
        threshold = timedelta(minutes=30)  # Use timedelta for comparison

        stalled_orders = []
        for order_id, details in self.orders.items():
            last_update_iso = details.get("last_update")
            if not last_update_iso:
                continue
            try:
                last_update = datetime.fromisoformat(last_update_iso)
                # Make timezone-aware if necessary, assuming naive for now
                elapsed = now - last_update

                if elapsed > threshold and details.get("current_status") not in [
                    OrderStatus.COMPLETED.value,
                    OrderStatus.CANCELLED.value,
                    OrderStatus.EXCEPTION.value,
                ]:
                    stalled_orders.append(order_id)
                    logger.warning(f"Order {order_id} appears stalled in status '{details.get('current_status')}' (last update: {last_update_iso})")
            except ValueError:
                logger.error(f"Invalid timestamp format for order {order_id}: {last_update_iso}")

        # Trigger recovery or alerts for stalled orders
        for order_id in stalled_orders:
            await self.publish_event(
                "order.stalled_alert",
                {
                    "order_id": order_id,
                    "current_status": self.orders[order_id].get("current_status"),
                    "last_update": self.orders[order_id].get("last_update"),
                },
            )
            # Optionally, directly trigger recovery here
            # await self._apply_recovery_strategy(order_id, None) # Pass None or a specific stall event

    async def _apply_recovery_strategy(
        self,
        order_id: str,
        event: RetailEvent | None,  # Event can be None if triggered internally
    ) -> None:
        """Apply recovery strategy based on error context."""
        error_details = event.payload.get("error_details", {}) if event else {}
        error_type = error_details.get("error_type", "unknown")
        error_context = error_details.get("context", {})
        source_agent_type = event.source if event else None

        logger.info(f"Applying recovery strategy for order {order_id}. Error type: {error_type}, Source: {source_agent_type}")

        # Example recovery logic
        if source_agent_type == AgentType.INVENTORY and "allocation" in error_context.get("stage", ""):
            await self._handle_inventory_allocation_failure(order_id)
        elif source_agent_type == AgentType.PAYMENT:
            await self._handle_payment_failure(order_id, error_type)
        else:
            # Generic fallback: Escalate to human
            logger.warning(f"Unknown error source/context for order {order_id}. Escalating.")
            await self._escalate_to_human(order_id, error_details)

    async def _handle_inventory_allocation_failure(self, order_id: str) -> None:
        """Handle inventory allocation failures by requesting reallocation."""
        logger.info(f"Requesting inventory reallocation for order {order_id}.")
        await self.publish_event(
            "inventory.reallocation_requested",
            {
                "order_id": order_id,
                "allow_substitutions": True,  # Example policy
                "try_alternative_methods": True,
            },
        )

    async def _handle_payment_failure(self, order_id: str, error_type: str) -> None:
        """Handle payment processing failures."""
        logger.info(f"Handling payment failure for order {order_id} (Error: {error_type}).")
        if error_type in [
            "TemporaryProcessingError",
            "GatewayTimeout",
            "InsufficientFunds",
        ]:
            # Transient error or solvable error, retry payment after delay
            logger.info(f"Requesting payment retry for order {order_id}.")
            await self.publish_event(
                "payment.retry_requested",
                {"order_id": order_id, "retry_count": 1, "delay_minutes": 5},
            )
        else:
            # Permanent error (e.g., invalid card), request alternative payment
            logger.warning(f"Requesting alternative payment for order {order_id}.")
            await self.publish_event(
                "payment.alternative_requested",
                {"order_id": order_id, "original_error": error_type},
            )

    async def _escalate_to_human(self, order_id: str, error_details: dict[str, Any]) -> None:
        """Escalate exception to human operator by creating a support ticket."""
        logger.warning(f"Escalating issue for order {order_id} to human support.")
        # Create a support ticket event
        await self.publish_event(
            "support.ticket_created",
            {
                "order_id": order_id,
                "error_details": error_details,
                "priority": "high",
                "queue": "order_exceptions",
            },
        )
        # Optionally, send notification directly as well
        # await self.publish_event(
        #     "notification.sent",
        #     {...}
        # )
