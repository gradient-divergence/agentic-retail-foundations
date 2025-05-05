"""
Base class for retail agents.
"""

import logging
from typing import Any

# Import necessary models and utilities
from models.enums import AgentType, OrderStatus
from models.events import RetailEvent
from models.fulfillment import Order  # Assuming Order is used by handle_exception
from utils.event_bus import EventBus  # Assuming EventBus is used

logger_base = logging.getLogger(__name__)  # Use a specific logger


class BaseAgent:
    """Base class for all retail agents (Extracted from notebook)"""

    def __init__(self, agent_id: str, agent_type: AgentType, event_bus: EventBus):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.event_bus = event_bus
        # Call registration in subclass __init__ after setting up handlers
        # self.register_event_handlers()

    # Subclasses should override this to register for specific events
    def register_event_handlers(self) -> None:
        """Register for events this agent cares about"""
        pass

    async def publish_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Publish an event to the event bus"""
        if not hasattr(self, "event_bus") or self.event_bus is None:
            logger_base.error(f"Agent {self.agent_id} has no event bus to publish to.")
            return

        event = RetailEvent(
            event_type=event_type, payload=payload, source=self.agent_type
        )
        await self.event_bus.publish(event)

    # Generic exception handler - can be overridden by subclasses
    async def handle_exception(
        self,
        exception: Exception,
        context: dict[str, Any],
        order: Order | None = None,  # Make order optional
    ) -> None:
        """Handle exceptions during processing"""
        error_details = {
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "context": context,
            "agent_id": self.agent_id,
        }
        logger_base.error(
            f"Exception in {self.agent_type.value} agent ({self.agent_id}): {str(exception)}",
            exc_info=True,  # Include traceback in log
        )

        # Update order status if an order object is provided
        if (
            order is not None
            and hasattr(order, "update_status")
            and hasattr(order, "order_id")
        ):
            try:
                order.update_status(
                    OrderStatus.EXCEPTION, self.agent_type, error_details
                )
                error_details["order_id"] = order.order_id  # Add order_id if updated
            except Exception as update_err:
                logger_base.error(
                    f"Failed to update order status during exception handling for order {getattr(order, 'order_id', 'N/A')}: {update_err}"
                )

        # Publish exception event
        # Ensure payload keys match expected structure if defined elsewhere
        await self.publish_event("system.exception", {"error_details": error_details})
