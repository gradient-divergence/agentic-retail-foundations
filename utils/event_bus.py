"""
Simple asynchronous event bus for agent communication.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

# Assuming RetailEvent is defined in models.events
from models.events import RetailEvent

logger_event_bus = logging.getLogger(__name__)  # Use a specific logger


class EventBus:
    """Simple event bus for agent communication (Extracted from notebook)"""

    def __init__(self):
        self.subscribers: dict[str, list[Callable[[RetailEvent], Coroutine[Any, Any, None]]]] = {}

    def subscribe(
        self,
        event_type: str,
        callback: Callable[[RetailEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to an event type."""
        if not callable(callback):
            raise TypeError("Callback must be a callable async function.")
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        if callback not in self.subscribers[event_type]:  # Avoid duplicate subscriptions
            self.subscribers[event_type].append(callback)
            logger_event_bus.debug(f"Callback {callback.__name__} subscribed to {event_type}")
        else:
            logger_event_bus.warning(f"Callback {callback.__name__} already subscribed to {event_type}")

    def unsubscribe(
        self,
        event_type: str,
        callback: Callable[[RetailEvent], Coroutine[Any, Any, None]],
    ) -> None:
        """Unsubscribe a specific callback from an event type."""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(callback)
                logger_event_bus.debug(f"Callback {callback.__name__} unsubscribed from {event_type}")
                if not self.subscribers[event_type]:  # Clean up empty list
                    del self.subscribers[event_type]
            except ValueError:
                logger_event_bus.warning(f"Callback {callback.__name__} not found for event type {event_type}")

    async def publish(self, event: RetailEvent) -> None:
        """Publish an event to subscribers."""
        if not isinstance(event, RetailEvent):
            logger_event_bus.error(f"Attempted to publish invalid event type: {type(event)}")
            return

        logger_event_bus.info(f"Event published: {event.event_type} from {event.source.value}")
        if event.event_type in self.subscribers:
            # Gather tasks to run handlers concurrently
            tasks = [asyncio.create_task(callback(event)) for callback in self.subscribers[event.event_type]]
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        callback_name = self.subscribers[event.event_type][i].__name__
                        logger_event_bus.error(
                            f"Error in subscriber callback '{callback_name}' for event {event.event_type}: {result}",
                            exc_info=False,
                        )
