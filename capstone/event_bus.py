from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from capstone.schemas import EventEnvelope

logger_event_bus = logging.getLogger(__name__)


class EventBus:
    """Lightweight async event bus that uses the capstone event envelope."""

    def __init__(self) -> None:
        self.subscribers: dict[str, list[Callable[[EventEnvelope], Coroutine[Any, Any, None]]]] = {}

    def subscribe(
        self,
        event_type: str,
        callback: Callable[[EventEnvelope], Coroutine[Any, Any, None]],
    ) -> None:
        if not callable(callback):
            raise TypeError("Callback must be a callable async function.")
        self.subscribers.setdefault(event_type, [])
        if callback in self.subscribers[event_type]:
            logger_event_bus.warning("Callback %s already subscribed to %s", callback.__name__, event_type)
            return
        self.subscribers[event_type].append(callback)
        logger_event_bus.debug("Callback %s subscribed to %s", callback.__name__, event_type)

    async def publish(self, event: EventEnvelope) -> None:
        if not isinstance(event, EventEnvelope):
            logger_event_bus.error("Attempted to publish invalid event type: %s", type(event))
            return

        logger_event_bus.info("Event published: %s", event.event_type)
        callbacks = self.subscribers.get(event.event_type, [])
        if not callbacks:
            return

        tasks: list[asyncio.Task[None]] = [asyncio.create_task(callback(event)) for callback in callbacks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for callback, result in zip(callbacks, results, strict=False):
            if isinstance(result, Exception):
                logger_event_bus.error(
                    "Error in subscriber callback '%s' for event %s: %s",
                    callback.__name__,
                    event.event_type,
                    result,
                    exc_info=False,
                )
