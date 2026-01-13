"""
Agent communication protocol classes for FIPA-inspired messaging in retail multi-agent systems.
"""

# region book:agent-communication-message-broker
import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

# Import the data models from the models directory
from models.messaging import AgentMessage

logger = logging.getLogger(__name__)


class MessageBroker:
    """
    Message broker that routes messages between agents, supporting direct and topic-based delivery.
    Handles both persistent and one-time message handlers.
    """

    def __init__(self) -> None:
        # Stores agent_id -> persistent handler mapping
        self._primary_handlers: dict[str, Callable[[AgentMessage], Coroutine[Any, Any, None]]] = {}
        # Stores agent_id -> the next one-time handler (if any)
        self._one_time_handlers: dict[str, Callable[[AgentMessage], Coroutine[Any, Any, None]]] = {}
        # Stores topic -> set of subscriber agent_ids
        self._subscriptions: dict[str, set[str]] = defaultdict(set)

    def register_agent(
        self,
        agent_id: str,
        handler_func: Callable[[AgentMessage], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Register an agent with its primary message handler.
        Overwrites existing primary handler for the same agent_id.
        """
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        if not callable(handler_func):
            raise TypeError("handler must be a callable async function")

        self._primary_handlers[agent_id] = handler_func
        logger.info("Agent %s registered with primary handler.", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """
        Remove an agent and its handlers from the broker and subscriptions.
        """
        if agent_id in self._primary_handlers:
            del self._primary_handlers[agent_id]
            logger.info("Removed primary handler for %s.", agent_id)
        if agent_id in self._one_time_handlers:
            del self._one_time_handlers[agent_id]
            logger.info("Removed one-time handler for %s.", agent_id)

        # Also remove from any subscriptions
        for topic in list(self._subscriptions.keys()):
            if agent_id in self._subscriptions[topic]:
                self._subscriptions[topic].remove(agent_id)
                if not self._subscriptions[topic]:  # Clean up empty topic lists
                    del self._subscriptions[topic]
        logger.info("Agent %s fully unregistered.", agent_id)

    def register_one_time_handler(
        self,
        agent_id: str,
        handler: Callable[[AgentMessage], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Register a handler that will be called only once for the next message
        received by the specified agent_id, then automatically removed.
        Useful for handling specific replies in a conversation.
        This handler takes precedence over the primary handler for the next message.
        """
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        if not callable(handler):
            raise TypeError("handler must be a callable async function")

        self._one_time_handlers[agent_id] = handler
        logger.info("Registered one-time handler for agent %s.", agent_id)

    def subscribe(self, agent_id: str, topic: str) -> None:
        """
        Subscribe an agent to a topic. Idempotent.
        """
        if not agent_id or not topic:
            raise ValueError("agent_id and topic cannot be empty")
        # Agent must be registered to subscribe (have a primary handler)
        if agent_id not in self._primary_handlers:
            logger.warning("Agent %s must be registered before subscribing to topics.", agent_id)
            return

        self._subscriptions[topic].add(agent_id)
        logger.info("Agent %s subscribed to topic %s.", agent_id, topic)

    def unsubscribe(self, agent_id: str, topic: str) -> None:
        """
        Unsubscribe an agent from a topic.
        """
        if topic in self._subscriptions:
            self._subscriptions[topic].discard(agent_id)  # Use discard to avoid KeyError
            if not self._subscriptions[topic]:  # Clean up empty topic lists
                del self._subscriptions[topic]
            logger.info("Agent %s unsubscribed from topic %s.", agent_id, topic)

    async def deliver_message(self, msg: AgentMessage) -> None:  # noqa: C901
        """
        Deliver a message to a direct recipient or all subscribers of a topic.
        Checks for and executes one-time handlers first, then primary handlers.
        """
        if not isinstance(msg, AgentMessage):
            logger.error("Invalid message type received: %s", type(msg))
            return

        receiver_id = msg.receiver
        logger.info(
            "Broker attempting delivery: %s -> %s (%s)",
            msg.sender,
            msg.receiver,
            msg.performative.name if msg.performative else "N/A",
        )

        # Helper function to execute handler for a specific agent_id
        async def _execute_handler(agent_id: str, message: AgentMessage) -> None:
            executed = False
            # Prioritize one-time handler
            if agent_id in self._one_time_handlers:
                handler_to_run = self._one_time_handlers.pop(agent_id)  # Get and remove
                try:
                    logger.info("Executing one-time handler for %s...", agent_id)
                    await handler_to_run(message)
                    executed = True
                except Exception:
                    logger.exception("Error in one-time handler for %s.", agent_id)
            # If no one-time handler was executed, try the primary handler
            elif agent_id in self._primary_handlers:
                handler_to_run = self._primary_handlers[agent_id]
                try:
                    logger.info("Executing primary handler for %s...", agent_id)
                    await handler_to_run(message)
                    executed = True
                except Exception:
                    logger.exception("Error in primary handler for %s.", agent_id)

            if not executed:
                logger.warning("No handler found or executed for agent %s.", agent_id)

        # --- Delivery Logic ---
        if receiver_id.startswith("topic:"):
            topic = receiver_id.split(":", 1)[1]
            if topic in self._subscriptions:
                # Create copy in case subscriptions change during iteration
                subscribers = list(self._subscriptions[topic])
                logger.info("Delivering to topic '%s' subscribers: %s", topic, subscribers)
                tasks = [_execute_handler(sub_id, msg) for sub_id in subscribers]
                if tasks:
                    await asyncio.gather(*tasks)
            else:
                logger.warning("No subscribers for topic '%s'.", topic)
        else:
            # Direct delivery
            await _execute_handler(receiver_id, msg)


# endregion book:agent-communication-message-broker
