"""Manages conversation history and concurrency for customer interactions."""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Handles storage and retrieval of conversation history and manages
    concurrency locks for individual customer conversations.
    """

    def __init__(self, max_history_per_user: int = 50):
        """
        Initializes the conversation manager.

        Args:
            max_history_per_user: The maximum number of conversation turns
                                  (user message + agent response) to keep per user.
        """
        self.max_history_per_user = max_history_per_user
        # Stores conversation history as deque([{role: str, content: str, timestamp: str}])
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.max_history_per_user))
        # Stores asyncio locks per customer_id
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        logger.info(f"ConversationManager initialized with max history: {max_history_per_user}")

    def get_lock(self, customer_id: str) -> asyncio.Lock:
        """
        Retrieves the asyncio.Lock for a given customer ID.

        Args:
            customer_id: The unique identifier for the customer.

        Returns:
            The asyncio.Lock associated with the customer ID.
        """
        # defaultdict creates the lock if it doesn't exist
        return self._locks[customer_id]

    def add_message(self, customer_id: str, role: str, content: str):
        """
        Adds a message to the conversation history for a specific customer.

        Note: Concurrency control (locking) should be handled by the caller
        using get_lock() before calling this method if concurrent writes
        for the same customer are possible.

        Args:
            customer_id: The customer's ID.
            role: The role of the message sender ('customer' or 'agent').
            content: The text content of the message.
        """
        if not customer_id:
            logger.warning("Attempted to add message with empty customer_id.")
            return

        message_entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        # defaultdict creates deque if first message for this customer
        self._history[customer_id].append(message_entry)
        logger.debug(f"Added {role} message for customer {customer_id}. History size: {len(self._history[customer_id])}")

    def get_recent_history(self, customer_id: str, n: int = 5) -> list[dict[str, Any]]:
        """
        Retrieves the most recent messages from a customer's conversation history.

        Args:
            customer_id: The customer's ID.
            n: The maximum number of recent messages to retrieve.

        Returns:
            A list of message dictionaries, ordered from oldest to newest.
        """
        if customer_id not in self._history:
            return []

        history_deque = self._history[customer_id]
        # Get the last n elements. If n > len(deque), it returns all elements.
        num_to_get = min(n, len(history_deque))
        recent_history = list(history_deque)[-num_to_get:]
        return recent_history

    def clear_history(self, customer_id: str):
        """
        Clears the conversation history for a specific customer.

        Args:
            customer_id: The customer's ID.
        """
        if customer_id in self._history:
            self._history[customer_id].clear()
            logger.info(f"Cleared conversation history for customer {customer_id}.")
        else:
            logger.debug(f"No history found to clear for customer {customer_id}.")

    def get_full_history(self, customer_id: str) -> list[dict[str, Any]]:
        """
        Retrieves the full conversation history for a customer.

        Args:
            customer_id: The customer's ID.

        Returns:
            A list of all message dictionaries for the customer.
        """
        return list(self._history.get(customer_id, deque()))
