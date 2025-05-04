"""
Agent communication protocol classes for FIPA-inspired messaging in retail multi-agent systems.
"""

from enum import Enum
from typing import Any, Dict, Optional, Set, Coroutine
from collections.abc import Callable, Awaitable
from datetime import datetime
import uuid
from collections import defaultdict
import asyncio

# Import the data models from the models directory
from models.messaging import AgentMessage, Performative


class MessageBroker:
    """
    Message broker that routes messages between agents, supporting direct and topic-based delivery.
    Handles both persistent and one-time message handlers.
    """

    def __init__(self):
        # Stores agent_id -> persistent handler mapping
        self._primary_handlers: Dict[str, Callable[[AgentMessage], Coroutine[Any, Any, None]]] = {}
        # Stores agent_id -> the next one-time handler (if any)
        self._one_time_handlers: Dict[str, Callable[[AgentMessage], Coroutine[Any, Any, None]]] = {}
        # Stores topic -> set of subscriber agent_ids
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)

    def register_agent(
        self, agent_id: str, handler_func: Callable[[AgentMessage], Coroutine[Any, Any, None]]
    ):
        """
        Register an agent with its primary message handler.
        Overwrites existing primary handler for the same agent_id.
        """
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        if not callable(handler_func):
            raise TypeError("handler must be a callable async function")
        
        self._primary_handlers[agent_id] = handler_func
        print(f"Agent {agent_id} registered with primary handler.")

    def unregister_agent(self, agent_id: str):
        """
        Remove an agent and its handlers from the broker and subscriptions.
        """
        if agent_id in self._primary_handlers:
            del self._primary_handlers[agent_id]
            print(f"Removed primary handler for {agent_id}.")
        if agent_id in self._one_time_handlers:
            del self._one_time_handlers[agent_id]
            print(f"Removed one-time handler for {agent_id}.")
            
        # Also remove from any subscriptions
        for topic in list(self._subscriptions.keys()):
            if agent_id in self._subscriptions[topic]:
                self._subscriptions[topic].remove(agent_id)
                if not self._subscriptions[topic]:  # Clean up empty topic lists
                    del self._subscriptions[topic]
        print(f"Agent {agent_id} fully unregistered.")

    def register_one_time_handler(
        self,
        agent_id: str,
        handler: Callable[[AgentMessage], Coroutine[Any, Any, None]]
    ):
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
        print(f"Registered one-time handler for agent {agent_id}.")

    def subscribe(self, agent_id: str, topic: str):
        """
        Subscribe an agent to a topic. Idempotent.
        """
        if not agent_id or not topic:
            raise ValueError("agent_id and topic cannot be empty")
        # Agent must be registered to subscribe (have a primary handler)
        if agent_id not in self._primary_handlers:
            print(f"Warning: Agent {agent_id} must be registered before subscribing to topics.")
            return
            
        self._subscriptions[topic].add(agent_id)
        print(f"Agent {agent_id} subscribed to topic {topic}.")

    def unsubscribe(self, agent_id: str, topic: str):
        """
        Unsubscribe an agent from a topic.
        """
        if topic in self._subscriptions:
            self._subscriptions[topic].discard(agent_id) # Use discard to avoid KeyError
            if not self._subscriptions[topic]: # Clean up empty topic lists
                del self._subscriptions[topic]
            print(f"Agent {agent_id} unsubscribed from topic {topic}.")

    async def deliver_message(self, msg: AgentMessage):
        """
        Deliver a message to a direct recipient or all subscribers of a topic.
        Checks for and executes one-time handlers first, then primary handlers.
        """
        if not isinstance(msg, AgentMessage):
            print(f"Error: Invalid message type received: {type(msg)}")
            return

        receiver_id = msg.receiver
        print(f"Broker attempting delivery: {msg.sender} -> {msg.receiver} ({msg.performative.name if msg.performative else 'N/A'})", flush=True)

        # Helper function to execute handler for a specific agent_id
        async def _execute_handler(agent_id: str, message: AgentMessage):
            executed = False
            # Prioritize one-time handler
            if agent_id in self._one_time_handlers:
                handler_to_run = self._one_time_handlers.pop(agent_id) # Get and remove
                try:
                    print(f"  Executing one-time handler for {agent_id}...", flush=True)
                    await handler_to_run(message)
                    executed = True
                except Exception as e:
                    print(f"  Error in one-time handler for {agent_id}: {e}", flush=True)
            # If no one-time handler was executed, try the primary handler
            elif agent_id in self._primary_handlers:
                handler_to_run = self._primary_handlers[agent_id]
                try:
                    print(f"  Executing primary handler for {agent_id}...", flush=True)
                    await handler_to_run(message)
                    executed = True
                except Exception as e:
                    print(f"  Error in primary handler for {agent_id}: {e}", flush=True)
            
            if not executed:
                print(f"  Warning: No handler found or executed for agent {agent_id}", flush=True)

        # --- Delivery Logic --- 
        if receiver_id.startswith("topic:"):
            topic = receiver_id.split(":", 1)[1]
            if topic in self._subscriptions:
                # Create copy in case subscriptions change during iteration
                subscribers = list(self._subscriptions[topic]) 
                print(f"  Delivering to topic '{topic}' subscribers: {subscribers}", flush=True)
                tasks = [_execute_handler(sub_id, msg) for sub_id in subscribers]
                if tasks:
                    await asyncio.gather(*tasks)
            else:
                print(f"  No subscribers for topic '{topic}'", flush=True)
        else:
            # Direct delivery
            await _execute_handler(receiver_id, msg)
