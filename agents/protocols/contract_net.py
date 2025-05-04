"""
Contract Net Protocol (CNP) for retail agents.

This module implements the Contract Net Protocol for task allocation in retail settings.
The CNP is a negotiation protocol used to solve distributed problem solving tasks.
"""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import random
import asyncio
from dataclasses import asdict

from models.task import Task, TaskStatus, Bid
from models.messaging import AgentMessage, Performative


class RetailCoordinator:
    """
    Implements the Contract Net Protocol for retail task allocation.
    Acts as the initiator/manager in CNP negotiations.
    """

    def __init__(self, coordinator_id: str, name: str):
        """
        Initialize a retail coordinator.

        Args:
            coordinator_id: Unique identifier for this coordinator
            name: Human-readable name
        """
        self.coordinator_id = coordinator_id
        self.name = name
        self.tasks: Dict[str, Task] = {}
        self.bids: Dict[str, List[Bid]] = {}  # task_id -> list of bids
        self.message_handlers: Dict[Performative, Callable] = {}
        self.participant_ids: List[str] = []
        self.task_history: List[Dict[str, Any]] = []

    def register_participant(self, participant_id: str) -> None:
        """
        Register a participant agent with the coordinator.

        Args:
            participant_id: ID of the participant to register
        """
        if participant_id not in self.participant_ids:
            self.participant_ids.append(participant_id)

    def register_message_handler(
        self, performative: Performative, handler: Callable[[AgentMessage], None]
    ) -> None:
        """
        Register a handler for a specific message performative.

        Args:
            performative: The type of message to handle
            handler: Callback function that processes the message
        """
        self.message_handlers[performative] = handler

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message based on its performative.

        Args:
            message: The message to process

        Returns:
            Optional response message
        """
        if message.performative in self.message_handlers:
            # Assume handler returns the correct type, add assertion if needed
            # for more robust checking, or ensure handlers are typed correctly.
            result = self.message_handlers[message.performative](message)
            # If unsure about handler return types, could assert here:
            # assert isinstance(result, (AgentMessage, type(None))), f"Handler for {message.performative} returned wrong type"
            return result # type: ignore[no-any-return]
        return None

    async def announce_task(self, task: Task) -> List[str]:
        """
        Announce a task to all participants and collect their bids.

        Args:
            task: The task to announce

        Returns:
            List of participant IDs that the task was announced to
        """
        self.tasks[task.id] = task
        self.bids[task.id] = []

        # In a real implementation, this would send actual messages
        # to participants and wait for responses

        # For simulation purposes, we'll just record that the task was announced
        announced_to = self.participant_ids.copy()
        task_record = {
            "timestamp": datetime.now(),
            "task_id": task.id,
            "action": "announced",
            "participants": announced_to,
        }
        self.task_history.append(task_record)

        return announced_to

    def handle_bid(self, bid: Bid) -> None:
        """
        Handle a bid from a participant.

        Args:
            bid: The bid to process
        """
        if bid.task_id not in self.tasks:
            return  # Ignore bids for unknown tasks

        if bid.task_id not in self.bids:
            self.bids[bid.task_id] = []

        self.bids[bid.task_id].append(bid)

        # Record the bid
        task_record = {
            "timestamp": datetime.now(),
            "task_id": bid.task_id,
            "action": "bid_received",
            "participant_id": bid.agent_id,
            "bid_value": bid.bid_value,
            "estimated_completion_time": bid.estimated_completion_time,
        }
        self.task_history.append(task_record)

    async def award_task(self, task_id: str) -> Optional[Bid]:
        """
        Evaluate bids and award the task to the best bidder.

        Args:
            task_id: ID of the task to award

        Returns:
            The winning bid, if any
        """
        if task_id not in self.tasks or task_id not in self.bids:
            return None

        if not self.bids[task_id]:
            # No bids received
            self.tasks[task_id].status = TaskStatus.FAILED
            return None

        # Select the best bid (lowest bid_value)
        best_bid = min(self.bids[task_id], key=lambda b: b.bid_value)

        # Update task status
        self.tasks[task_id].status = TaskStatus.ALLOCATED
        self.tasks[task_id].assigned_agent_id = best_bid.agent_id

        # Record the award
        task_record = {
            "timestamp": datetime.now(),
            "task_id": task_id,
            "action": "awarded",
            "participant_id": best_bid.agent_id,
            "bid_value": best_bid.bid_value,
        }
        self.task_history.append(task_record)

        return best_bid

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the current status of a task.

        Args:
            task_id: ID of the task to check

        Returns:
            Current status of the task, if it exists
        """
        if task_id not in self.tasks:
            return None
        return self.tasks[task_id].status

    def update_task_status(self, task_id: str, new_status: TaskStatus) -> bool:
        """
        Update the status of a task.

        Args:
            task_id: ID of the task to update
            new_status: New status to set

        Returns:
            True if update was successful, False otherwise
        """
        if task_id not in self.tasks:
            return False

        old_status = self.tasks[task_id].status
        self.tasks[task_id].status = new_status

        # Record the status change
        task_record = {
            "timestamp": datetime.now(),
            "task_id": task_id,
            "action": "status_change",
            "old_status": old_status.value,
            "new_status": new_status.value,
        }
        self.task_history.append(task_record)

        return True
