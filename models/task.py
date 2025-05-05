"""
Data models related to tasks and bidding in coordination protocols like Contract Net.
"""

from enum import Enum
from dataclasses import dataclass, field
import uuid
from typing import Any
import time


class TaskStatus(Enum):
    """
    Enum for the status of a task in coordination protocols.
    """

    PENDING = "PENDING"  # Task created but not yet announced/processed
    ANNOUNCED = "ANNOUNCED"  # Task announced (e.g., CFP sent)
    BIDDING = "BIDDING"  # Bids are being collected (optional intermediate state)
    ALLOCATED = "ALLOCATED"  # Task assigned to an agent
    IN_PROGRESS = "IN_PROGRESS"  # Task execution started
    COMPLETED = "COMPLETED"  # Task successfully finished
    FAILED = "FAILED"  # Task execution failed
    CANCELLED = "CANCELLED"  # Task cancelled before completion


class TaskType(Enum):
    """
    Enum for the type of task, relevant for agent capabilities and bidding logic.
    """

    DELIVERY = "DELIVERY"
    RESTOCKING = "RESTOCKING"
    INVENTORY_CHECK = "INVENTORY_CHECK"
    CUSTOMER_ASSISTANCE = "CUSTOMER_ASSISTANCE"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    PICKUP = "PICKUP"
    # Add more specific retail task types as needed


@dataclass
class Task:
    """
    Data class representing a task to be allocated and executed.
    """

    type: TaskType
    description: str
    urgency: int  # Example: 1-10 scale, higher is more urgent
    required_capacity: int  # Abstract measure of effort/resources needed
    location: str | None = None  # Optional: relevant for physical tasks
    deadline: float | None = None  # Optional: e.g., timestamp or duration
    status: TaskStatus = TaskStatus.PENDING
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assigned_agent_id: str | None = None
    winning_bid: float | None = None  # Store the value of the winning bid
    creation_time: float = field(default_factory=time.time)
    data: dict[str, Any] | None = None  # For additional task-specific data

    def __post_init__(self):
        # Basic validation
        if not isinstance(self.type, TaskType):
            raise TypeError("Task type must be a TaskType Enum member.")
        if not isinstance(self.status, TaskStatus):
            raise TypeError("Task status must be a TaskStatus Enum member.")
        if not (1 <= self.urgency <= 10):
            # Consider logging a warning instead of raising error for flexibility
            print(
                f"Warning: Task urgency ({self.urgency}) outside typical range 1-10 for task {self.id}"
            )
            # raise ValueError("Urgency must be between 1 and 10.")


@dataclass
class Bid:
    """
    Data class representing a bid submitted by an agent for a task
    (specifically for Contract Net Protocol context).
    """

    agent_id: str
    task_id: str
    bid_value: float  # The calculated bid score/cost (lower is often better)
    # Optional fields providing more context for bid evaluation:
    estimated_completion_time: float | None = None
    agent_capacity_available: int | None = None
    confidence_score: float | None = (
        None  # Agent's confidence in completing the task
    )

    def __post_init__(self):
        if self.bid_value < 0:
            # Depending on bid semantics, negative might be invalid
            print(
                f"Warning: Bid value ({self.bid_value}) is negative for task {self.task_id} by agent {self.agent_id}"
            )
            # raise ValueError("Bid value cannot be negative.")
