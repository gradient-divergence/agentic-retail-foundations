"""
Defines the Store Agent class representing a store participating in protocols like CNP.
"""

from dataclasses import dataclass, field
import asyncio
import random

# Import the data models from the models directory
from models.task import TaskStatus, Task, Bid


@dataclass
class StoreAgent:
    """
    Agent representing a store, capable of bidding for and executing tasks.
    Assumes Task and Bid models are imported from models.task
    """

    agent_id: str
    name: str
    capacity: int
    efficiency: float
    location: str = field(init=False)  # Set in post_init
    assigned_tasks: list[Task] = field(default_factory=list)

    def __post_init__(self):
        # Use name as location if not specified otherwise
        self.location = self.name

    def calculate_bid(self, task: Task) -> Bid | None:
        """
        Calculate a bid for a given task, considering capacity and efficiency.
        Returns None if the agent cannot perform the task.
        The bid_value represents the agent's cost/desirability (lower is better).
        """
        used_capacity = sum(
            t.required_capacity
            for t in self.assigned_tasks
            if t.status in [TaskStatus.ALLOCATED, TaskStatus.IN_PROGRESS]
        )
        if used_capacity + task.required_capacity > self.capacity:
            return None

        efficiency_cost = self.efficiency

        urgency_factor = 1 + (task.urgency / 20.0) * (
            len(self.assigned_tasks) / max(1, self.capacity)
        )

        location_penalty = 0.0
        if task.location and task.location != self.location:
            location_penalty = 5.0

        base_cost = task.required_capacity * 2.0

        bid_amount = (base_cost * efficiency_cost * urgency_factor) + location_penalty

        current_workload_duration = sum(
            t.required_capacity * self.efficiency for t in self.assigned_tasks
        )
        estimated_task_duration = task.required_capacity * self.efficiency
        completion_time = current_workload_duration + estimated_task_duration

        available_capacity = self.capacity - used_capacity

        return Bid(
            agent_id=self.agent_id,
            task_id=task.id,
            bid_value=bid_amount,
            estimated_completion_time=completion_time,
            agent_capacity_available=available_capacity,
        )

    async def execute_task(self, task: Task) -> bool:
        """
        Execute a task asynchronously, simulating execution time and success/failure.
        Updates task status internally.
        """
        print(f"Agent {self.name} executing task {task.id}: {task.description}")
        task.status = TaskStatus.IN_PROGRESS
        execution_time = max(0.1, task.required_capacity * self.efficiency * 0.1)
        await asyncio.sleep(execution_time)

        success_probability = max(0.1, min(0.98, 1.0 / (self.efficiency + 0.1)))
        success = random.random() < success_probability

        if success:
            print(f"Agent {self.name} completed task {task.id}")
            task.status = TaskStatus.COMPLETED
        else:
            print(f"Agent {self.name} failed task {task.id}")
            task.status = TaskStatus.FAILED

        # Remove task from assigned list upon completion/failure
        self.assigned_tasks = [t for t in self.assigned_tasks if t.id != task.id]
        return success
