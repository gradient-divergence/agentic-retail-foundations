"""
Task allocation and contract net protocol classes for distributed task management in retail MAS.
"""

import asyncio
import logging
from typing import Any

# Import StoreAgent from its new location
from agents.store import StoreAgent

# Import the data models from the models directory
from models.task import Bid, Task, TaskStatus, TaskType

logger = logging.getLogger(__name__)


class RetailCoordinator:
    """
    Coordinator for the Contract Net Protocol (CNP).
    Manages task announcement, bidding, allocation, and tracks execution status.
    Assumes Task, Bid, TaskStatus, TaskType models are imported from models.task
    Assumes StoreAgent class is defined (or imported)
    """

    def __init__(self) -> None:
        self.agents: dict[str, StoreAgent] = {}
        self.tasks: dict[str, Task] = {}

    def register_agent(self, agent: StoreAgent) -> None:
        """
        Register a store agent that can participate in task allocation.
        """
        if not isinstance(agent, StoreAgent):
            raise TypeError("Registered entity must be a StoreAgent instance.")
        if agent.agent_id in self.agents:
            logger.warning("Re-registering agent %s", agent.agent_id)
        self.agents[agent.agent_id] = agent
        logger.info("Agent %s (%s) registered with coordinator.", agent.name, agent.agent_id)

    def create_task(
        self,
        task_type: TaskType,
        description: str,
        urgency: int,
        required_capacity: int,
        location: str | None = None,
        deadline: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new task and add it to the coordinator's task list.
        Returns the unique task ID.
        """
        new_task = Task(
            type=task_type,
            description=description,
            urgency=urgency,
            required_capacity=required_capacity,
            location=location,
            deadline=deadline,
            status=TaskStatus.ANNOUNCED,
            data=data,
        )
        self.tasks[new_task.id] = new_task
        logger.info(
            "Coordinator created Task %s: %s...",
            new_task.id,
            description[:50],
        )
        return new_task.id

    async def allocate_task(self, task_id: str) -> str | None:
        """
        Perform the CNP allocation for a specific task:
        1. Announce Task (Implicit - task is already created and ANNOUNCED)
        2. Collect Bids from registered agents.
        3. Select Winner based on lowest bid value.
        4. Award Task (update task status and agent assignment).
        Returns the winning agent's ID or None if no agent could be allocated.
        """
        if task_id not in self.tasks:
            logger.error("Task %s not found for allocation.", task_id)
            return None

        task = self.tasks[task_id]
        if task.status != TaskStatus.ANNOUNCED:
            logger.warning(
                "Task %s is not in ANNOUNCED state (current: %s), cannot allocate.",
                task_id,
                task.status.name,
            )
            return None

        logger.info(
            "Allocating Task %s (%s...)",
            task_id,
            task.description[:30],
        )
        logger.info("Collecting bids from %s agents...", len(self.agents))

        bids: list[Bid] = []
        for _, agent in self.agents.items():
            bid = agent.calculate_bid(task)
            if bid:
                bids.append(bid)
                logger.info("Agent %s bid: %.2f", agent.name, bid.bid_value)

        if not bids:
            logger.warning("No bids received for task %s. Allocation failed.", task_id)
            task.status = TaskStatus.FAILED
            return None

        bids.sort(key=lambda b: b.bid_value)
        best_bid = bids[0]
        winner_id = best_bid.agent_id
        winner_agent = self.agents[winner_id]

        task.status = TaskStatus.ALLOCATED
        task.assigned_agent_id = winner_id
        task.winning_bid = best_bid.bid_value
        winner_agent.assigned_tasks.append(task)

        logger.info(
            "Task %s awarded to %s (Bid: %.2f)",
            task_id,
            winner_agent.name,
            best_bid.bid_value,
        )
        return winner_id

    async def execute_allocated_tasks(self) -> None:
        """
        Trigger the execution of all tasks currently in the ALLOCATED state.
        Uses asyncio.gather to run task executions concurrently.
        """
        tasks_to_execute = []
        executions: list[tuple[str, str]] = []

        logger.info("Triggering execution of allocated tasks.")
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.ALLOCATED and task.assigned_agent_id:
                agent_id = task.assigned_agent_id
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    tasks_to_execute.append(agent.execute_task(task))
                    executions.append((agent_id, task_id))
                else:
                    logger.error(
                        "Agent %s assigned to task %s not found during execution phase.",
                        agent_id,
                        task_id,
                    )
                    task.status = TaskStatus.FAILED

        if not tasks_to_execute:
            logger.info("No tasks currently allocated for execution.")
            return

        logger.info(
            "Starting execution for %s tasks across %s agents...",
            len(tasks_to_execute),
            len({agent_id for agent_id, _ in executions}),
        )
        results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
        logger.info("Task execution cycle complete.")

        for (agent_id, task_id), result in zip(executions, results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    "Error during execution of task %s by agent %s: %s",
                    task_id,
                    agent_id,
                    result,
                )
                if task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.FAILED
