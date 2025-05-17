"""
Task allocation and contract net protocol classes for distributed task management in retail MAS.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any

# Import StoreAgent from its new location
from agents.store import StoreAgent

# Import the data models from the models directory
from models.task import Bid, Task, TaskStatus, TaskType


class RetailCoordinator:
    """
    Coordinator for the Contract Net Protocol (CNP).
    Manages task announcement, bidding, allocation, and tracks execution status.
    Assumes Task, Bid, TaskStatus, TaskType models are imported from models.task
    Assumes StoreAgent class is defined (or imported)
    """

    def __init__(self):
        self.agents: dict[str, StoreAgent] = {}
        self.tasks: dict[str, Task] = {}

    def register_agent(self, agent: StoreAgent) -> None:
        """
        Register a store agent that can participate in task allocation.
        """
        if not isinstance(agent, StoreAgent):
            raise TypeError("Registered entity must be a StoreAgent instance.")
        if agent.agent_id in self.agents:
            print(f"Warning: Re-registering agent {agent.agent_id}")
        self.agents[agent.agent_id] = agent
        print(f"Agent {agent.name} ({agent.agent_id}) registered with coordinator.")

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
        print(f"Coordinator created Task {new_task.id}: {description[:50]}...")
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
            print(f"Error: Task {task_id} not found for allocation.")
            return None

        task = self.tasks[task_id]
        if task.status != TaskStatus.ANNOUNCED:
            print(f"Warning: Task {task_id} is not in ANNOUNCED state (current: {task.status.name}), cannot allocate.")
            return None

        print(f"\n--- Allocating Task {task_id} ({task.description[:30]}...) ---")
        print(f"Collecting bids from {len(self.agents)} agents...")

        bids: list[Bid] = []
        for agent_id, agent in self.agents.items():
            bid = agent.calculate_bid(task)
            if bid:
                bids.append(bid)
                print(f"  Agent {agent.name} bid: {bid.bid_value:.2f}")

        if not bids:
            print(f"--> No bids received for task {task_id}. Allocation failed.")
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

        print(f"--> Task {task_id} awarded to {winner_agent.name} (Bid: {best_bid.bid_value:.2f})")
        print("----------------------------------------------------")
        return winner_id

    async def execute_allocated_tasks(self):
        """
        Trigger the execution of all tasks currently in the ALLOCATED state.
        Uses asyncio.gather to run task executions concurrently.
        """
        tasks_to_execute = []
        agent_task_map = defaultdict(list)

        print("\n--- Triggering Execution of Allocated Tasks ---")
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.ALLOCATED and task.assigned_agent_id:
                agent_id = task.assigned_agent_id
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    tasks_to_execute.append(agent.execute_task(task))
                    agent_task_map[agent_id].append(task_id)
                else:
                    logging.error(f"Agent {agent_id} assigned to task {task_id} not found during execution phase.")
                    task.status = TaskStatus.FAILED

        if not tasks_to_execute:
            print("No tasks currently allocated for execution.")
            return

        print(f"Starting execution for {len(tasks_to_execute)} tasks across {len(agent_task_map)} agents...")
        results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
        print("--- Task Execution Cycle Complete ---")

        i = 0
        for agent_id, task_ids in agent_task_map.items():
            for task_id in task_ids:
                result = results[i]
                if isinstance(result, Exception):
                    logging.error(f"Error during execution of task {task_id} by agent {agent_id}: {result}")
                    if task_id in self.tasks:
                        self.tasks[task_id].status = TaskStatus.FAILED
                i += 1
