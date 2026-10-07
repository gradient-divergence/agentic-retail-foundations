import asyncio
from unittest.mock import AsyncMock

from agents.store import StoreAgent
from agents.task_allocation import RetailCoordinator
from models.task import Task, TaskStatus, TaskType


def test_execution_errors_remain_attached_to_interleaved_tasks():
    coordinator = RetailCoordinator()
    first = StoreAgent("a", "A", 10, 1)
    second = StoreAgent("b", "B", 10, 1)
    first.execute_task = AsyncMock(return_value=True)
    second.execute_task = AsyncMock(side_effect=RuntimeError("B failed"))
    coordinator.register_agent(first)
    coordinator.register_agent(second)
    tasks = [
        Task(TaskType.DELIVERY, "Deliver", 5, 1, status=TaskStatus.ALLOCATED, assigned_agent_id=agent)
        for agent in ["a", "b", "a"]
    ]
    coordinator.tasks = {task.id: task for task in tasks}
    asyncio.run(coordinator.execute_allocated_tasks())
    assert [task.status for task in tasks] == [TaskStatus.ALLOCATED, TaskStatus.FAILED, TaskStatus.ALLOCATED]
