"""
Orchestrator and worker pattern sketch for ADK-style agent coordination.
Adapt the structure to your ADK runtime and tool registry.
"""

from __future__ import annotations

from pydantic import BaseModel


class InventoryCheckPayload(BaseModel):
    sku: str


class PriceUpdatePayload(BaseModel):
    sku: str


TaskPayload = InventoryCheckPayload | PriceUpdatePayload


class Task(BaseModel):
    task_id: str
    intent: str
    payload: TaskPayload


class TaskResult(BaseModel):
    task_id: str
    status: str
    result: TaskPayload


def worker(task: Task) -> TaskResult:
    return TaskResult(task_id=task.task_id, status="done", result=task.payload)


def orchestrator(tasks: list[Task]) -> list[TaskResult]:
    return [worker(task) for task in tasks]


def run_demo() -> None:
    tasks = [
        Task(task_id="t1", intent="check_inventory", payload={"sku": "SKU123"}),
        Task(task_id="t2", intent="update_price", payload={"sku": "SKU456"}),
    ]
    results = orchestrator(tasks)
    print(results)


if __name__ == "__main__":
    run_demo()
