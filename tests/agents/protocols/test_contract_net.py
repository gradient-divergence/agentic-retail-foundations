import asyncio

from agents.protocols.contract_net import RetailCoordinator
from models.task import Bid, Task, TaskStatus, TaskType


def test_lowest_bid_wins_and_first_bid_breaks_tie():
    coordinator = RetailCoordinator("coordinator", "Coordinator")
    task = Task(TaskType.DELIVERY, "Deliver", 5, 1)
    for agent in ["a", "b", "c"]:
        coordinator.register_participant(agent)
    asyncio.run(coordinator.announce_task(task))
    for agent, value in [("a", 5), ("b", 3), ("c", 3)]:
        coordinator.handle_bid(Bid(agent, task.id, value))
    winner = asyncio.run(coordinator.award_task(task.id))
    assert winner.agent_id == "b"
    assert task.status == TaskStatus.ALLOCATED
    assert task.assigned_agent_id == "b"
    assert task.winning_bid == 3


def test_unregistered_and_nonfinite_bids_do_not_win():
    coordinator = RetailCoordinator("coordinator", "Coordinator")
    task = Task(TaskType.DELIVERY, "Deliver", 5, 1)
    coordinator.register_participant("a")
    asyncio.run(coordinator.announce_task(task))
    coordinator.handle_bid(Bid("outsider", task.id, 0))
    coordinator.handle_bid(Bid("a", task.id, float("nan")))
    assert asyncio.run(coordinator.award_task(task.id)) is None
    assert task.status == TaskStatus.FAILED


def test_awarded_task_does_not_accept_late_bids_or_awards():
    coordinator = RetailCoordinator("coordinator", "Coordinator")
    task = Task(TaskType.DELIVERY, "Deliver", 5, 1)
    coordinator.register_participant("a")
    asyncio.run(coordinator.announce_task(task))
    assert task.status == TaskStatus.ANNOUNCED
    coordinator.handle_bid(Bid("a", task.id, 5))
    asyncio.run(coordinator.award_task(task.id))
    coordinator.handle_bid(Bid("a", task.id, 1))
    assert len(coordinator.bids[task.id]) == 1
    assert asyncio.run(coordinator.award_task(task.id)) is None
