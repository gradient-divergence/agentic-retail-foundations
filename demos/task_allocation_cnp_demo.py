"""
Demonstrates task allocation using a simplified Contract Net Protocol (CNP).

This demo keeps the focus on the mechanism:
- announce tasks,
- collect bids,
- award each task to the best bid,
- optionally simulate execution.
"""

# region book:task-allocation-demo

import asyncio

from agents.protocols.contract_net import RetailCoordinator
from agents.store import StoreAgent
from models.task import Task, TaskType


async def demo_contract_net_protocol() -> None:
    coordinator = RetailCoordinator(coordinator_id="coordinator-1", name="Main Coordinator")

    store_agents = [
        StoreAgent(agent_id="store-north", name="North Store", capacity=10, efficiency=0.9),
        StoreAgent(agent_id="store-south", name="South Store", capacity=5, efficiency=1.2),
        StoreAgent(agent_id="store-east", name="East Store", capacity=12, efficiency=1.0),
    ]
    for agent in store_agents:
        coordinator.register_participant(agent.agent_id)

    tasks = [
        Task(
            type=TaskType.DELIVERY,
            description="Deliver package to Zone A",
            urgency=7,
            required_capacity=3,
            location="North",
        ),
        Task(
            type=TaskType.INVENTORY_CHECK,
            description="Check stock for Item X",
            urgency=5,
            required_capacity=1,
            location="South",
        ),
    ]

    for task in tasks:
        participant_ids = await coordinator.announce_task(task)

        for agent in store_agents:
            if agent.agent_id not in participant_ids:
                continue
            bid = agent.calculate_bid(task)
            if bid is not None:
                coordinator.handle_bid(bid)

        winning_bid = await coordinator.award_task(task.id)
        if winning_bid is None:
            print(f"Task {task.id[:8]}: no suitable bids")
            continue

        winner = next(a for a in store_agents if a.agent_id == winning_bid.agent_id)
        winner.assigned_tasks.append(task)
        print(f"Task {task.id[:8]} awarded to {winner.name} (bid={winning_bid.bid_value:.2f})")

        await winner.execute_task(task)


if __name__ == "__main__":
    asyncio.run(demo_contract_net_protocol())

# endregion book:task-allocation-demo
