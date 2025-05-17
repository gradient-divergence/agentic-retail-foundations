"""
Demonstrates the Contract Net Protocol (CNP) for task allocation.
"""

import asyncio
import logging
import random

import pandas as pd

from agents.protocols.contract_net import RetailCoordinator
from agents.store import StoreAgent

# Import necessary components from the project structure
from models.task import Task, TaskStatus, TaskType

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def demo_contract_net_protocol():
    """Runs the Contract Net Protocol demonstration."""
    logger.info("Initializing Contract Net Protocol Demo...")

    # Create Coordinator - Provide coordinator_id and name
    coordinator = RetailCoordinator(coordinator_id="coordinator-1", name="Main Coordinator")
    logger.info("Coordinator created.")

    # Create Store Agents
    store_agents = [
        StoreAgent(agent_id="store-north", name="North Store", capacity=10, efficiency=0.9),
        StoreAgent(agent_id="store-south", name="South Store", capacity=5, efficiency=1.2),
        StoreAgent(agent_id="store-east", name="East Store", capacity=12, efficiency=1.0),
    ]
    logger.info(f"Created {len(store_agents)} store agents.")

    # Register agents with the coordinator
    for agent in store_agents:
        coordinator.register_participant(agent.agent_id)  # Use register_participant with agent ID
    logger.info("Registered store agents with the coordinator.")

    # Define Tasks
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
        Task(
            type=TaskType.RESTOCKING,
            description="Restock Shelf 5",
            urgency=9,
            required_capacity=8,
            location="East",
        ),
        Task(
            type=TaskType.DELIVERY,
            description="Urgent delivery to Zone C",
            urgency=10,
            required_capacity=5,
            location="South",
        ),
    ]
    logger.info(f"Created {len(tasks)} tasks.")

    # Announce tasks and run bidding process
    for task in tasks:
        logger.info(f"Announcing Task: {task.id} - {task.description}")
        participant_ids = await coordinator.announce_task(task)
        await asyncio.sleep(0.05)  # Short delay for announcement propagation

        # Simulate agents calculating and submitting bids
        logger.info(f"Simulating bid submission for Task: {task.id}")
        # Retrieve agent objects (in a real system, coordinator might not hold full objects)
        # For demo, we assume coordinator knows agent objects or can retrieve them
        # Here, we just use the list we created earlier.
        participant_agents = [agent for agent in store_agents if agent.agent_id in participant_ids]
        for agent in participant_agents:
            bid = agent.calculate_bid(task)  # Agent calculates bid
            if bid:
                coordinator.handle_bid(bid)  # Coordinator receives bid
                logger.debug(f"Agent {agent.agent_id} submitted bid {bid.bid_value:.2f} for task {task.id}")
        await asyncio.sleep(0.05)  # Allow time for bids to be processed

        logger.info(f"Evaluating bids for Task: {task.id}")
        winning_bid = await coordinator.award_task(task.id)  # Use award_task
        winning_agent_id = winning_bid.agent_id if winning_bid else None  # Get agent_id from bid

        if winning_agent_id:
            logger.info(f"Task {task.id} awarded to Agent {winning_agent_id}")
        else:
            logger.info(f"No suitable bids received for Task {task.id}")
        # await asyncio.sleep(0.1) # Allow time for processing awards - Reduced delay needed here

    # Simulate Task Execution (Added Step)
    logger.info("\nSimulating task execution...")
    for task_id, task in coordinator.tasks.items():
        if task.status == TaskStatus.ALLOCATED:
            # Simulate completion or failure randomly
            final_status = random.choice([TaskStatus.COMPLETED, TaskStatus.COMPLETED, TaskStatus.FAILED])
            coordinator.update_task_status(task_id, final_status)
            logger.info(f"Task {task_id} finished with status: {final_status.name}")
    logger.info("Task execution simulation complete.")

    # Log results (task history)
    try:
        history_df = pd.DataFrame(coordinator.task_history)
        logger.info("\n--- Contract Net Protocol Event Log ---")
        logger.info(history_df.to_string())
        logger.info("--------------------------------------\n")
    except Exception as e:
        logger.error(f"Could not display task history as DataFrame: {e}")
        logger.info("\n--- Raw Task History ---")
        logger.info(str(coordinator.task_history))
        logger.info("------------------------\n")

    # Add Final Task Status Summary (Restored Logic)
    final_statuses = []
    logger.info("Preparing final task status summary...")
    for tid, task in coordinator.tasks.items():
        # Need to get agent name from our list, as coordinator might only store ID
        agent_name = "-"
        if task.assigned_agent_id:
            assigned_agent = next((a for a in store_agents if a.agent_id == task.assigned_agent_id), None)
            if assigned_agent:
                agent_name = assigned_agent.name

        # Format winning bid value
        # winning_bid_value = getattr(task, "winning_bid_value", None) # F841: Removed as unused
        # Let's find the winning bid from history for display
        winning_bid_display = "-"
        if task.assigned_agent_id:
            for event in reversed(coordinator.task_history):
                if event.get("task_id") == tid and event.get("action") == "awarded" and event.get("participant_id") == task.assigned_agent_id:
                    winning_bid_display = f"{event.get('bid_value'):.2f}"
                    break
            if winning_bid_display == "-":  # Fallback if award event missing value
                winning_bid_obj = getattr(task, "winning_bid", None)
                if winning_bid_obj and hasattr(winning_bid_obj, "bid_value"):
                    winning_bid_display = f"{winning_bid_obj.bid_value:.2f}"

        final_statuses.append(
            {
                "Task ID": tid,
                "Description": task.description,
                "Final Status": task.status.name,
                "Assigned Agent": agent_name,
                "Winning Bid": winning_bid_display,
            }
        )
    if final_statuses:
        logger.info("\n--- Final Task Status Summary ---")
        summary_df = pd.DataFrame(final_statuses)
        logger.info(summary_df.to_string(index=False))
        logger.info("-------------------------------")
    else:
        logger.info("No final task statuses to display.")

    logger.info("Contract Net Protocol Demo completed.")


if __name__ == "__main__":
    asyncio.run(demo_contract_net_protocol())
