"""
Demonstrates the basic agent communication framework using MessageBroker.
"""

import asyncio
import logging

from agents.messaging import MessageBroker

# Import necessary components from the project structure
from models.messaging import AgentMessage, Performative

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def demo_retail_agent_communication():
    """Runs the agent communication demonstration."""
    log_messages = []
    log_messages.append("Initializing Agent Communication Demo...")
    logger.info("Initializing Agent Communication Demo...")

    broker = MessageBroker()

    async def inventory_agent_handler(msg: AgentMessage):
        log_msg = f"Inventory agent received: {msg.performative.value} from {msg.sender}"
        log_messages.append(log_msg)
        logger.info(log_msg)

        if msg.performative == Performative.QUERY:
            product_id = msg.content.get("product_id")
            stock_level = 15 if product_id == "P1001" else 5  # Simulate stock
            response = msg.create_reply(
                Performative.INFORM,
                {"product_id": product_id, "stock_level": stock_level},
            )
            await broker.deliver_message(response)
            log_msg = f"Inventory agent responded with stock level: {stock_level}"
            log_messages.append(log_msg)
            logger.info(log_msg)

        elif msg.performative == Performative.SUBSCRIBE:
            topic = msg.content.get("topic", "default_topic")  # Use topic from content
            broker.subscribe(msg.sender, topic)
            log_msg = f"Registered {msg.sender} for topic '{topic}'"
            log_messages.append(log_msg)
            logger.info(log_msg)

    async def replenishment_agent_handler(msg: AgentMessage):
        log_msg = f"Replenishment agent received: {msg.performative.value} from {msg.sender}"
        log_messages.append(log_msg)
        logger.info(log_msg)

        if msg.performative == Performative.INFORM and "stock_level" in msg.content:
            stock_level = msg.content["stock_level"]
            product_id = msg.content.get("product_id", "Unknown")
            if stock_level < 10:
                log_msg = f"Low stock alert for {product_id}: {stock_level} units. Replenishment agent will create a restock order."
            else:
                log_msg = f"Adequate stock level for {product_id}: {stock_level} units."
            log_messages.append(log_msg)
            logger.info(log_msg)

    broker.register_agent("inventory", inventory_agent_handler)
    broker.register_agent("replenishment", replenishment_agent_handler)

    logger.info("Step 1: Replenishment agent queries inventory agent about P1001")
    query_msg = AgentMessage(
        performative=Performative.QUERY,
        sender="replenishment",
        receiver="inventory",
        content={"product_id": "P1001"},
    )
    await broker.deliver_message(query_msg)
    await asyncio.sleep(0.1)

    logger.info("Step 2: Replenishment agent subscribes to inventory alerts")
    subscribe_msg = AgentMessage(
        performative=Performative.SUBSCRIBE,
        sender="replenishment",
        receiver="inventory",
        content={"topic": "inventory_alerts"},  # Specify topic
    )
    await broker.deliver_message(subscribe_msg)
    await asyncio.sleep(0.1)

    logger.info("Step 3: Inventory system sends an alert about low stock of P1002")
    alert_msg = AgentMessage(
        performative=Performative.INFORM,
        sender="inventory_system",  # Simulate system message
        receiver="topic:inventory_alerts",  # Publish to topic
        content={
            "product_id": "P1002",
            "stock_level": 3,
            "alert_type": "low_stock",
        },
    )
    await broker.deliver_message(alert_msg)
    await asyncio.sleep(0.1)  # Ensure alert is processed

    logger.info("Agent Communication Demo completed.")
    print("\n--- Communication Log ---")
    print("\n".join(log_messages))
    print("-------------------------\n")


if __name__ == "__main__":
    asyncio.run(demo_retail_agent_communication())
