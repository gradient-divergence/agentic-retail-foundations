"""
Demonstration of the Dynamic Pricing Agent with a real-time feedback loop.

NOTE: This script requires running Redis and Kafka brokers.
      It also requires sales events to be published to the Kafka topic
      (e.g., sales-events-SKU123456).
"""

import asyncio
import logging

# Import the refactored agent
from agents.dynamic_pricing_feedback import DynamicPricingAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dynamic-pricing-demo")


async def run_dynamic_pricing_demo():
    """Instantiate and run the dynamic pricing agent demo."""
    logger.info("--- Starting Dynamic Pricing Feedback Loop Demo ---")

    # --- Configuration ---
    # In a real app, load these from config files or environment variables
    product_id = "SKU123456"
    initial_price = 29.99
    min_price = 19.99
    max_price = 39.99
    redis_host = "localhost"
    redis_port = 6379
    kafka_brokers = "localhost:9092"

    # Instantiate the agent
    agent = DynamicPricingAgent(
        product_id=product_id,
        initial_price=initial_price,
        min_price=min_price,
        max_price=max_price,
        redis_host=redis_host,
        redis_port=redis_port,
        kafka_brokers=kafka_brokers,
    )

    # Start the agent's feedback loop
    # This will run indefinitely until stopped (e.g., Ctrl+C)
    try:
        await agent.run_feedback_loop()
    except Exception as e:
        logger.error(f"Dynamic pricing demo failed: {e}", exc_info=True)
    finally:
        logger.info("--- Dynamic Pricing Feedback Loop Demo Finished ---")


if __name__ == "__main__":
    try:
        asyncio.run(run_dynamic_pricing_demo())
    except KeyboardInterrupt:
        logger.info("Demo stopped by user.")
