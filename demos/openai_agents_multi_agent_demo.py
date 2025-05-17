"""Demonstrates a simple multi-agent conversation using OpenAI's Agents SDK.

Two assistants are created: an inventory agent and a pricing agent. Both operate
on the same thread, allowing them to share context. The user asks about a
product's stock level and requests pricing suggestions. The inventory agent
responds first, then the pricing agent runs using the inventory information.
"""

import logging
from openai import OpenAI, OpenAIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_multi_agent_demo() -> None:
    """Run the OpenAI Agents SDK multi-agent demonstration."""
    client = OpenAI()
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    try:
        logger.info("Creating assistants...")
        inventory = client.beta.assistants.create(
            name="Inventory Assistant",
            instructions="Provide stock availability for products when asked.",
            model="gpt-4o",
        )
        pricing = client.beta.assistants.create(
            name="Pricing Assistant",
            instructions="Suggest price adjustments based on inventory levels.",
            model="gpt-4o",
        )

        thread = client.beta.threads.create()
        user_query = "Check inventory for SKU123 and suggest pricing."
        client.beta.threads.messages.create(thread_id=thread.id, role="user", content=user_query)

        logger.info("Running inventory assistant...")
        client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=inventory.id)

        logger.info("Running pricing assistant...")
        client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=pricing.id)

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        for msg in reversed(messages.data):
            content = msg.content[0].text.value if msg.content else ""
            print(f"{msg.role.capitalize()}: {content}")
    except OpenAIError as e:
        logger.error("OpenAI API Error: %s", e)
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected error: %s", e)


if __name__ == "__main__":
    run_multi_agent_demo()
