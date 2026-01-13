"""
Demonstrates agent-to-agent communication with a simple broker.

This demo showcases:
- direct messaging (agent -> agent),
- publish/subscribe messaging (agent -> topic -> subscribers),
- reply correlation via conversation IDs.
"""

# region book:agent-communication-setup

import asyncio

from agents.messaging import MessageBroker
from models.messaging import AgentMessage, Performative


async def demo_retail_agent_communication() -> None:
    broker = MessageBroker()

    async def inventory_on_message(msg: AgentMessage) -> None:
        if msg.performative == Performative.QUERY:
            product_id = str(msg.content.get("product_id"))
            stock_level = 15 if product_id == "P1001" else 5
            print(f"[inventory] {product_id} stock={stock_level}")

            reply = msg.create_reply(
                Performative.INFORM,
                {"product_id": product_id, "stock_level": stock_level},
            )
            await broker.deliver_message(reply)
            return

        if msg.performative == Performative.SUBSCRIBE:
            topic = str(msg.content.get("topic", "inventory_alerts"))
            broker.subscribe(msg.sender, topic)
            print(f"[inventory] subscribed {msg.sender} to topic '{topic}'")
            return

    async def replenishment_on_message(msg: AgentMessage) -> None:
        if msg.performative != Performative.INFORM:
            return

        product_id = str(msg.content.get("product_id"))
        stock_level = int(msg.content.get("stock_level", 0))
        if stock_level < 10:
            print(f"[replenishment] low stock: {product_id}={stock_level}")
        else:
            print(f"[replenishment] ok stock: {product_id}={stock_level}")

    broker.register_agent("inventory", inventory_on_message)
    broker.register_agent("replenishment", replenishment_on_message)

    # endregion book:agent-communication-setup

    # region book:agent-communication-run
    await broker.deliver_message(
        AgentMessage(
            performative=Performative.QUERY,
            sender="replenishment",
            receiver="inventory",
            content={"product_id": "P1001"},
        )
    )
    await broker.deliver_message(
        AgentMessage(
            performative=Performative.SUBSCRIBE,
            sender="replenishment",
            receiver="inventory",
            content={"topic": "inventory_alerts"},
        )
    )
    await broker.deliver_message(
        AgentMessage(
            performative=Performative.INFORM,
            sender="inventory_system",
            receiver="topic:inventory_alerts",
            content={
                "product_id": "P1002",
                "stock_level": 3,
                "alert_type": "low_stock",
            },
        )
    )


if __name__ == "__main__":
    asyncio.run(demo_retail_agent_communication())

# endregion book:agent-communication-run
