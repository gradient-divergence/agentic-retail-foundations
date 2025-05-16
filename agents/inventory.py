"""
Inventory agent module for agentic-retail-foundations.
Defines the InventoryAgent class for inventory management using (s, S) policy.
"""

import logging

from agents.messaging import MessageBroker  # Assuming MessageBroker is defined here
from models.messaging import AgentMessage, Performative

# Set up logging
logger = logging.getLogger(__name__)


async def inventory_agent_handler(msg: AgentMessage, broker: MessageBroker, log_messages: list):
    """Handles messages directed to the inventory agent."""
    logger.info(f"Inventory agent received: {msg.performative.value} from {msg.sender}")
    log_messages.append(f"Inventory agent received: {msg.performative.value} from {msg.sender}")

    if msg.performative == Performative.QUERY:
        product_id = msg.content.get("product_id")
        # Simulate stock lookup
        stock_level = 15 if product_id == "P1001" else 5
        logger.debug(f"Looked up stock for {product_id}: {stock_level}")

        response = msg.create_reply(
            Performative.INFORM,
            {"product_id": product_id, "stock_level": stock_level},
        )
        await broker.deliver_message(response)
        log_messages.append(f"Inventory agent responded with stock level: {stock_level}")
        logger.info(f"Inventory agent responded to query for {product_id} with stock level: {stock_level}")

    elif msg.performative == Performative.SUBSCRIBE:
        # Assuming broker handles subscription logic
        topic = msg.content.get("topic", "inventory_alerts")  # Get topic from message content or default
        try:
            broker.subscribe(msg.sender, topic)
            log_messages.append(f"Registered {msg.sender} for {topic}")
            logger.info(f"Registered {msg.sender} for topic '{topic}'")
        except Exception as e:
            logger.error(f"Failed to subscribe {msg.sender} to {topic}: {e}")
            log_messages.append(f"Failed to register {msg.sender} for {topic}.")

    # TODO: Add handling for other performatives if needed


# Placeholder for a potential InventoryAgent class if needed later
# class InventoryAgent:
#     def __init__(self, agent_id: str, broker: MessageBroker):
#         self.agent_id = agent_id
#         self.broker = broker
#         self.broker.register_agent(self.agent_id, self.handle_message)

#     async def handle_message(self, msg: AgentMessage):
#         # Wrapper around the handler logic
#         log_messages = [] # Or manage logging differently
#         await inventory_agent_handler(msg, self.broker, log_messages)
#         # Process log_messages if necessary


class InventoryAgent:
    """
    Agent for inventory management using an (s, S) policy.
    - s = reorder_threshold: reorder when inventory falls below this level
    - S = max_capacity: order up to this level when reordering
    """

    def __init__(self, reorder_threshold, max_capacity):
        self.reorder_threshold = reorder_threshold  # When stock falls below this, agent should reorder
        self.max_capacity = max_capacity  # Max storage capacity or desired stock level
        self.current_stock = 0

    def perceive(self, external_data):
        """Sense the environment: get current stock level (and any other signals)."""
        self.current_stock = external_data.get("stock_level", self.current_stock)

    def decide(self):
        """
        Reason about whether and how much to reorder.
        Implements optimal (s,S) inventory policy where:
        - s = reorder_threshold: reorder when inventory falls below this level
        - S = max_capacity: order up to this level when reordering
        Optimality condition: s and S minimize total expected cost:
        C(s,S) = ordering costs + holding costs + stockout costs
        """
        if self.current_stock < self.reorder_threshold:
            # Plan action: calculate reorder quantity up to max capacity
            order_quantity = self.max_capacity - self.current_stock
            return {"action": "reorder", "amount": order_quantity}
        else:
            # No action needed
            return {"action": "wait"}

    def act(self, decision):
        """Execute the decided action (e.g., place an order)."""
        if decision["action"] == "reorder":
            amount = decision["amount"]
            print(f"Placing order for {amount} units.")  # In real system, call supplier API
            # For simulation, assume order immediately refills stock:
            self.current_stock += amount

    def learn(self, feedback):
        """Update agent's strategy based on outcomes (simplified as no-op here)."""
        # In a real agent, you might adjust thresholds or models based on feedback.
        pass
