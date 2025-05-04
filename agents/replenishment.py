import logging
from models.messaging import AgentMessage, Performative

# Set up logging
logger = logging.getLogger(__name__)


async def replenishment_agent_handler(msg: AgentMessage, log_messages: list):
    """Handles messages directed to the replenishment agent."""
    logger.info(
        f"Replenishment agent received: {msg.performative.value} from {msg.sender}"
    )
    log_messages.append(
        f"Replenishment agent received: {msg.performative.value} from {msg.sender}"
    )

    if msg.performative == Performative.INFORM and "stock_level" in msg.content:
        product_id = msg.content.get("product_id", "[unknown product]")
        stock_level = msg.content["stock_level"]

        if stock_level < 10:  # Example threshold
            log_messages.append(
                f"Low stock alert for {product_id}: {stock_level} units"
            )
            log_messages.append("Replenishment agent will create a restock order")
            logger.warning(
                f"Low stock detected for {product_id} ({stock_level} units). Initiating restock process."
            )
            # TODO: Add logic to actually create a restock order
        else:
            log_messages.append(
                f"Adequate stock level for {product_id}: {stock_level} units"
            )
            logger.info(
                f"Adequate stock level for {product_id} ({stock_level} units). No action needed."
            )

    # TODO: Add handling for other performatives if needed


# Placeholder for a potential ReplenishmentAgent class if needed later
# from agents.messaging import MessageBroker # Import needed if class is used
# class ReplenishmentAgent:
#     def __init__(self, agent_id: str, broker: MessageBroker):
#         self.agent_id = agent_id
#         self.broker = broker
#         # Might subscribe to topics or be directly messaged
#         # self.broker.register_agent(self.agent_id, self.handle_message)

#     async def handle_message(self, msg: AgentMessage):
#         # Wrapper around the handler logic
#         log_messages = [] # Or manage logging differently
#         await replenishment_agent_handler(msg, log_messages)
#         # Process log_messages if necessary
