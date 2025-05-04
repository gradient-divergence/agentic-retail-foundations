"""
Demo script for a simple dynamic pricing agent.

This script demonstrates an agent that adjusts product prices based on
competitor pricing and inventory levels. It uses placeholder classes for
the agent framework (Agent, Tool, Runner) for illustration.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Placeholder Agent Framework (assuming it's shared or redefined) ---
# Note: In a real project, these would likely be imported from a central module.


class Tool:
    """Placeholder for an agent tool."""

    def __init__(self, name: str, func, description: str):
        self.name = name
        self.func = func
        self.description = description


class Agent:
    """Placeholder for an agent."""

    def __init__(self, name: str, instructions: str, tools: list[Tool]):
        self.name = name
        self.instructions = instructions
        self.tools = {tool.name: tool for tool in tools}
        logger.info(
            f"Agent '{self.name}' initialized with tools: {list(self.tools.keys())}"
        )

    def run(self, task: str):
        """Simulates the agent running a task (highly simplified)."""
        logger.info(f"Agent '{self.name}' starting task: {task}")
        # Simple logic based on task keywords
        if "product_456" in task and "adjust the price" in task:
            # Simulate checking competitor price and inventory
            price_tool = self.tools.get("get_competitor_price")
            stock_tool = self.tools.get("get_inventory")
            update_tool = self.tools.get("update_price")

            if not price_tool or not stock_tool or not update_tool:
                return type(
                    "Result", (), {"final_output": "Error: Missing required tools."}
                )()

            competitor_price = price_tool.func(product_id="product_456")
            current_stock = stock_tool.func(product_id="product_456")
            logger.info(
                f"Checked product_456: Competitor Price=${competitor_price}, Current Stock={current_stock}"
            )

            # Simple pricing logic simulation
            current_price = current_prices.get("product_456", 0)  # Get current price
            new_price = current_price  # Default to no change

            if (
                competitor_price
                and current_stock <= 5
                and competitor_price > current_price
            ):
                # Low stock, competitor is higher -> Increase price (e.g., halfway to competitor)
                new_price = current_price + (competitor_price - current_price) * 0.5
                logger.info(
                    f"Low stock and competitor price higher. Suggesting price increase to ${new_price:.2f}"
                )
            elif competitor_price and competitor_price < current_price:
                # Competitor is lower -> Decrease price (e.g., match competitor)
                new_price = competitor_price
                logger.info(
                    f"Competitor price lower. Suggesting price decrease to ${new_price:.2f}"
                )
            # Add more rules as needed (e.g., high stock -> decrease)

            if new_price != current_price:
                update_result = update_tool.func(
                    product_id="product_456", new_price=new_price
                )
                return type("Result", (), {"final_output": update_result})()
            else:
                logger.info("No price adjustment needed based on current rules.")
                return type(
                    "Result", (), {"final_output": "No price change recommended."}
                )()

        else:
            logger.warning("Simplified agent logic cannot handle this task.")
            return type(
                "Result",
                (),
                {"final_output": "Task not understood by simplified agent."},
            )()


class Runner:
    """Placeholder for an agent runner."""

    @staticmethod
    def run_sync(agent: Agent, task: str):
        return agent.run(task)


# --- Pricing Demo Specific Logic ---

# Simulated data sources
competitor_prices = {"product_456": 120.00}
current_prices = {"product_456": 100.00}
inventory_levels = {"product_456": 5}


# Tool Functions
def get_competitor_price(product_id: str) -> float | None:
    """Fetch the latest competitor price for a product."""
    price = competitor_prices.get(product_id, None)
    logger.debug(
        f"Tool 'get_competitor_price' called for {product_id}. Returning {price}"
    )
    return price


def get_inventory(product_id: str) -> int:
    """Get current inventory level for a product."""
    stock = inventory_levels.get(product_id, 0)
    logger.debug(f"Tool 'get_inventory' called for {product_id}. Returning {stock}")
    return stock


def update_price(product_id: str, new_price: float) -> str:
    """Update the product's price to the new value."""
    logger.debug(f"Tool 'update_price' called for {product_id}, new_price {new_price}")
    current_prices[product_id] = new_price
    result_msg = f"Price for {product_id} updated to ${new_price:.2f}"
    logger.info(result_msg)
    return result_msg


# --- Demo Execution ---


def run_pricing_demo():
    """Sets up and runs the dynamic pricing agent demo."""
    logger.info("--- Starting Dynamic Pricing Agent Demo ---")

    # Create tools
    price_tool = Tool(
        name="get_competitor_price",
        func=get_competitor_price,
        description="Get competitor's price for a product.",
    )
    stock_tool = Tool(
        name="get_inventory",
        func=get_inventory,
        description="Get current stock level for a product.",
    )
    update_tool = Tool(
        name="update_price",
        func=update_price,
        description="Set a new price for a product.",
    )

    # Create agent
    pricing_agent = Agent(
        name="PricingAgent",
        instructions=(
            "You are a pricing agent that optimizes product prices for profit while avoiding stockouts. "
            "Use tools to check competitor pricing and inventory. If our price is too low and stock is limited, consider raising it. "
            "If stock is high or competitor price is lower, consider lowering our price to boost sales."
        ),
        tools=[price_tool, stock_tool, update_tool],
    )

    # Define the task
    task = "Evaluate and adjust the price for product_456."
    logger.info(f"Task: {task}")

    # Print initial state
    logger.info(
        f"Initial price (product_456): ${current_prices.get('product_456', 0):.2f}"
    )
    logger.info(
        f"Competitor price (product_456): ${competitor_prices.get('product_456', 0):.2f}"
    )
    logger.info(
        f"Current inventory (product_456): {inventory_levels.get('product_456', 0)} units"
    )

    # Run the agent
    result = Runner.run_sync(pricing_agent, task)

    # Print the final result and updated price
    logger.info(f"Agent Result: {result.final_output}")
    logger.info(
        f"Final price (product_456): ${current_prices.get('product_456', 0):.2f}"
    )
    logger.info("--- Dynamic Pricing Agent Demo Finished ---")
    return result.final_output  # Return result for potential display in Marimo


if __name__ == "__main__":
    run_pricing_demo()
