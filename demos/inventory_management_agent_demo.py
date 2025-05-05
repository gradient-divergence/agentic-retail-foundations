"""
Demo script for a simple inventory management agent.

This script demonstrates an agent that checks inventory levels and reorders
products if the stock falls below a defined threshold. It uses placeholder
classes for the agent framework (Agent, Tool, Runner) for illustration.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Placeholder Agent Framework ---

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
        logger.info(f"Agent '{self.name}' initialized with tools: {list(self.tools.keys())}")

    def run(self, task: str):
        """Simulates the agent running a task (highly simplified)."""
        logger.info(f"Agent '{self.name}' starting task: {task}")
        # Simple logic based on task keywords
        if "product_123" in task and "50 units" in task:
            # Simulate checking inventory
            stock_tool = self.tools.get("check_inventory")
            order_tool = self.tools.get("order_product")

            if not stock_tool or not order_tool:
                return type("Result", (), {"final_output": "Error: Missing required tools."})()

            current_stock = stock_tool.func(product_id="product_123")
            logger.info(f"Checked inventory for product_123: {current_stock}")

            if current_stock < 50:
                amount_to_order = 50 - current_stock
                logger.info(f"Stock below threshold ({current_stock} < 50). Ordering {amount_to_order} units.")
                # Simulate ordering
                order_result = order_tool.func(product_id="product_123", amount=amount_to_order)
                return type("Result", (), {"final_output": order_result})()
            else:
                logger.info("Stock is sufficient.")
                return type("Result", (), {"final_output": "Stock level is sufficient. No action needed."})()
        else:
            logger.warning("Simplified agent logic cannot handle this task.")
            return type("Result", (), {"final_output": "Task not understood by simplified agent."})()

class Runner:
    """Placeholder for an agent runner."""
    @staticmethod
    def run_sync(agent: Agent, task: str):
        return agent.run(task)

# --- Inventory Demo Specific Logic ---

# Simulated inventory database
inventory_db = {"product_123": 20}

# Tool Functions
def check_inventory(product_id: str) -> int:
    """Return current inventory level for the given product."""
    stock = inventory_db.get(product_id, 0)
    logger.debug(f"Tool 'check_inventory' called for {product_id}. Returning {stock}")
    return stock

def order_product(product_id: str, amount: int) -> str:
    """Order more of the given product and update inventory."""
    logger.debug(f"Tool 'order_product' called for {product_id}, amount {amount}")
    current = inventory_db.get(product_id, 0)
    inventory_db[product_id] = current + amount
    result_msg = f"Ordered {amount} units of {product_id}, new stock is {inventory_db[product_id]}."
    logger.info(result_msg)
    return result_msg

# --- Demo Execution ---

def run_inventory_demo():
    """Sets up and runs the inventory agent demo."""
    logger.info("--- Starting Inventory Management Agent Demo ---")

    # Create tools
    check_inventory_tool = Tool(
        name="check_inventory",
        func=check_inventory,
        description="Check the current stock level of a product by ID.",
    )
    order_tool = Tool(
        name="order_product",
        func=order_product,
        description="Order more units of a product by ID.",
    )

    # Create agent
    inventory_agent = Agent(
        name="InventoryAgent",
        instructions=(
            "You are an autonomous inventory agent. "
            "If stock for a product is below the required level, use tools to reorder."
        ),
        tools=[check_inventory_tool, order_tool],
    )

    # Define the task
    task = "Ensure product_123 has at least 50 units in stock."
    logger.info(f"Task: {task}")

    # Print initial stock
    logger.info(f"Initial stock level (product_123): {inventory_db.get('product_123', 0)}")

    # Run the agent
    result = Runner.run_sync(inventory_agent, task)

    # Print the final result and updated stock
    logger.info(f"Agent Result: {result.final_output}")
    logger.info(f"Final stock level (product_123): {inventory_db.get('product_123', 0)}")
    logger.info("--- Inventory Management Agent Demo Finished ---")
    return result.final_output # Return result for potential display in Marimo

if __name__ == "__main__":
    run_inventory_demo() 