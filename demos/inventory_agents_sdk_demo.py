from __future__ import annotations

# region book:inventory-agents-sdk-demo
from demos.openai_agents_sdk_import import import_openai_agents_sdk

agents_sdk = import_openai_agents_sdk()
Agent = agents_sdk.Agent
Runner = agents_sdk.Runner
Tool = agents_sdk.Tool

inventory_db = {"product_123": 20}


def check_inventory(product_id: str) -> int:
    return inventory_db.get(product_id, 0)


def order_product(product_id: str, amount: int) -> str:
    current = inventory_db.get(product_id, 0)
    inventory_db[product_id] = current + amount
    return f"Ordered {amount} units of {product_id}, new stock is {inventory_db[product_id]}."


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

inventory_agent = Agent(
    name="InventoryAgent",
    instructions=(
        "You are an autonomous inventory agent. "
        "If stock for a product is below the required level, use tools to reorder."
    ),
    tools=[check_inventory_tool, order_tool],
)

task = "Ensure product_123 has at least 50 units in stock."
result = Runner.run_sync(inventory_agent, task)
print(result.final_output)

# endregion book:inventory-agents-sdk-demo
