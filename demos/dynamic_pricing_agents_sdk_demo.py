from __future__ import annotations

# region book:dynamic-pricing-agents-sdk-demo
from demos.openai_agents_sdk_import import import_openai_agents_sdk

agents_sdk = import_openai_agents_sdk()
Agent = agents_sdk.Agent
Runner = agents_sdk.Runner
Tool = agents_sdk.Tool

competitor_prices = {"product_456": 120.00}
current_prices = {"product_456": 100.00}
inventory_levels = {"product_456": 5}


def get_competitor_price(product_id: str) -> float | None:
    return competitor_prices.get(product_id)


def get_inventory(product_id: str) -> int:
    return inventory_levels.get(product_id, 0)


def update_price(product_id: str, new_price: float) -> str:
    current_prices[product_id] = new_price
    return f"Price for {product_id} updated to ${new_price:.2f}"


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

pricing_agent = Agent(
    name="PricingAgent",
    instructions=(
        "You are a pricing agent that optimizes product prices for profit while "
        "avoiding stockouts. Use tools to check competitor pricing and inventory. "
        "If our price is too low and stock is limited, consider raising it. "
        "If stock is high or competitor price is lower, consider lowering our price "
        "to boost sales."
    ),
    tools=[price_tool, stock_tool, update_tool],
)

task = "Evaluate and adjust the price for product_456."
result = Runner.run_sync(pricing_agent, task)
print(result.final_output)

# endregion book:dynamic-pricing-agents-sdk-demo
