from __future__ import annotations

# region book:dynamic-pricing-agents-sdk-demo
from math import isfinite

from demos.openai_agents_sdk_import import import_openai_agents_sdk

competitor_prices = {"product_456": 120.00}
current_prices = {"product_456": 100.00}
inventory_levels = {"product_456": 5}


def get_competitor_price(product_id: str) -> float | None:
    return competitor_prices.get(product_id)


def get_inventory(product_id: str) -> int:
    return inventory_levels.get(product_id, 0)


def update_price(product_id: str, new_price: float) -> str:
    if not isfinite(new_price) or new_price < 0:
        raise ValueError("Price must be finite and non-negative")
    current_prices[product_id] = new_price
    return f"Price for {product_id} updated to ${new_price:.2f}"


def run_demo() -> None:
    agents_sdk = import_openai_agents_sdk()
    Agent = agents_sdk.Agent
    Runner = agents_sdk.Runner
    function_tool = agents_sdk.function_tool

    price_tool = function_tool(
        get_competitor_price,
        description_override="Get competitor's price for a product.",
    )
    stock_tool = function_tool(
        get_inventory,
        description_override="Get current stock level for a product.",
    )
    update_tool = function_tool(
        update_price,
        description_override="Set a new price for a product.",
    )

    pricing_agent = Agent(
        name="PricingAgent",
        instructions=(
            "Demonstrate tool-calling repricing on this synthetic fixture. "
            "Inspect the recorded competitor price and stock, then apply the requested candidate price. "
            "These tools provide no demand or contribution model and cannot establish profit optimality."
        ),
        tools=[price_tool, stock_tool, update_tool],
    )

    task = "Inspect product_456 and set the demonstration candidate price to $110. Its initial price is $100."
    result = Runner.run_sync(pricing_agent, task)
    if current_prices.get("product_456") != 110.0:
        raise RuntimeError("Price postcondition failed: product_456 is not at the demonstration candidate")
    print(result.final_output)


if __name__ == "__main__":
    try:
        run_demo()
    except (RuntimeError, ImportError) as exc:
        raise SystemExit(str(exc)) from None

# endregion book:dynamic-pricing-agents-sdk-demo
