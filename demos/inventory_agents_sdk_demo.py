from __future__ import annotations

# region book:inventory-agents-sdk-demo
from demos.openai_agents_sdk_import import import_openai_agents_sdk

inventory_db = {"product_123": 20}


def check_inventory(product_id: str) -> int:
    return inventory_db.get(product_id, 0)


def order_product(product_id: str, amount: int) -> str:
    if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
        raise ValueError("Order amount must be a positive integer")
    current = inventory_db.get(product_id, 0)
    inventory_db[product_id] = current + amount
    return f"Ordered {amount} units of {product_id}, new stock is {inventory_db[product_id]}."


def run_demo() -> None:
    agents_sdk = import_openai_agents_sdk()
    Agent = agents_sdk.Agent
    Runner = agents_sdk.Runner
    function_tool = agents_sdk.function_tool

    check_inventory_tool = function_tool(
        check_inventory,
        description_override="Check the current stock level of a product by ID.",
    )
    order_tool = function_tool(
        order_product,
        description_override="Order more units of a product by ID.",
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


if __name__ == "__main__":
    try:
        run_demo()
    except (RuntimeError, ImportError) as exc:
        raise SystemExit(str(exc)) from None

# endregion book:inventory-agents-sdk-demo
