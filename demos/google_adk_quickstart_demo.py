from __future__ import annotations

import os

from pydantic import BaseModel


class TimeLookupResult(BaseModel):
    status: str
    city: str
    time: str


class InventoryLookupResult(BaseModel):
    status: str
    sku: str
    store_id: str
    on_hand: int
    reorder_point: int
    action: str


def get_current_time(city: str) -> TimeLookupResult:
    return TimeLookupResult(status="success", city=city, time="10:30 AM")


def lookup_inventory(sku: str, store_id: str) -> InventoryLookupResult:
    on_hand = 8
    reorder_point = 12
    action = "review_reorder" if on_hand < reorder_point else "ok"
    return InventoryLookupResult(
        status="success",
        sku=sku,
        store_id=store_id,
        on_hand=on_hand,
        reorder_point=reorder_point,
        action=action,
    )


def __getattr__(name: str):
    if name != "root_agent":
        raise AttributeError(name)
    if not (os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()):
        raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) to run this provider demo.")
    try:
        from google.adk.agents.llm_agent import Agent
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'google-adk is not installed. Install with: uv pip install -e ".[agent_protocols]"'
        ) from exc

    agent = Agent(
        model="gemini-3-flash-preview",
        name="retail_ops_agent",
        description="Answers retail operations questions using time and inventory tools.",
        instruction=(
            "You are a retail operations assistant. Use tools to answer questions about "
            "store inventory and local time. Be concise and action-oriented."
        ),
        tools=[get_current_time, lookup_inventory],
    )
    globals()["root_agent"] = agent
    return agent


def main() -> None:
    print("ADK agent ready:", __getattr__("root_agent").name)
    print("Sample tool inputs:")
    print(" - get_current_time(city='Seattle')")
    print(" - lookup_inventory(sku='SKU-123', store_id='SEA-01')")
    print("Run with: adk run my_agent (from an ADK project directory).")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ImportError) as exc:
        raise SystemExit(str(exc)) from None
