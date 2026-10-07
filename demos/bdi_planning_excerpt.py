"""
Compact, print-friendly excerpt for intention planning in the BDI agent.
"""

# region book:bdi-planning-excerpt
from pydantic import BaseModel

from agents.bdi import InventoryBDIAgent


class Intention(BaseModel):
    action: str
    product_id: str
    quantity: int | None = None
    discount_percentage: int | None = None
    supplier_id: str | None = None
    promotion_type: str | None = None
    priority: float


def plan_for_goals(agent: InventoryBDIAgent, prioritized_goals: list[str]) -> list[Intention]:
    """Trigger planning routines for a subset of goals."""
    agent.generate_intentions(prioritized_goals)
    return [Intention(**intention) for intention in agent.active_intentions]


# endregion book:bdi-planning-excerpt
