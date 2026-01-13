"""
Compact, print-friendly excerpt for goal scoring in the BDI agent.
"""

# region book:bdi-goal-scoring-excerpt
from pydantic import BaseModel

from agents.bdi import InventoryBDIAgent


class GoalScores(BaseModel):
    minimize_stockouts: float
    reduce_excess: float
    maximize_profit: float
    preserve_freshness: float


def score_goals(agent: InventoryBDIAgent) -> GoalScores:
    """Return a lightweight score map used to rank goals."""
    return GoalScores(
        minimize_stockouts=agent._evaluate_stockout_prevention(),
        reduce_excess=agent._evaluate_excess_reduction(),
        maximize_profit=agent._evaluate_profit_maximization(),
        preserve_freshness=agent._evaluate_freshness(),
    )


# endregion book:bdi-goal-scoring-excerpt
