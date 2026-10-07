"""Arithmetic for scoring_rubric.md; component scores are supplied by the evaluator."""

from math import fsum, isfinite


def score_outcome(
    action_accuracy: float, policy_compliance: float, kpi_impact: float, operational_health: float
) -> tuple[float, bool]:
    components = (action_accuracy, policy_compliance, kpi_impact, operational_health)
    if any(not isfinite(value) or not 0 <= value <= 1 for value in components):
        raise ValueError("Component scores must be finite numbers between 0 and 1")
    score = fsum(weight * value for weight, value in zip((0.5, 0.2, 0.2, 0.1), components, strict=True))
    return score, policy_compliance == 1 and score >= 0.85
