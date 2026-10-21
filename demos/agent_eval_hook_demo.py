#!/usr/bin/env python3
# region book:eval-hook-demo
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import BaseModel, Field


@dataclass
class AgentRun:
    trace_id: str
    question: str
    answer: str
    policy_violation: bool
    latency_ms: int
    timestamp: datetime


class EvalResult(BaseModel):
    trace_id: str
    score: float = Field(..., ge=0, le=1)
    passed: bool
    failure_reason: str | None = None


def eval_hook(run: AgentRun) -> EvalResult:
    """Arbitrary keyword scores illustrate the record interface, not answer correctness."""
    score = 0.92
    failure_reason = None

    if run.policy_violation:
        score = 0.0
        failure_reason = "policy_violation"
    elif "return" in run.question.lower() and "policy" not in run.answer.lower():
        score = 0.4
        failure_reason = "missing_policy_reference"

    return EvalResult(
        trace_id=run.trace_id,
        score=score,
        passed=score >= 0.8,
        failure_reason=failure_reason,
    )


def log_eval(result: EvalResult) -> None:
    print(result.model_dump())


def run_example() -> None:
    run = AgentRun(
        trace_id="trace_demo",
        question="Can I return sale items?",
        answer="We accept returns within 30 days with receipt.",
        policy_violation=False,
        latency_ms=824,
        timestamp=datetime(2026, 10, 13, tzinfo=timezone.utc),
    )
    for answer in ("policy: purple elephants", "Returns accepted within 30 days with receipt."):
        run.answer = answer
        print(f"Keyword stub answer: {answer}")
        log_eval(eval_hook(run))
    # A supplied policy fact gives a separate, grounded assertion for this fixture.
    policy_window_days = 30
    correct_answer = "Returns accepted within 30 days with receipt."
    assert f"{policy_window_days} days" in correct_answer and "receipt" in correct_answer
    assert "30 days" not in "policy: purple elephants"
    print("Grounded fixture assertion: 30-day window and receipt requirement verified")


if __name__ == "__main__":
    run_example()
# endregion book:eval-hook-demo
