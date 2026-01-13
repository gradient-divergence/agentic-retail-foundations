#!/usr/bin/env python3
# region book:eval-hook-demo
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

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
        trace_id=f"trace_{uuid4().hex[:8]}",
        question="Can I return sale items?",
        answer="We accept returns within 30 days with receipt.",
        policy_violation=False,
        latency_ms=824,
        timestamp=datetime.now(timezone.utc),
    )
    result = eval_hook(run)
    log_eval(result)


if __name__ == "__main__":
    run_example()
# endregion book:eval-hook-demo
