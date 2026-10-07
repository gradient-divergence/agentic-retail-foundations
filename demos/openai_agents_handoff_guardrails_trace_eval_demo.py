from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from demos.openai_agents_sdk_import import import_openai_agents_sdk


class PriceProposalInput(BaseModel):
    product_id: str
    current_price: float = Field(gt=0, allow_inf_nan=False)
    discount_pct: float = Field(ge=0, le=100, allow_inf_nan=False)


class PriceProposal(BaseModel):
    product_id: str
    current_price: float = Field(gt=0, allow_inf_nan=False)
    discount_pct: float = Field(ge=0, le=100, allow_inf_nan=False)
    new_price: float = Field(ge=0, allow_inf_nan=False)


def propose_price(payload: PriceProposalInput) -> PriceProposal:
    new_price = round(payload.current_price * (1 - payload.discount_pct / 100), 2)
    return PriceProposal(
        product_id=payload.product_id,
        current_price=payload.current_price,
        discount_pct=payload.discount_pct,
        new_price=new_price,
    )


TraceMetadata: TypeAlias = dict[str, str | float | int | bool]


@dataclass
class TraceEvent:
    name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: TraceMetadata = field(default_factory=dict)


class TraceLog:
    def __init__(self) -> None:
        self.trace_id = f"trace_{uuid4().hex[:8]}"
        self.events: list[TraceEvent] = []

    def log(self, name: str, metadata: TraceMetadata | None = None) -> None:
        self.events.append(TraceEvent(name=name, metadata=metadata or {}))


class EvalResult(BaseModel):
    trace_id: str
    score: float
    passed: bool
    notes: str | None = None


class GuardrailResult(BaseModel):
    passed: bool
    reason: str | None = None


class ExecutionSummary(BaseModel):
    status: Literal["executed", "blocked", "failed"]


def guardrail_check(recommendation: PriceProposal) -> GuardrailResult:
    if recommendation.discount_pct > 25 or recommendation.new_price < round(
        recommendation.current_price * 0.75, 2
    ):
        return GuardrailResult(passed=False, reason="discount_exceeds_threshold")
    return GuardrailResult(passed=True)


def eval_output(execution_summary: str, trace_id: str) -> EvalResult:
    try:
        summary = ExecutionSummary.model_validate_json(execution_summary)
        executed = summary.status == "executed"
    except ValidationError:
        executed = False
    score = 0.9 if executed else 0.4
    return EvalResult(trace_id=trace_id, score=score, passed=score >= 0.8)


def run_demo() -> None:
    agents_sdk = import_openai_agents_sdk()
    Agent = agents_sdk.Agent
    Runner = agents_sdk.Runner
    function_tool = agents_sdk.function_tool

    price_tool = function_tool(
        propose_price,
        description_override="Propose a new price based on a discount percentage.",
    )

    planner_agent = Agent(
        name="PlannerAgent",
        instructions=(
            "You propose price changes using the propose_price tool. "
            "Return only the tool output as JSON with product_id, current_price, discount_pct, and new_price."
        ),
        tools=[price_tool],
    )

    executor_agent = Agent(
        name="ExecutorAgent",
        instructions=(
            "You simulate approved pricing changes; no live price is changed. "
            'Return only JSON with a status of "executed", "blocked", or "failed".'
        ),
    )

    trace = TraceLog()

    task = "Propose a 15% discount for SKU123 priced at 49.99."
    trace.log("plan_started", {"task": task})
    plan = Runner.run_sync(planner_agent, task)
    trace.log("plan_completed", {"output": plan.final_output})

    try:
        recommendation = PriceProposal.model_validate_json(plan.final_output)
    except ValidationError:
        trace.log("handoff_blocked", {"reason": "invalid_price_proposal"})
        print("Guardrail blocked handoff: invalid_price_proposal")
        return
    if recommendation.product_id != "SKU123" or recommendation.current_price != 49.99:
        trace.log("handoff_blocked", {"reason": "proposal_does_not_match_task"})
        print("Guardrail blocked handoff: proposal_does_not_match_task")
        return
    guardrail = guardrail_check(recommendation)
    trace.log("guardrail_checked", {"passed": guardrail.passed, "reason": guardrail.reason or ""})

    if not guardrail.passed:
        trace.log("handoff_blocked")
        print("Guardrail blocked handoff:", guardrail.reason)
        return

    handoff_prompt = (
        f"Execute this approved change: {recommendation.product_id} to {recommendation.new_price}"
    )
    trace.log("handoff_started", {"payload": recommendation.model_dump()})
    execution = Runner.run_sync(executor_agent, handoff_prompt)
    trace.log("execution_completed", {"output": execution.final_output})

    eval_result = eval_output(execution.final_output, trace.trace_id)
    trace.log("eval_completed", {"score": eval_result.score, "passed": eval_result.passed})

    print("Trace:")
    for event in trace.events:
        print(event)
    print("Eval:", eval_result)


if __name__ == "__main__":
    try:
        run_demo()
    except (RuntimeError, ImportError) as exc:
        raise SystemExit(str(exc)) from None
