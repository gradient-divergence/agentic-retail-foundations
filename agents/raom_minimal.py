from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import TracebackType
from typing import Protocol, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger(__name__)

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list[JsonPrimitive] | dict[str, JsonPrimitive]


# region book:raom-models
class ActionRequest(BaseModel):
    tool_name: str
    payload: dict[str, JsonValue]
    estimated_cost: float | None = None
    request_id: str = Field(default_factory=lambda: f"req_{uuid4().hex}")


class ActionResult(BaseModel):
    status: str
    output: dict[str, JsonValue] | None = None
    error: str | None = None
    trace_id: str | None = None
    span_id: str | None = None


class PolicyDecision(BaseModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)


# endregion book:raom-models


# region book:raom-tools
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel], BaseModel]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def validate_and_run(self, tool_name: str, payload: dict[str, JsonValue]) -> BaseModel:
        if tool_name not in self._tools:
            raise KeyError(f"Unknown tool: {tool_name}")
        spec = self._tools[tool_name]
        validated_input = spec.input_model.model_validate(payload)
        result = spec.handler(validated_input)
        if isinstance(result, spec.output_model):
            return result
        return spec.output_model.model_validate(result)


# endregion book:raom-tools


# region book:raom-tracing
class TraceEvent(BaseModel):
    name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass
class TraceSpan:
    name: str
    trace_id: str = field(default_factory=lambda: f"trace_{uuid4().hex}")
    span_id: str = field(default_factory=lambda: f"span_{uuid4().hex}")
    events: list[TraceEvent] = field(default_factory=list)

    def __enter__(self) -> TraceSpan:
        self.log_event("span_started")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc:
            self.log_event("span_error", {"error": str(exc)})
        self.log_event("span_finished")

    def log_event(self, name: str, metadata: dict[str, JsonValue] | None = None) -> None:
        self.events.append(TraceEvent(name=name, metadata=metadata or {}))


# endregion book:raom-tracing


# region book:raom-policy
class PolicyRule(Protocol):
    def evaluate(self, action: ActionRequest) -> list[str]: ...


@dataclass(frozen=True)
class ToolAllowlistRule:
    allowed_tools: set[str]

    def evaluate(self, action: ActionRequest) -> list[str]:
        if action.tool_name not in self.allowed_tools:
            return [f"tool_not_allowed:{action.tool_name}"]
        return []


@dataclass(frozen=True)
class MaxCostRule:
    max_cost: float

    def evaluate(self, action: ActionRequest) -> list[str]:
        if action.estimated_cost is None:
            return []
        if action.estimated_cost > self.max_cost:
            return [f"cost_exceeded:{action.estimated_cost:.2f}>{self.max_cost:.2f}"]
        return []


class PolicyEngine:
    def __init__(self, rules: list[PolicyRule] | None = None) -> None:
        self.rules = rules or []

    def evaluate(self, action: ActionRequest) -> PolicyDecision:
        reasons: list[str] = []
        for rule in self.rules:
            reasons.extend(rule.evaluate(action))
        return PolicyDecision(allowed=not reasons, reasons=reasons)


# endregion book:raom-policy


# region book:raom-eval
class EvaluationRecord(BaseModel):
    trace_id: str
    observation: dict[str, JsonValue]
    action: ActionRequest
    result: ActionResult
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvaluationRecorder:
    def __init__(self) -> None:
        self.records: list[EvaluationRecord] = []

    def record(self, record: EvaluationRecord) -> None:
        self.records.append(record)


# endregion book:raom-eval


# region book:raom-agent
DecisionFunction: TypeAlias = Callable[[BaseModel], ActionRequest]


class RAOMAgent:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        policy_engine: PolicyEngine,
        decision_fn: DecisionFunction,
        evaluator: EvaluationRecorder | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.policy_engine = policy_engine
        self.decision_fn = decision_fn
        self.evaluator = evaluator

    def step(self, observation: BaseModel, trace_id: str | None = None) -> ActionResult:
        with TraceSpan("raom.step", trace_id=trace_id or f"trace_{uuid4().hex}") as span:
            action = self.decision_fn(observation)
            decision = self.policy_engine.evaluate(action)
            if not decision.allowed:
                result = ActionResult(
                    status="blocked",
                    error="; ".join(decision.reasons),
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                )
                self._record_eval(span, observation, action, result)
                return result

            try:
                output_model = self.tool_registry.validate_and_run(action.tool_name, action.payload)
                result = ActionResult(
                    status="ok",
                    output=output_model.model_dump(),
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                )
            except Exception as exc:
                logger.warning("Tool execution failed: %s", exc)
                result = ActionResult(
                    status="error",
                    error=str(exc),
                    trace_id=span.trace_id,
                    span_id=span.span_id,
                )

            self._record_eval(span, observation, action, result)
            return result

    def _record_eval(
        self,
        span: TraceSpan,
        observation: BaseModel,
        action: ActionRequest,
        result: ActionResult,
    ) -> None:
        if not self.evaluator:
            return
        self.evaluator.record(
            EvaluationRecord(
                trace_id=span.trace_id,
                observation=observation.model_dump(),
                action=action,
                result=result,
            )
        )


# endregion book:raom-agent


# region book:raom-demo
class InventoryObservation(BaseModel):
    product_id: str
    on_hand: int
    reorder_point: int
    suggested_order: int


class ReorderRequest(BaseModel):
    product_id: str
    quantity: int


class ReorderResponse(BaseModel):
    status: str
    order_id: str


def place_reorder(payload: ReorderRequest) -> ReorderResponse:
    return ReorderResponse(status="submitted", order_id=f"PO-{uuid4().hex[:8]}")


def reorder_decision(observation: InventoryObservation) -> ActionRequest:
    quantity = observation.suggested_order if observation.on_hand <= observation.reorder_point else 0
    return ActionRequest(
        tool_name="place_reorder",
        payload={"product_id": observation.product_id, "quantity": quantity},
        estimated_cost=120.0,
    )


# endregion book:raom-demo
