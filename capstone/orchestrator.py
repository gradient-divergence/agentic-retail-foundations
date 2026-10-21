from __future__ import annotations

from math import isfinite

from jsonschema import ValidationError as SchemaValidationError
from jsonschema import validate
from pydantic import JsonValue, ValidationError

from capstone.policies import PolicyEvaluator, PolicyInput
from capstone.schemas import AuditRecord, ToolCall, ToolResult
from capstone.tools import ToolSpec
from capstone.tracing import TraceContext


class CapstoneOrchestrator:
    """Minimal orchestrator and tool gateway for capstone demos."""

    def __init__(
        self,
        tools: list[ToolSpec],
        policy: PolicyEvaluator,
        *,
        max_tool_calls: int = 10,
        max_tokens: int = 4096,
    ) -> None:
        if (
            type(max_tool_calls) is not int
            or type(max_tokens) is not int
            or min(max_tool_calls, max_tokens) < 0
        ):
            raise ValueError("Budgets must be nonnegative integers")
        self._tools = {tool.name: tool for tool in tools}
        self._policy = policy
        self.max_tool_calls, self.max_tokens = max_tool_calls, max_tokens

    def handle_tool_call(
        self, call: ToolCall, trace: TraceContext, actor: str = "capstone-agent", *, token_count: int = 0
    ) -> tuple[ToolResult, AuditRecord]:
        if type(token_count) is not int or token_count < 0:
            raise ValueError("Token count must be a nonnegative integer from provider usage")
        trace.tokens_used += token_count
        if trace.tool_calls >= self.max_tool_calls or trace.tokens_used > self.max_tokens:
            return (
                ToolResult(
                    name=call.name,
                    status="error",
                    output={"error": "budget_exhausted"},
                    trace_id=trace.trace_id,
                ),
                AuditRecord(
                    trace_id=trace.trace_id,
                    actor=actor,
                    action=call.name,
                    decision="budget_exhausted",
                    evidence=[],
                ),
            )
        trace.tool_calls += 1
        spec = self._tools.get(call.name)
        if spec is None:
            return (
                ToolResult(
                    name=call.name,
                    status="error",
                    output={"error": "unknown_tool"},
                    trace_id=trace.trace_id,
                ),
                AuditRecord(
                    trace_id=trace.trace_id,
                    actor=actor,
                    action=call.name,
                    decision="unknown_tool",
                    evidence=[],
                ),
            )

        try:
            validate(instance=call.args, schema=spec.schema_)
            policy_input = PolicyInput(
                action=call.name,
                amount=_extract_amount(call.args),
                currency=call.args.get("currency", "USD"),
                risk_score=_coerce_float(call.args.get("risk_score")) or 0.0,
                evidence=[f"permission={spec.permission}"],
            )
            if call.name in {"price_change", "supplier_commitment"} and policy_input.amount is None:
                raise ValueError("Financial actions require an amount")
            if call.name == "supplier_commitment" and policy_input.amount < 0:
                raise ValueError("Supplier commitments require a nonnegative amount")
        except (SchemaValidationError, ValidationError, ValueError):
            return (
                ToolResult(
                    name=call.name,
                    status="error",
                    output={"error": "invalid_arguments"},
                    trace_id=trace.trace_id,
                ),
                AuditRecord(
                    trace_id=trace.trace_id,
                    actor=actor,
                    action=call.name,
                    decision="invalid_arguments",
                    evidence=[f"permission={spec.permission}"],
                ),
            )
        decision = self._policy.evaluate(policy_input)
        if spec.permission in {"write", "restricted"} and not decision.allowed:
            return (
                ToolResult(
                    name=call.name,
                    status="error",
                    output={"error": "blocked_by_policy", "reason": decision.reason},
                    trace_id=trace.trace_id,
                ),
                AuditRecord(
                    trace_id=trace.trace_id,
                    actor=actor,
                    action=call.name,
                    decision=decision.reason,
                    evidence=policy_input.evidence,
                ),
            )

        output = spec.execute(**call.args)
        return (
            ToolResult(
                name=call.name,
                status="ok",
                output=output,
                trace_id=trace.trace_id,
            ),
            AuditRecord(
                trace_id=trace.trace_id,
                actor=actor,
                action=call.name,
                decision=decision.reason,
                evidence=policy_input.evidence,
            ),
        )


def _extract_amount(args: dict[str, JsonValue]) -> float | None:
    value = args.get("amount")
    if value is None:
        value = args.get("price_change")
    return _coerce_float(value)


def _coerce_float(value: JsonValue | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError("Policy values must be finite numbers")
    number = float(value)
    if not isfinite(number):
        raise ValueError("Policy values must be finite numbers")
    return number
