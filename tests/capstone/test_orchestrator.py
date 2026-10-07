from unittest.mock import Mock

import pytest

from capstone.orchestrator import CapstoneOrchestrator
from capstone.policies import PolicyEvaluator
from capstone.schemas import ToolCall
from capstone.tools import ToolSpec
from capstone.tracing import TraceContext


@pytest.mark.parametrize(
    "args", [{}, {"quantity": "2"}, {"quantity": True}, {"quantity": -1}, {"quantity": 2, "extra": 3}]
)
def test_gateway_rejects_schema_invalid_arguments_before_execution(args):
    handler = Mock(return_value={"reserved": True})
    tool = ToolSpec(
        name="reserve",
        description="Reserve",
        permission="write",
        handler=handler,
        schema={
            "type": "object",
            "properties": {"quantity": {"type": "integer", "minimum": 1}},
            "required": ["quantity"],
            "additionalProperties": False,
        },
    )
    result, audit = CapstoneOrchestrator([tool], PolicyEvaluator()).handle_tool_call(
        ToolCall(name="reserve", args=args), TraceContext(route="test")
    )
    assert result.status == "error"
    assert result.output["error"] == audit.decision == "invalid_arguments"
    handler.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("amount", "invalid"),
        ("amount", "nan"),
        ("amount", True),
        ("risk_score", "invalid"),
        ("risk_score", "nan"),
        ("risk_score", -1),
        ("risk_score", 2),
    ],
)
def test_gateway_rejects_malformed_policy_inputs(field, value):
    handler = Mock(return_value={"committed": True})
    tool = ToolSpec(
        name="supplier_commitment",
        description="Commit",
        permission="restricted",
        handler=handler,
        schema={"type": "object"},
    )
    result, audit = CapstoneOrchestrator([tool], PolicyEvaluator()).handle_tool_call(
        ToolCall(name=tool.name, args={"amount": 100, field: value}), TraceContext(route="test")
    )
    assert result.status == "error"
    assert audit.decision == "invalid_arguments"
    handler.assert_not_called()


def test_gateway_valid_call_and_policy_threshold():
    handler = Mock(return_value={"committed": True})
    tool = ToolSpec(
        name="supplier_commitment",
        description="Commit",
        permission="restricted",
        handler=handler,
        schema={"type": "object", "required": ["amount"]},
    )
    gateway = CapstoneOrchestrator([tool], PolicyEvaluator())
    trace = TraceContext(route="test")
    result, audit = gateway.handle_tool_call(ToolCall(name=tool.name, args={"amount": 100}), trace)
    assert result.status == "ok" and audit.trace_id == result.trace_id == trace.trace_id
    result, _ = gateway.handle_tool_call(ToolCall(name=tool.name, args={"amount": 5000}), trace)
    assert result.output["error"] == "blocked_by_policy"
    assert handler.call_count == 1


@pytest.mark.parametrize("args", [{}, {"amount": None}, {"amount": -100}])
def test_commitment_requires_nonnegative_amount(args):
    handler = Mock(return_value={"committed": True})
    tool = ToolSpec(
        name="supplier_commitment",
        description="Commit",
        permission="restricted",
        handler=handler,
        schema={"type": "object"},
    )
    result, audit = CapstoneOrchestrator([tool], PolicyEvaluator()).handle_tool_call(
        ToolCall(name=tool.name, args=args), TraceContext(route="test")
    )
    assert result.output["error"] == audit.decision == "invalid_arguments"
    handler.assert_not_called()
