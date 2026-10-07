import pytest
from pydantic import ValidationError

from agents.raom_minimal import (
    ActionRequest,
    EvaluationRecorder,
    InventoryObservation,
    MaxCostRule,
    PolicyEngine,
    RAOMAgent,
    ReorderRequest,
    ReorderResponse,
    ToolAllowlistRule,
    ToolRegistry,
    ToolSpec,
    TraceSpan,
    place_reorder,
    reorder_decision,
)


def test_tool_registry_validates_payload():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="place_reorder",
            description="Submit a reorder.",
            input_model=ReorderRequest,
            output_model=ReorderResponse,
            handler=place_reorder,
        )
    )

    with pytest.raises(ValidationError):
        registry.validate_and_run("place_reorder", {"product_id": "SKU-1"})


def test_policy_engine_blocks_unapproved_tool():
    policy = PolicyEngine(rules=[ToolAllowlistRule(allowed_tools={"safe_tool"}), MaxCostRule(max_cost=50)])
    action = ActionRequest(
        tool_name="place_reorder",
        payload={"product_id": "SKU-1", "quantity": 5},
        estimated_cost=120.0,
    )
    decision = policy.evaluate(action)
    assert decision.allowed is False
    assert any("tool_not_allowed" in reason for reason in decision.reasons)
    assert any("cost_exceeded" in reason for reason in decision.reasons)


def test_raom_agent_records_eval():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="place_reorder",
            description="Submit a reorder.",
            input_model=ReorderRequest,
            output_model=ReorderResponse,
            handler=place_reorder,
        )
    )
    policy = PolicyEngine(rules=[ToolAllowlistRule(allowed_tools={"place_reorder"})])
    evaluator = EvaluationRecorder()
    agent = RAOMAgent(
        tool_registry=registry,
        policy_engine=policy,
        decision_fn=reorder_decision,
        evaluator=evaluator,
    )

    result = agent.step(
        InventoryObservation(
            product_id="SKU-1",
            on_hand=3,
            reorder_point=5,
            suggested_order=12,
        )
    )

    assert result.status == "ok"
    assert result.output is not None
    assert result.trace_id is not None
    assert len(evaluator.records) == 1


def test_trace_span_records_events():
    with TraceSpan("raom.test") as span:
        span.log_event("decision_made", {"tool": "place_reorder"})

    event_names = [event.name for event in span.events]
    assert "span_started" in event_names
    assert "decision_made" in event_names
    assert "span_finished" in event_names


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -1.0])
def test_invalid_cost_cannot_bypass_budget(cost):
    with pytest.raises(ValidationError):
        ActionRequest(tool_name="place_reorder", payload={}, estimated_cost=cost)


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), -1.0])
def test_invalid_policy_limit_is_rejected(limit):
    with pytest.raises(ValueError):
        MaxCostRule(limit)


def test_blocked_action_does_not_execute_and_records_trace():
    executed = []
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="place_reorder",
            description="Submit a reorder.",
            input_model=ReorderRequest,
            output_model=ReorderResponse,
            handler=lambda payload: executed.append(payload),
        )
    )
    evaluator = EvaluationRecorder()
    agent = RAOMAgent(registry, PolicyEngine([MaxCostRule(50)]), reorder_decision, evaluator)
    result = agent.step(
        InventoryObservation(product_id="SKU-1", on_hand=3, reorder_point=5, suggested_order=12),
        trace_id="parent-trace",
    )
    assert result.status == "blocked"
    assert not executed
    assert evaluator.records[0].result == result
    assert result.trace_id == evaluator.records[0].trace_id == "parent-trace"


def test_tool_failure_is_recorded_with_trace():
    evaluator = EvaluationRecorder()
    agent = RAOMAgent(ToolRegistry(), PolicyEngine(), reorder_decision, evaluator)
    result = agent.step(
        InventoryObservation(product_id="SKU-1", on_hand=3, reorder_point=5, suggested_order=12),
        trace_id="parent-trace",
    )
    assert result.status == "error"
    assert "Unknown tool" in result.error
    assert evaluator.records[0].result.span_id == result.span_id
