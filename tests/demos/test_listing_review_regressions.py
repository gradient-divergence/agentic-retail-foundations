import asyncio
import logging
import random
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from capstone.orchestrator import CapstoneOrchestrator
from capstone.policies import PolicyEvaluator
from capstone.schemas import ToolCall
from capstone.tools import ToolSpec
from capstone.tracing import TraceContext
from demos import inventory_event_api_demo, task_allocation_cnp_demo
from models.task import TaskStatus


@pytest.mark.parametrize("next_usage", [0, 1])
def test_token_exhaustion_persists_after_over_budget_usage(next_usage):
    handler = Mock(return_value={"ok": True})
    tool = ToolSpec(
        name="lookup",
        description="Fixture",
        permission="read",
        schema={"type": "object"},
        handler=handler,
    )
    gateway = CapstoneOrchestrator([tool], PolicyEvaluator(), max_tokens=10)
    trace = TraceContext(route="test")
    call = ToolCall(name="lookup", args={})

    for usage in (11, next_usage):
        result, audit = gateway.handle_tool_call(call, trace, token_count=usage)
        assert result.output == {"error": "budget_exhausted"}
        assert audit.decision == "budget_exhausted"
    assert trace.tokens_used == 11 + next_usage
    assert trace.tool_calls == 0
    handler.assert_not_called()


def test_adjustment_retry_is_idempotent_after_stock_decreases(monkeypatch):
    demo = inventory_event_api_demo
    monkeypatch.setattr(demo, "inventory_state", {})
    monkeypatch.setattr(demo, "applied_events", {})
    monkeypatch.setattr(demo, "publish_event", AsyncMock())
    common = {"product_id": "SKU", "location_id": "North"}
    with TestClient(demo.app) as client:
        receipt = {**common, "event_id": "receipt", "quantity": 5, "supplier_id": "S"}
        assert client.post("/events/receive", json=receipt).status_code == 200
        adjustment = {**common, "event_id": "adjustment", "quantity": -4, "reason_code": "count"}
        first = client.post("/events/adjust", json=adjustment)
        retry = client.post("/events/adjust", json=first.json())
        assert first.status_code == retry.status_code == 200
        assert retry.json() == first.json()
        assert demo.inventory_state["SKU"]["North"].quantity_available == 1
        invalid = client.post("/events/adjust", json={**adjustment, "event_id": "new"})
        assert invalid.status_code == 400
        reused = client.post("/events/adjust", json={**first.json(), "quantity": -1})
        assert reused.status_code == 409
        assert demo.inventory_state["SKU"]["North"].quantity_available == 1
        assert set(demo.applied_events) == {"receipt", "adjustment"}


def test_contract_net_execution_replays_after_caller_randomness(monkeypatch, capsys, caplog):
    execute = task_allocation_cnp_demo.StoreAgent.execute_task
    outcomes = []

    async def record_execution(agent, task):
        success = await execute(agent, task)
        outcomes.append((task.id, agent.agent_id, success, task.status))
        return success

    monkeypatch.setattr(task_allocation_cnp_demo.StoreAgent, "execute_task", record_execution)
    monkeypatch.setattr(task_allocation_cnp_demo.asyncio, "sleep", AsyncMock())
    state = random.getstate()
    try:
        for seed in (1, 3):
            random.seed(seed)
            asyncio.run(task_allocation_cnp_demo.demo_contract_net_protocol())
            assert capsys.readouterr().out.splitlines() == [
                "Task delivery-north awarded to North Store (bid=5.40)",
                "Task inventory-south awarded to South Store (bid=2.40)",
            ]
    finally:
        random.setstate(state)
    expected = [
        ("delivery-north", "store-north", True, TaskStatus.COMPLETED),
        ("inventory-south", "store-south", True, TaskStatus.COMPLETED),
    ]
    assert outcomes == expected * 2
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_readme_demo_module_command_runs_from_companion_root():
    root = Path(__file__).resolve().parents[2]
    readme = (root / "README.md").read_text()
    demo_commands = readme.split("**Run Demo Scripts:**", 1)[1].split("```", 2)[1]
    example = next(
        line.strip().removeprefix("# e.g., ") for line in demo_commands.splitlines() if "e.g." in line
    )
    command = shlex.split(example)
    assert command == ["python", "-m", "demos.task_allocation_cnp_demo"]
    completed = subprocess.run(
        [sys.executable, *command[1:]],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert completed.stdout.splitlines() == [
        "Task delivery-north awarded to North Store (bid=5.40)",
        "Task inventory-south awarded to South Store (bid=2.40)",
    ]
