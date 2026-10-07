from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from demos import tool_calling_rag_demo as demo


@pytest.mark.parametrize(
    "call",
    [
        SimpleNamespace(name="delete_order", args={}),
        {"name": "search_catalog", "args": {"query": 42}},
        {"name": "get_return_policy", "args": {"order_id": "ORD987"}},
    ],
)
def test_invalid_call_is_rejected_before_execution(monkeypatch, call):
    monkeypatch.setattr(demo, "route_query", lambda query: call)
    with pytest.raises(ValidationError):
        demo.run_tool_call("query")


def test_valid_tool_call_keeps_policy_answer():
    reply = demo.run_tool_call("Can I return this?")
    assert reply.intent == "get_return_policy"
    assert reply.answer == "Returns accepted within 30 days with receipt. Worn items excluded."
    assert reply.sources == ["policy:returns"]


def test_main_demonstrates_a_catalog_match(capsys):
    demo.main()
    assert "Top matches: All-Terrain Running Shoes" in capsys.readouterr().out
