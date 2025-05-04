import logging
from types import SimpleNamespace

import pytest

from agents import response_builder as rb

# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


@pytest.mark.asyncio
async def test_build_response_prompt_contains_sections():
    prompt = rb.build_response_prompt(
        customer_info={"name": "Alice", "loyalty_tier": "Gold"},
        intent="general_inquiry",
        message="What are your store hours?",
        context_data={},
        conversation_history=[],
    )
    assert "CUSTOMER INFORMATION" in prompt
    assert "INSTRUCTIONS FOR RESPONSE" in prompt


@pytest.mark.asyncio
async def test_extract_actions_rule_based(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        return _FakeCompletion("[]")

    monkeypatch.setattr(rb, "safe_chat_completion", fake_safe_chat_completion)

    actions = await rb.extract_actions(
        client=None,
        intent="order_status",
        response_text="Your order is currently delayed. I will escalate this issue.",
        context_data={"order_details": {"order_id": "ABC123", "status": "delayed"}},
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert {
        "type": "escalate_issue",
        "reason": "delayed_order",
        "order_id": "ABC123",
    } in actions
