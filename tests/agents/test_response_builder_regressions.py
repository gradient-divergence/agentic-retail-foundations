import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents import response_builder as rb
from connectors.dummy_order_system import DummyOrderSystem


def test_prompt_builder_imports_without_sdk():
    import subprocess
    import sys

    code = """
import sys
sys.modules['openai'] = None
from agents.response_builder import build_response_prompt
assert build_response_prompt(customer_info={}, intent='general_inquiry', message='hello',
    context_data={}, conversation_history=[])
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_order_connector_fields_reach_customer_prompt():
    order = asyncio.run(DummyOrderSystem().get_order_details("ORD988"))
    prompt = rb.build_response_prompt(
        customer_info={},
        intent="order_status",
        message="Where is it?",
        context_data={"order_details": order},
        conversation_history=[],
    )
    assert "Estimated delivery: 2023-11-01" in prompt
    assert "Tracking: TRK123" in prompt


@pytest.mark.parametrize("reply", ['{"unrelated": ["delete_order"]}', '["send_email", 42]'])
def test_malformed_actions_are_rejected(monkeypatch, reply):
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
    monkeypatch.setattr(rb, "safe_chat_completion", AsyncMock(return_value=completion))
    result = asyncio.run(
        rb.extract_actions(
            AsyncMock(),
            intent="general_inquiry",
            response_text="Hello",
            context_data={},
            model="fake",
            logger=logging.getLogger(__name__),
        )
    )
    assert result == []


def test_duplicate_actions_are_not_repeated(monkeypatch):
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='["send_email", "send_email"]'))]
    )
    monkeypatch.setattr(rb, "safe_chat_completion", AsyncMock(return_value=completion))
    result = asyncio.run(
        rb.extract_actions(
            AsyncMock(),
            intent="general_inquiry",
            response_text="I will email you",
            context_data={},
            model="fake",
            logger=logging.getLogger(__name__),
        )
    )
    assert result == [{"type": "send_email"}]
