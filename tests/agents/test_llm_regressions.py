import asyncio
from unittest.mock import AsyncMock

import pytest

from agents.llm import RetailCustomerServiceAgent


def test_no_key_agent_imports_and_initializes_without_sdk():
    import subprocess
    import sys

    code = """
import os, sys
os.environ.pop('OPENAI_API_KEY', None)
sys.modules['openai'] = None
from agents.llm import RetailCustomerServiceAgent
agent = RetailCustomerServiceAgent(None, None, None, {})
assert agent.client is None
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("message", ["Where is my package?", "Please return this product"])
def test_words_are_not_order_ids(monkeypatch, message):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = RetailCustomerServiceAgent(None, None, None, {})
    assert asyncio.run(agent._extract_order_id(message, [])) is None


def test_recent_id_is_found_after_an_ordinary_word(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = RetailCustomerServiceAgent(None, None, None, {})
    assert asyncio.run(agent._extract_order_id("Please check ORD987", [{"order_id": "ORD987"}])) == "ORD987"


def test_multiple_order_ids_are_ambiguous(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = RetailCustomerServiceAgent(None, None, None, {})
    assert asyncio.run(agent._extract_order_id("ORD987 or ORD988?", [])) is None


def test_client_can_be_injected_without_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = AsyncMock()
    agent = RetailCustomerServiceAgent(None, None, None, {}, client=client)
    agent._classify_intent = AsyncMock(return_value="general_inquiry")
    agent._generate_response = AsyncMock(
        return_value={"message": "Hello", "intent": "general_inquiry", "actions": []}
    )
    agent.customer_db = AsyncMock()
    agent.customer_db.get_customer.return_value = {}
    agent.order_system = AsyncMock()
    agent.order_system.get_recent_orders.return_value = []
    assert asyncio.run(agent.process_customer_inquiry("C123", "Hi"))["message"] == "Hello"


def test_injected_client_exercises_entire_model_pipeline(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion

    from connectors.dummy_db import DummyDB
    from connectors.dummy_order_system import DummyOrderSystem

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    completions = []
    for content in ["order_status", "Your order is shipped.", "[]", "neutral"]:
        completion = MagicMock(spec=ChatCompletion)
        completion.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        completions.append(completion)
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return completions.pop(0)

    client = AsyncMock(spec=AsyncOpenAI)
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    agent = RetailCustomerServiceAgent(DummyDB(), DummyOrderSystem(), DummyDB(), {}, client=client)
    reply = asyncio.run(agent.process_customer_inquiry("C123", "Please check ORD988"))
    assert reply["message"] == "Your order is shipped."
    assert reply["actions"] == []
    assert "Estimated delivery: 2023-11-01" in calls[1]["messages"][0]["content"]
    assert "Tracking: TRK123" in calls[1]["messages"][0]["content"]
    assert len(calls) == 4
