import asyncio
from unittest.mock import AsyncMock

from agents.llm import RetailCustomerServiceAgent
from connectors.dummy_db import DummyDB
from connectors.dummy_order_system import DummyOrderSystem
from demos.customer_service_agent_excerpt import Inquiry, handle_inquiry


def test_product_inquiry_resolves_identifier_to_catalog_id(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = RetailCustomerServiceAgent(DummyDB(), DummyOrderSystem(), DummyDB(), {})
    agent._classify_intent = AsyncMock(return_value="product_question")
    agent._extract_product_identifier = AsyncMock(return_value="Running Shoes")

    async def fake_response(**kwargs):
        return {
            "message": kwargs["context_data"]["product_details"]["name"],
            "intent": "product_question",
            "actions": [],
        }

    agent._generate_response = fake_response
    reply = asyncio.run(handle_inquiry(agent, Inquiry(customer_id="C123", message="Shoes?")))
    assert reply.message == "Running Shoes"


def test_malformed_reply_returns_error_envelope(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = RetailCustomerServiceAgent(DummyDB(), DummyOrderSystem(), DummyDB(), {})
    agent._classify_intent = AsyncMock(return_value="general_inquiry")
    agent._generate_response = AsyncMock(return_value={"message": None, "actions": "bad"})
    reply = asyncio.run(handle_inquiry(agent, Inquiry(customer_id="C123", message="Hi")))
    assert reply.intent == "error"
    assert reply.actions == []


def test_missing_customer_uses_anonymous_context(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion

    completion = MagicMock(spec=ChatCompletion)
    completion.choices = [SimpleNamespace(message=SimpleNamespace(content="Hello"))]
    client = AsyncMock(spec=AsyncOpenAI)
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=completion)))
    agent = RetailCustomerServiceAgent(DummyDB(), DummyOrderSystem(), DummyDB(), {}, client=client)
    agent._classify_intent = AsyncMock(return_value="general_inquiry")
    reply = asyncio.run(handle_inquiry(agent, Inquiry(customer_id="missing", message="Hi")))
    assert reply.message == "Hello"
