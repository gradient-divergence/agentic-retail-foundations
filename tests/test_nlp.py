import logging
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import builtins

# Module to test
from utils import nlp as nlp_mod

# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------


class _FakeChoice:
    def __init__(self, content):
        self.message = MagicMock()
        self.message.content = content


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_intent(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        assert len(args) > 0  # client is positional
        assert 'messages' in kwargs
        # Check if the system prompt is correct
        assert "You are an intent classification system" in kwargs['messages'][0]['content']
        return _FakeCompletion("order_status")

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    # Mock isinstance temporarily to allow AsyncMock to pass the check
    def mock_isinstance(obj, classinfo):
        # If checking for AsyncOpenAI, return True if obj is our AsyncMock
        if classinfo is nlp_mod.AsyncOpenAI and isinstance(obj, AsyncMock):
            return True
        # Otherwise, fallback to the real isinstance
        return builtins.isinstance(obj, classinfo)

    with patch('utils.nlp.isinstance', mock_isinstance):
        # Use the new signature
        intent = await nlp_mod.classify_intent(
            client=AsyncMock(),
            message="Where is my package?",
        )
    assert intent == "order_status"


@pytest.mark.asyncio
async def test_extract_order_id(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        assert len(args) > 0
        assert 'messages' in kwargs
        # Check system prompt related to order IDs
        assert "You extract order IDs" in kwargs['messages'][0]['content']
        return _FakeCompletion("ABC123")

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    order_id = await nlp_mod.extract_order_id_llm(
        client=AsyncMock(),
        message="I need an update on order ABC123 please.",
        recent_order_ids=["ABC123", "XYZ789"],
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert order_id == "ABC123"


@pytest.mark.asyncio
async def test_extract_product_identifier(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        assert len(args) > 0
        assert 'messages' in kwargs
        assert "You extract product names or SKUs" in kwargs['messages'][0]['content']
        return _FakeCompletion("Yoga Mat")

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    product = await nlp_mod.extract_product_id(
        client=AsyncMock(),
        message="Is the Yoga Mat latex-free?",
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert product == "Yoga Mat"


@pytest.mark.asyncio
async def test_analyze_sentiment(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        assert len(args) > 0
        assert 'messages' in kwargs
        assert "You classify text sentiment" in kwargs['messages'][0]['content']
        return _FakeCompletion("positive")

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    sentiment = await nlp_mod.sentiment_analysis(
        client=AsyncMock(),
        message="Great service!",
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert sentiment == "positive"
