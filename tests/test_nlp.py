import logging
from types import SimpleNamespace

import pytest

from utils import nlp as nlp_mod

# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------


class _FakeCompletion:
    """Mimics the minimal shape of OpenAI SDK completion objects."""

    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_intent(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        return _FakeCompletion("order_status")

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    intent = await nlp_mod.classify_intent(
        client=None,
        message="Where is my package?",
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert intent == "order_status"


@pytest.mark.asyncio
async def test_extract_order_id(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        return _FakeCompletion("ABC123")

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    order_id = await nlp_mod.extract_order_id_via_llm(
        client=None,
        message="I need an update on order ABC123 please.",
        recent_order_ids=["ABC123", "XYZ789"],
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert order_id == "ABC123"


@pytest.mark.asyncio
async def test_extract_product_identifier(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        return _FakeCompletion("Yoga Mat")

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    product = await nlp_mod.extract_product_identifier(
        client=None,
        message="Is the Yoga Mat latex-free?",
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert product == "Yoga Mat"


@pytest.mark.asyncio
async def test_analyze_sentiment(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        return _FakeCompletion("positive")

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    sentiment = await nlp_mod.analyze_sentiment(
        client=None,
        message="Great service!",
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert sentiment == "positive"
