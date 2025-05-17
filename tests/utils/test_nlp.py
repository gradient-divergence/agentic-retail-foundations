import builtins
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        assert "messages" in kwargs
        # Check if the system prompt is correct
        assert "You are an intent classification system" in kwargs["messages"][0]["content"]
        return _FakeCompletion("order_status")

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    # Mock isinstance temporarily to allow AsyncMock to pass the check
    def mock_isinstance(obj, classinfo):
        # If checking for AsyncOpenAI, return True if obj is our AsyncMock
        if classinfo is nlp_mod.AsyncOpenAI and isinstance(obj, AsyncMock):
            return True
        # Otherwise, fallback to the real isinstance
        return builtins.isinstance(obj, classinfo)

    with patch("utils.nlp.isinstance", mock_isinstance):
        # Use the new signature
        intent = await nlp_mod.classify_intent(
            client=AsyncMock(),
            message="Where is my package?",
        )
    assert intent == "order_status"


@pytest.mark.parametrize(
    "llm_response_content, expected_intent, expect_warning_log, expected_log_substring",
    [
        (None, "unknown", False, None),
        ("", "unknown", True, "LLM completion failed or gave empty content."),
        (
            "   ",
            "unknown",
            True,
            "LLM returned unexpected intent:",
        ),  # Strips to empty, then fails validation
        (
            "cancel_order",
            "unknown",
            True,
            "LLM returned unexpected intent:",
        ),  # Unexpected intent
    ],
)
@pytest.mark.asyncio
async def test_classify_intent_edge_cases(
    monkeypatch,
    caplog,
    llm_response_content,
    expected_intent,
    expect_warning_log,
    expected_log_substring,
):
    """Test classify_intent with None or unexpected LLM responses."""

    async def fake_safe_chat_completion(*args, **kwargs):
        if llm_response_content is None:
            return None  # Simulate completion failure
        return _FakeCompletion(llm_response_content)

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    # Mock isinstance as in the original test
    def mock_isinstance(obj, classinfo):
        if classinfo is nlp_mod.AsyncOpenAI and isinstance(obj, AsyncMock):
            return True
        return builtins.isinstance(obj, classinfo)

    with (
        patch("utils.nlp.isinstance", mock_isinstance),
        caplog.at_level(logging.WARNING),
    ):
        intent = await nlp_mod.classify_intent(
            client=AsyncMock(),
            message="Some message",
        )

    assert intent == expected_intent

    log_actually_present = False
    if expect_warning_log:
        if expected_log_substring:
            log_actually_present = expected_log_substring in caplog.text
        else:
            # If we expect a generic warning but no specific substring,
            # check if any warning occurred.
            # This branch might need refinement based on actual desired behavior
            # for generic warnings.
            log_actually_present = any(record.levelno == logging.WARNING for record in caplog.records)
    else:
        # If we don't expect a log, ensure no warnings related to intent
        # classification are present.
        # This is a bit broad; ideally, we'd ensure *no* warnings if that's the
        # strict expectation.
        unexpected_log_present = any(
            sub in caplog.text
            for sub in [
                "LLM completion failed or gave empty content.",
                "LLM returned unexpected intent:",
            ]
            if sub
        )
        log_actually_present = not unexpected_log_present

    assert log_actually_present == expect_warning_log, (
        f"Log check failed. Expected warning? {expect_warning_log}. Expected substring: '{expected_log_substring}'. Actual logs: {caplog.text!r}"
    )


@pytest.mark.asyncio
async def test_extract_order_id(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        assert len(args) > 0
        assert "messages" in kwargs
        # Check system prompt related to order IDs
        assert "You extract order IDs" in kwargs["messages"][0]["content"]
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


@pytest.mark.parametrize(
    "llm_response_content, expected_order_id",
    [
        (None, None),
        ("", None),  # Empty string after strip should result in None
        ("   ", None),  # Whitespace strips to empty, should result in None
        ("none", None),
        ("None", None),
        ("ambiguous", None),
        ("not_recent", None),
        ("not_found", None),
    ],
)
@pytest.mark.asyncio
async def test_extract_order_id_llm_edge_cases(monkeypatch, llm_response_content, expected_order_id):
    """Test extract_order_id_llm with None or explicit non-ID LLM responses."""

    async def fake_safe_chat_completion(*args, **kwargs):
        if llm_response_content is None:
            return None
        return _FakeCompletion(llm_response_content)

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    order_id = await nlp_mod.extract_order_id_llm(
        client=AsyncMock(),
        message="Check order status",
        recent_order_ids=["XYZ789"],  # Provide some context
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert order_id == expected_order_id


@pytest.mark.asyncio
async def test_extract_product_identifier(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        assert len(args) > 0
        assert "messages" in kwargs
        assert "You extract product names or SKUs" in kwargs["messages"][0]["content"]
        return _FakeCompletion("Yoga Mat")

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    product = await nlp_mod.extract_product_id(
        client=AsyncMock(),
        message="Is the Yoga Mat latex-free?",
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert product == "Yoga Mat"


@pytest.mark.parametrize(
    "llm_response_content, expected_product_id",
    [
        (None, None),
        ("", None),  # Empty string after strip should result in None
        ("   ", None),  # Whitespace strips to empty, should result in None
        ("none", None),
        ("None", None),
        (
            ("I couldn't find a specific product mentioned in your query, could you please clarify?"),
            None,
        ),
    ],
)
@pytest.mark.asyncio
async def test_extract_product_id_edge_cases(monkeypatch, llm_response_content, expected_product_id):
    """Test extract_product_id with None or non-ID LLM responses."""

    async def fake_safe_chat_completion(*args, **kwargs):
        if llm_response_content is None:
            return None
        return _FakeCompletion(llm_response_content)

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    product_id = await nlp_mod.extract_product_id(
        client=AsyncMock(),
        message="Is it available?",
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert product_id == expected_product_id


@pytest.mark.asyncio
async def test_analyze_sentiment(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        assert len(args) > 0
        assert "messages" in kwargs
        assert "You classify text sentiment" in kwargs["messages"][0]["content"]
        return _FakeCompletion("positive")

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    sentiment = await nlp_mod.sentiment_analysis(
        client=AsyncMock(),
        message="Great service!",
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert sentiment == "positive"


@pytest.mark.parametrize(
    "llm_response_content, expected_sentiment, log_level, log_message",
    [
        (None, "neutral", None, None),  # Simulate safe_chat_completion returning None
        ("", "neutral", None, None),  # Simulate empty content
        ("   ", "neutral", None, None),  # Simulate whitespace content
        (
            "mostly_positive",
            "neutral",
            logging.WARNING,
            "LLM returned unexpected sentiment: mostly_positive",
        ),  # Simulate unexpected sentiment
        (
            "positive but maybe neutral",
            "neutral",
            logging.WARNING,
            "LLM returned unexpected sentiment: positive but maybe neutral",
        ),  # Simulate conversational junk
    ],
)
@pytest.mark.asyncio
async def test_analyze_sentiment_edge_cases(
    monkeypatch,
    caplog,
    llm_response_content,
    expected_sentiment,
    log_level,
    log_message,
):
    """Test sentiment_analysis with None or unexpected LLM responses."""

    async def fake_safe_chat_completion(*args, **kwargs):
        if llm_response_content is None:
            return None
        return _FakeCompletion(llm_response_content)

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    with caplog.at_level(logging.WARNING):
        sentiment = await nlp_mod.sentiment_analysis(
            client=AsyncMock(),
            message="Some feedback",
            model="dummy",
            logger=logging.getLogger(__name__),
        )

    assert sentiment == expected_sentiment
    if log_level is not None and log_message is not None:
        assert log_message in caplog.text
        # Check the last record for the warning
        assert caplog.records[-1].levelno == log_level
