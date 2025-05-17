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


@pytest.mark.parametrize(
    "llm_response_content, expected_intent, expect_warning_log",
    [
        (None, "unknown", False),
        ("", "unknown", True), # Empty string fails validation -> unknown, warning
        ("   ", "unknown", True), # Whitespace strips empty -> unknown, warning
        ("cancel_order", "unknown", True), # Unexpected intent -> unknown, WITH warning
    ],
)
@pytest.mark.asyncio
async def test_classify_intent_edge_cases(
    monkeypatch, caplog, llm_response_content, expected_intent, expect_warning_log
):
    """Test classify_intent with None or unexpected LLM responses."""
    async def fake_safe_chat_completion(*args, **kwargs):
        if llm_response_content is None:
            return None # Simulate completion failure
        return _FakeCompletion(llm_response_content)

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    # Mock isinstance as in the original test
    def mock_isinstance(obj, classinfo):
        if classinfo is nlp_mod.AsyncOpenAI and isinstance(obj, AsyncMock):
            return True
        return builtins.isinstance(obj, classinfo)

    with patch('utils.nlp.isinstance', mock_isinstance), caplog.at_level(logging.WARNING):
        intent = await nlp_mod.classify_intent(
            client=AsyncMock(),
            message="Some message",
        )

    assert intent == expected_intent
    # Adjust log check based on the specific warning expected
    log_present = False
    expected_log_substring = None
    if llm_response_content is not None:
        # Handle the specific test cases explicitly
        if llm_response_content == "": # Empty string
            expected_log_substring = "LLM completion failed or gave empty content."
        elif llm_response_content == "   ": # Whitespace only
             expected_log_substring = "LLM returned unexpected intent:"
        elif llm_response_content == "cancel_order": # Non-empty, unexpected
             expected_log_substring = "LLM returned unexpected intent:"
        # Add other specific unexpected content checks here if needed

    if expect_warning_log and expected_log_substring:
        log_present = expected_log_substring in caplog.text
    elif expect_warning_log and not expected_log_substring:
        # This case shouldn't happen if parametrize is correct, but indicates a mismatch
        log_present = False
    elif not expect_warning_log:
        # If we don't expect a log, ensure *none* of the warning substrings are present
        log_present = not any(sub in caplog.text for sub in [
            "LLM completion failed or gave empty content.",
            "LLM returned unexpected intent:"
        ] if sub)
    else: # Should not happen
        log_present = False

    assert log_present == expect_warning_log, \
           f"Log check failed. Expected log? {expect_warning_log}. Expected substring: '{expected_log_substring}'. Actual logs: {caplog.text!r}"


@pytest.mark.asyncio
async def test_extract_order_id(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        assert len(args) > 0
        assert 'messages' in kwargs
        # Check system prompt related to order IDs
        assert "You extract order IDs" in kwargs['messages'][0]['content']
        return _FakeCompletion("ABC123")

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    # Mock isinstance to treat AsyncMock as AsyncOpenAI
    def mock_isinstance(obj, classinfo):
        if classinfo is nlp_mod.AsyncOpenAI and isinstance(obj, AsyncMock):
            return True
        return builtins.isinstance(obj, classinfo)

    with patch("utils.nlp.isinstance", mock_isinstance):
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
        ("", None),    # Empty string after strip should result in None
        ("   ", None), # Whitespace strips to empty, should result in None
        ("none", None),
        ("None", None),
        ("ambiguous", None),
        ("not_recent", None),
        ("not_found", None),
    ],
)
@pytest.mark.asyncio
async def test_extract_order_id_llm_edge_cases(
    monkeypatch, llm_response_content, expected_order_id
):
    """Test extract_order_id_llm with None or explicit non-ID LLM responses."""
    async def fake_safe_chat_completion(*args, **kwargs):
        if llm_response_content is None:
            return None
        return _FakeCompletion(llm_response_content)

    monkeypatch.setattr(nlp_mod, "safe_chat_completion", fake_safe_chat_completion)

    def mock_isinstance(obj, classinfo):
        if classinfo is nlp_mod.AsyncOpenAI and isinstance(obj, AsyncMock):
            return True
        return builtins.isinstance(obj, classinfo)

    with patch("utils.nlp.isinstance", mock_isinstance):
        order_id = await nlp_mod.extract_order_id_llm(
            client=AsyncMock(),
            message="Check order status",
            recent_order_ids=["XYZ789"],  # Provide some context
            model="dummy",
            logger=logging.getLogger(__name__),
        )
    assert order_id == expected_order_id


@pytest.mark.asyncio
async def test_extract_order_id_llm_invalid_client(monkeypatch):
    """Ensure TypeError is raised when client is not AsyncOpenAI."""
    with pytest.raises(TypeError):
        await nlp_mod.extract_order_id_llm(
            client=MagicMock(),
            message="Check order",
            recent_order_ids=["XYZ789"],
            model="dummy",
            logger=logging.getLogger(__name__),
        )


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


@pytest.mark.parametrize(
    "llm_response_content, expected_product_id",
    [
        (None, None),
        ("", None),    # Empty string after strip should result in None
        ("   ", None), # Whitespace strips to empty, should result in None
        ("none", None),
        ("None", None),
        ("I couldn't find a specific product mentioned in your query, could you please clarify?", None),
    ],
)
@pytest.mark.asyncio
async def test_extract_product_id_edge_cases(
    monkeypatch, llm_response_content, expected_product_id
):
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


@pytest.mark.parametrize(
    "llm_response_content, expected_sentiment, log_level, log_message",
    [
        (None, "neutral", None, None), # Simulate safe_chat_completion returning None
        ("", "neutral", None, None),    # Simulate empty content
        ("   ", "neutral", None, None), # Simulate whitespace content
        ("mostly_positive", "neutral", logging.WARNING, "LLM returned unexpected sentiment: mostly_positive"), # Simulate unexpected sentiment
        ("positive but maybe neutral", "neutral", logging.WARNING, "LLM returned unexpected sentiment: positive but maybe neutral"), # Simulate conversational junk
    ],
)
@pytest.mark.asyncio
async def test_analyze_sentiment_edge_cases(
    monkeypatch, caplog, llm_response_content, expected_sentiment, log_level, log_message
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
