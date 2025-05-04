import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.llm import RetailCustomerServiceAgent
from openai import OpenAI
from utils.openai_utils import safe_chat_completion
from agents.response_builder import extract_actions


@pytest.fixture
def mock_connectors():
    product_db = MagicMock()
    order_system = MagicMock()
    customer_db = MagicMock()
    policies = {
        "returns": {"return_window_days": 30, "return_methods": ["mail", "in-store"]}
    }
    return product_db, order_system, customer_db, policies


@pytest.fixture
def agent(mock_connectors):
    product_db, order_system, customer_db, policies = mock_connectors
    with patch("agents.llm.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        return RetailCustomerServiceAgent(
            product_db, order_system, customer_db, policies, api_key="test-key"
        )


@pytest.mark.asyncio
@patch('utils.nlp.safe_chat_completion', new_callable=AsyncMock)
async def test_classify_intent(mock_safe_completion, agent):
    """Test intent classification for various messages."""
    # Mock the completion object structure expected by _classify_intent (via nlp_classify_intent)
    mock_response = MagicMock()
    mock_response.choices[0].message.content.strip.return_value = "order_status"
    mock_safe_completion.return_value = mock_response

    intent = await agent._classify_intent("Where is my order?")
    assert intent == "order_status"
    mock_safe_completion.assert_awaited_once()


@pytest.mark.asyncio
@patch('utils.nlp.safe_chat_completion', new_callable=AsyncMock)
async def test_extract_order_id_regex(mock_safe_completion, agent):
    """Test order ID extraction via regex."""
    recent_orders = [{"order_id": "ABC123"}]
    result = await agent._extract_order_id("My order #ABC123", recent_orders)
    assert result == "ABC123"


@pytest.mark.asyncio
@patch('utils.nlp.safe_chat_completion', new_callable=AsyncMock)
async def test_extract_order_id_llm(mock_safe_completion, agent):
    """Test order ID extraction via LLM fallback."""
    # Mock the completion object structure expected by _extract_order_id (via nlp_extract_order_id_llm)
    mock_response = MagicMock()
    mock_response.choices[0].message.content.strip.return_value = "ABC123"
    mock_safe_completion.return_value = mock_response

    recent_orders = [{"order_id": "ABC123"}]
    result = await agent._extract_order_id("Where is my recent order?", recent_orders)
    assert result == "ABC123"
    mock_safe_completion.assert_awaited_once()


@pytest.mark.asyncio
@patch('utils.nlp.safe_chat_completion', new_callable=AsyncMock)
async def test_extract_product_identifier(mock_safe_completion, agent):
    """Test product identifier extraction via LLM."""
    # Mock the completion object structure
    mock_response = MagicMock()
    mock_response.choices[0].message.content.strip.return_value = "Yoga Mat"
    mock_safe_completion.return_value = mock_response

    result = await agent._extract_product_identifier("Is the Yoga Mat latex-free?")
    assert result == "Yoga Mat"
    mock_safe_completion.assert_awaited_once()


@pytest.mark.asyncio
@patch('agents.llm.extract_actions', new_callable=AsyncMock)
@patch('agents.llm.safe_chat_completion', new_callable=AsyncMock)
async def test_generate_response(mock_safe_completion, mock_extract_actions, agent, mock_connectors):
    """Test response generation with context."""
    product_db, order_system, customer_db, policies = mock_connectors

    # --- Mock the LLM calls --- #

    # 1. Mock the *first* call to safe_chat_completion (message generation)
    mock_message_response = MagicMock()
    mock_message_response.choices[0].message.content.strip.return_value = "Your order is on the way."
    mock_safe_completion.return_value = mock_message_response # Only one call expected now

    # 2. Mock the return value of the extract_actions helper directly
    mock_extract_actions.return_value = [{"type": "escalate_issue"}]

    # Mock sentiment analysis separately
    agent._analyze_sentiment = AsyncMock(return_value="neutral")

    # --- Test Setup --- #
    customer_info = {"name": "Alice", "loyalty_tier": "Gold"}
    context_data = {
        "order_details": {
            "order_id": "ABC123",
            "status": "shipped",
            "items": [],
            "estimated_delivery": "2024-07-01",
            "tracking_number": "TRACK123",
        }
    }
    conversation_history = [{"role": "customer", "content": "Where is my order?"}]

    # --- Execute --- #
    response = await agent._generate_response(
        customer_info,
        "order_status",
        "Where is my order?",
        context_data,
        conversation_history,
    )

    # --- Assert --- #
    # Check that the mocks were called as expected
    mock_safe_completion.assert_awaited_once() # Called once for message
    mock_extract_actions.assert_awaited_once() # Called once for actions
    agent._analyze_sentiment.assert_awaited_once()

    # Check the final response structure
    assert response.get("message") == "Your order is on the way."
    assert response.get("actions") == [{"type": "escalate_issue"}]
    assert response.get("customer_sentiment") == "neutral"


@pytest.mark.asyncio
@patch('utils.nlp.safe_chat_completion', new_callable=AsyncMock)
async def test_analyze_sentiment(mock_safe_completion, agent):
    """Test sentiment analysis via LLM."""
    # Mock the completion object structure
    mock_response = MagicMock()
    mock_response.choices[0].message.content.strip.return_value = "positive"
    mock_safe_completion.return_value = mock_response

    sentiment = await agent._analyze_sentiment("Thank you so much!")
    assert sentiment == "positive"
    mock_safe_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_log_interaction(agent, caplog):
    """Test that _log_interaction logs expected info."""
    with caplog.at_level("INFO"):
        await agent._log_interaction(
            "cust1",
            "order_status",
            "Where is my order?",
            {"message": "On the way", "actions": [], "customer_sentiment": "neutral"},
        )
        assert any("Interaction logged" in r for r in caplog.text.splitlines())


@pytest.mark.asyncio
async def test_process_customer_inquiry_error(agent, mock_connectors):
    """Test process_customer_inquiry returns error if LLM unavailable."""
    agent.client = None
    response = await agent.process_customer_inquiry("cust1", "Where is my order?")
    assert response.get("intent") == "error"
    assert "unavailable" in response.get("message", "").lower()
    assert response.get("error") == "LLM client not available"
    assert response.get("actions") == []
    assert agent.policies == mock_connectors[3]
