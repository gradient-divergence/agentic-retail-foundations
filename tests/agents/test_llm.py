import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.llm import RetailCustomerServiceAgent
from tests.mocks import MockProductDB, MockOrderSystem, MockCustomerDB


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
def mock_nlp_module(monkeypatch):
    mock_nlp = MagicMock()
    # Configure return values for the nlp functions called by the agent
    mock_nlp.classify_intent = AsyncMock(return_value="order_status")
    mock_nlp.extract_order_id_llm = AsyncMock(return_value="ORD123")
    mock_nlp.extract_product_id = AsyncMock(return_value="Product ABC")
    mock_nlp.sentiment_analysis = AsyncMock(return_value="neutral")
    monkeypatch.setattr("agents.llm.nlp", mock_nlp) # Patch where it's imported
    return mock_nlp


@pytest.fixture
def agent(mock_connectors):
    # Use mocks for dependencies
    _, _, _, policies = mock_connectors # Unpack to get the same policies dict
    return RetailCustomerServiceAgent(
        product_database=MockProductDB(),
        order_management_system=MockOrderSystem(),
        customer_database=MockCustomerDB(),
        policy_guidelines=policies, # Use the consistent policy dict
        api_key="dummy_key" # Provide a dummy key to attempt client init
    )


@pytest.mark.asyncio
async def test_classify_intent(agent, mock_nlp_module):
    """Test intent classification delegation."""
    message = "Where is my order?"
    # We mock the nlp module itself, so we just need to ensure classify_intent is called
    mock_nlp_module.classify_intent.return_value = "order_status" # Ensure mock returns expected

    # Call the agent's method which *uses* the nlp function
    intent = await agent._classify_intent(message)

    # Assert that the agent's method called the mocked nlp function correctly
    mock_nlp_module.classify_intent.assert_awaited_once_with(
        client=agent.client, # Check that the agent passes its client
        message=message # Check that the agent passes the message
        # No model or other args expected here anymore
    )
    assert intent == "order_status"


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


@pytest.mark.asyncio
async def test_extract_order_id_logic(agent, mock_nlp_module):
    """Test order ID extraction logic combines regex and LLM call."""
    message_with_id = "My order is #ORD123, what's the status?"
    recent_orders = [{"order_id": "ORD123"}, {"order_id": "ORD456"}]
    recent_order_ids_list = ["ORD123", "ORD456"]

    # Setup mock return value for the LLM call
    mock_nlp_module.extract_order_id_llm.return_value = "ORD123"

    extracted_id = await agent._extract_order_id(message_with_id, recent_orders)

    # Regex should find ORD123 and return it because it matches recent orders
    assert extracted_id == "ORD123"
    # Ensure the LLM function was NOT called because regex found a match in recent orders
    mock_nlp_module.extract_order_id_llm.assert_not_awaited()

    # --- Test case where regex finds ID but it's not recent ---
    mock_nlp_module.extract_order_id_llm.reset_mock()
    message_not_recent = "Check on order REGEX999?"
    # LLM will be called, configure its return value
    mock_nlp_module.extract_order_id_llm.return_value = "REGEX999" # Simulate LLM confirming

    extracted_id = await agent._extract_order_id(message_not_recent, recent_orders)

    # Assert LLM was called with correct arguments
    mock_nlp_module.extract_order_id_llm.assert_awaited_once_with(
        client=agent.client,
        message=message_not_recent,
        recent_order_ids=recent_order_ids_list,
        model=agent.utility_model,
        logger=agent.logger,
        retry_attempts=agent.retry_attempts,
        retry_backoff=agent.retry_backoff
    )
    # Assert the result from LLM is returned
    assert extracted_id == "REGEX999"

    # --- Test case where only LLM can find it ---
    mock_nlp_module.extract_order_id_llm.reset_mock()
    message_llm_only = "Where is my recent delivery?"
    mock_nlp_module.extract_order_id_llm.return_value = "ORD456"

    extracted_id = await agent._extract_order_id(message_llm_only, recent_orders)
    mock_nlp_module.extract_order_id_llm.assert_awaited_once_with(
        client=agent.client,
        message=message_llm_only,
        recent_order_ids=recent_order_ids_list,
        model=agent.utility_model,
        logger=agent.logger,
        retry_attempts=agent.retry_attempts,
        retry_backoff=agent.retry_backoff
    )
    assert extracted_id == "ORD456"
