from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import components that will be mocked or used
from agents.llm import RetailCustomerServiceAgent


@pytest.fixture
def mock_connectors():
    product_db = MagicMock()
    order_system = MagicMock()
    customer_db = MagicMock()
    policies = {"returns": {"return_window_days": 30, "return_methods": ["mail", "in-store"]}}
    return product_db, order_system, customer_db, policies


@pytest.fixture
def mock_nlp_module(monkeypatch):
    mock_nlp = MagicMock()
    # Configure return values for the nlp functions called by the agent
    mock_nlp.classify_intent = AsyncMock(return_value="order_status")
    mock_nlp.extract_order_id_llm = AsyncMock(return_value="ORD123")
    mock_nlp.extract_product_id = AsyncMock(return_value="Product ABC")
    mock_nlp.sentiment_analysis = AsyncMock(return_value="neutral")
    monkeypatch.setattr("agents.llm.nlp", mock_nlp)  # Patch where it's imported
    return mock_nlp


@pytest.fixture
def agent(mock_connectors):
    # Use mocks for dependencies
    product_db, order_system, customer_db, policies = mock_connectors
    return RetailCustomerServiceAgent(
        product_database=product_db,  # Use the MagicMock instances
        order_management_system=order_system,
        customer_database=customer_db,
        policy_guidelines=policies,  # Use the consistent policy dict
        api_key="dummy_key",  # Provide a dummy key to attempt client init
    )


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


@pytest.mark.asyncio
@patch("agents.llm.nlp.classify_intent", new_callable=AsyncMock)
@patch("agents.llm.nlp.extract_order_id_llm", new_callable=AsyncMock)
@patch("agents.llm.nlp.extract_product_id", new_callable=AsyncMock)
@patch("agents.llm.nlp.sentiment_analysis", new_callable=AsyncMock)
@patch("agents.llm.build_response_prompt")
@patch("agents.llm.generate_agent_response", new_callable=AsyncMock)
@patch("agents.llm.extract_actions", new_callable=AsyncMock)
async def test_process_customer_inquiry_order_status(
    mock_llm_extract_actions: AsyncMock,
    mock_llm_generate_response: AsyncMock,
    mock_llm_build_prompt: MagicMock,
    mock_nlp_sentiment: AsyncMock,
    mock_nlp_extract_product_id: AsyncMock,
    mock_nlp_extract_oid: AsyncMock,
    mock_classify_intent: AsyncMock,
    agent: RetailCustomerServiceAgent,
    mock_connectors: tuple,
):
    """Test the full flow for an order_status inquiry."""
    customer_id = "C123"
    # FIX: Use a message where regex won't find the ID, forcing LLM fallback
    message = "Where is my recent package?"
    test_order_id = "ORD123"

    # --- Mock Setup --- #
    # 1. Mock NLP functions
    mock_classify_intent.return_value = "order_status"
    # Assume regex fails, LLM extracts the ID
    mock_nlp_extract_oid.return_value = test_order_id
    mock_nlp_sentiment.return_value = "neutral"

    # 2. Mock Connectors and ContextRetriever
    mock_connectors[2].get_customer = AsyncMock(return_value={"name": "Alice", "loyalty_tier": "Gold"})
    mock_connectors[1].get_recent_orders = AsyncMock(return_value=[{"order_id": test_order_id, "order_date": "2024-07-01", "status": "Shipped"}])
    mock_order_details = {
        "order_id": test_order_id,
        "status": "Shipped",
        "items": [{"name": "Test Item"}],
        "est_delivery": "2024-07-10",
        "tracking": "TRACK123",
    }
    # get_order_details is part of order_system, which is part of context_retriever
    # No direct call from agent, but context_retriever.get_context will use it.
    # So, ensuring mock_connectors[1] (order_system mock) has get_order_details
    # is good if get_context is complex.
    # For this test, mocking get_context directly is simpler.
    agent.context_retriever.get_context = AsyncMock(return_value={"order_details": mock_order_details})

    # 3. Mock LLM Component calls
    mock_llm_build_prompt.return_value = "Mock system prompt"
    generated_agent_message = f"Your order {test_order_id} shipped and is due 2024-07-10."
    mock_llm_generate_response.return_value = generated_agent_message
    mock_llm_extract_actions.return_value = [{"type": "provide_tracking", "tracking_number": "TRACK123"}]

    # 4. Mock ConversationManager methods
    # The lock is a real asyncio.Lock, but we can mock the manager methods that use it.
    # If the lock itself needs specific testing, it would be more involved.
    mock_cm_add_message = MagicMock()
    mock_cm_get_recent_history = MagicMock(return_value=[{"role": "customer", "content": message}])  # Simulate some history
    mock_cm_get_lock = MagicMock()
    # Create a context manager mock for the lock
    mock_lock_instance = AsyncMock()  # The lock itself
    mock_cm_get_lock.return_value = mock_lock_instance  # get_lock returns the mock lock

    agent.conversation_manager.add_message = mock_cm_add_message
    agent.conversation_manager.get_recent_history = mock_cm_get_recent_history
    agent.conversation_manager.get_lock = mock_cm_get_lock

    # 5. Mock log_interaction
    agent._log_interaction = AsyncMock()

    # --- Execute --- #
    response = await agent.process_customer_inquiry(customer_id, message)

    # --- Assertions --- #
    # Check dependencies were called
    mock_connectors[2].get_customer.assert_awaited_once_with(customer_id)
    mock_nlp_extract_oid.assert_awaited_once()
    mock_connectors[1].get_recent_orders.assert_awaited_once_with(customer_id, limit=3)
    agent.context_retriever.get_context.assert_awaited_once_with("order_status", {"order_id": test_order_id})

    # Check LLM component calls
    mock_llm_build_prompt.assert_called_once_with(
        customer_info={"name": "Alice", "loyalty_tier": "Gold"},
        intent="order_status",
        message=message,
        context_data={"order_details": mock_order_details},
        conversation_history=mock_cm_get_recent_history.return_value,
        # brand_name="ACME Retail" # Default, or mock if customized
    )
    mock_llm_generate_response.assert_awaited_once_with(
        client=agent.client,
        model=agent.response_model,
        system_prompt="Mock system prompt",
        logger=agent.logger,
        retry_attempts=agent.retry_attempts,
        retry_backoff=agent.retry_backoff,
    )
    mock_llm_extract_actions.assert_awaited_once_with(
        client=agent.client,
        intent="order_status",
        response_text=generated_agent_message,
        context_data={"order_details": mock_order_details},
        model=agent.utility_model,
        logger=agent.logger,
        retry_attempts=agent.retry_attempts,
        retry_backoff=agent.retry_backoff,
    )

    mock_nlp_sentiment.assert_awaited_once_with(
        client=agent.client,
        message=message,
        model=agent.utility_model,
        logger=agent.logger,
        retry_attempts=agent.retry_attempts,
        retry_backoff=agent.retry_backoff,
    )

    # Check logging
    agent._log_interaction.assert_awaited_once()

    # Check final response structure
    assert response["intent"] == "order_status"
    assert response["message"] == generated_agent_message
    assert response["actions"] == [{"type": "provide_tracking", "tracking_number": "TRACK123"}]
    assert response["customer_sentiment"] == "neutral"
    assert "error" not in response

    # Check conversation manager calls
    mock_cm_get_lock.assert_called_with(customer_id)
    # Check the lock was used (entered and exited)
    mock_lock_instance.__aenter__.assert_awaited_once()
    mock_lock_instance.__aexit__.assert_awaited_once()

    mock_cm_add_message.assert_any_call(customer_id, "customer", message)
    mock_cm_add_message.assert_any_call(customer_id, "agent", generated_agent_message)
    assert mock_cm_add_message.call_count == 2
    mock_cm_get_recent_history.assert_called_once_with(customer_id, n=5)

    # Verify the mock_nlp_module calls (already patched at module level for agent)
    # These are effectively testing the @patch decorators on the test function
    mock_classify_intent.assert_awaited_once_with(client=agent.client, message=message)
    # No, extract_order_id is called directly by agent, not via the
    # mock_nlp_module fixture here
    # mock_nlp_module.extract_order_id_llm.assert_awaited_once() # This would be
    # for the fixture
    # The mock_nlp_extract_oid parameter is the one to check
    mock_nlp_extract_oid.assert_awaited_once_with(
        client=agent.client,
        message=message,
        recent_order_ids=[test_order_id],  # From mock_connectors[1].get_recent_orders
        model=agent.utility_model,
        logger=agent.logger,
        retry_attempts=agent.retry_attempts,
        retry_backoff=agent.retry_backoff,
    )
