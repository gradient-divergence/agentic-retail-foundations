# tests/agents/test_llm_components.py

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock, ANY
import logging
from openai import AsyncOpenAI # Import the real class for spec
from tests.mocks import MockAsyncOpenAI # Keep using this if it provides helpful defaults

# Functions to test
from agents.llm_components import (
    build_response_prompt,
    extract_actions,
    generate_agent_response
)

# --- Fixtures ---

@pytest.fixture
def sample_customer_info() -> dict:
    return {"name": "Test Customer", "loyalty_tier": "Gold"}

@pytest.fixture
def sample_context_data() -> dict:
    return {
        "order_details": {"order_id": "O123", "status": "Shipped", "items": [{"name": "Thingamajig"}]},
        "return_eligibility": {"eligible": True},
        "return_policy": {"return_window_days": 30, "return_methods": ["Mail"]}
    }

@pytest.fixture
def sample_history() -> list:
    return [
        {"role": "customer", "content": "Where is order O123?"},
        {"role": "agent", "content": "Order O123 has shipped."}
    ]

# --- Tests for build_response_prompt ---

def test_build_response_prompt_order_status(sample_customer_info, sample_context_data, sample_history):
    prompt = build_response_prompt(
        customer_info=sample_customer_info,
        intent="order_status",
        message="Status update?",
        context_data=sample_context_data,
        conversation_history=sample_history
    )
    assert "You are a helpful and friendly customer service agent" in prompt
    assert "CUSTOMER INFORMATION:" in prompt
    assert "- Name: Test Customer" in prompt
    assert "- Loyalty tier: Gold" in prompt
    assert "RECENT CONVERSATION HISTORY" in prompt
    assert "Customer: Where is order O123?" in prompt
    assert "Agent: Order O123 has shipped." in prompt
    assert "CURRENT CUSTOMER MESSAGE" in prompt
    assert '"Status update?"' in prompt
    assert "RELEVANT CONTEXT FOR RESPONSE" in prompt
    assert "- Order ID: O123" in prompt
    assert "- Status: Shipped" in prompt
    assert "- Items: Thingamajig" in prompt
    assert "INSTRUCTIONS FOR RESPONSE:" in prompt
    assert "Address the customer as Test Customer" in prompt
    assert "acknowledge their status positively" in prompt # Gold tier

def test_build_response_prompt_no_history(sample_customer_info, sample_context_data):
    prompt = build_response_prompt(
        customer_info=sample_customer_info,
        intent="product_question",
        message="Info on X?",
        context_data=sample_context_data,
        conversation_history=[]
    )
    assert "RECENT CONVERSATION HISTORY" not in prompt

def test_build_response_prompt_no_context(sample_customer_info, sample_history):
    prompt = build_response_prompt(
        customer_info=sample_customer_info,
        intent="general_inquiry",
        message="Hi",
        context_data={},
        conversation_history=sample_history
    )
    assert "RELEVANT CONTEXT FOR RESPONSE" in prompt
    assert "No specific order or product context retrieved" in prompt

def test_build_response_prompt_standard_loyalty(sample_context_data, sample_history):
    prompt = build_response_prompt(
        customer_info={"name": "Std Cust", "loyalty_tier": "Standard"},
        intent="order_status",
        message="Status?",
        context_data=sample_context_data,
        conversation_history=sample_history
    )
    assert "acknowledge their status positively" not in prompt

# --- Tests for generate_agent_response ---

@pytest.mark.asyncio
@patch('agents.llm_components.safe_chat_completion', new_callable=AsyncMock)
async def test_generate_agent_response_success(mock_safe_completion):
    # Create a mock client that passes the isinstance check
    mock_client = MagicMock(spec=AsyncOpenAI)
    # If MockAsyncOpenAI provides other useful defaults, try combining:
    # mock_client = MockAsyncOpenAI()
    # mock_client.__class__ = AsyncOpenAI # Or mock_client = MagicMock(spec=AsyncOpenAI, wraps=MockAsyncOpenAI())

    mock_response = MagicMock()
    mock_response.choices[0].message.content = " This is the generated response.  "
    mock_safe_completion.return_value = mock_response

    response = await generate_agent_response(
        client=mock_client, # Pass the spec'd mock
        model="test-model",
        system_prompt="Test prompt",
        logger=logging.getLogger("test")
    )

    mock_safe_completion.assert_awaited_once_with(
        mock_client,
        model="test-model",
        messages=[{"role": "system", "content": "Test prompt"}],
        logger=ANY,
        retry_attempts=ANY,
        retry_backoff=ANY,
        max_tokens=ANY,
        temperature=ANY,
        stop=None
    )
    assert response == "This is the generated response."

@pytest.mark.asyncio
@patch('agents.llm_components.safe_chat_completion', new_callable=AsyncMock)
async def test_generate_agent_response_failure(mock_safe_completion):
    # Use MagicMock with spec here too for consistency
    mock_client = MagicMock(spec=AsyncOpenAI)
    mock_safe_completion.side_effect = Exception("LLM Error")

    response = await generate_agent_response(
        client=mock_client,
        model="test-model",
        system_prompt="Test prompt",
        logger=logging.getLogger("test")
    )
    assert response == "" # Expect empty string on failure

@pytest.mark.asyncio
async def test_generate_agent_response_no_client():
    response = await generate_agent_response(
        client=None, # type: ignore # Test with None client
        model="test-model",
        system_prompt="Test prompt",
        logger=logging.getLogger("test")
    )
    assert response == ""

# --- Tests for extract_actions ---

@pytest.mark.asyncio
@patch('agents.llm_components.safe_chat_completion', new_callable=AsyncMock)
async def test_extract_actions_rule_based_return(mock_safe_completion):
    mock_client = MockAsyncOpenAI()
    context = {
        "return_eligibility": {"eligible": True},
        "order_details": {"order_id": "RET123"}
    }
    actions = await extract_actions(
        client=mock_client,
        intent="return_request",
        response_text="OK, you can return order RET123.",
        context_data=context,
        model="test-utility",
        logger=logging.getLogger("test")
    )
    assert actions == [{"type": "provide_return_instructions", "order_id": "RET123"}]
    mock_safe_completion.assert_awaited_once() # LLM is still called

@pytest.mark.asyncio
@patch('agents.llm_components.safe_chat_completion', new_callable=AsyncMock)
async def test_extract_actions_rule_based_escalate(mock_safe_completion):
    mock_client = MockAsyncOpenAI()
    context = {"order_details": {"order_id": "ESC456", "status": "delayed"}}
    actions = await extract_actions(
        client=mock_client,
        intent="order_status",
        response_text="Looks like order ESC456 is delayed.",
        context_data=context,
        model="test-utility",
        logger=logging.getLogger("test")
    )
    assert actions == [{"type": "escalate_issue", "reason": "delayed_order", "order_id": "ESC456"}]
    mock_safe_completion.assert_awaited_once()

@pytest.mark.asyncio
@patch('agents.llm_components.safe_chat_completion', new_callable=AsyncMock)
async def test_extract_actions_llm_extraction(mock_safe_completion):
    mock_client = MockAsyncOpenAI()
    mock_response = MagicMock()
    # Simulate LLM returning a JSON array string
    mock_response.choices[0].message.content = ' ["offer_discount", "log_feedback"] '
    mock_safe_completion.return_value = mock_response

    actions = await extract_actions(
        client=mock_client,
        intent="complaint", # No rule-based actions for this
        response_text="I am sorry, let me offer a discount.",
        context_data={},
        model="test-utility",
        logger=logging.getLogger("test")
    )
    assert actions == [{"type": "offer_discount"}, {"type": "log_feedback"}]
    mock_safe_completion.assert_awaited_once()

@pytest.mark.asyncio
@patch('agents.llm_components.safe_chat_completion', new_callable=AsyncMock)
async def test_extract_actions_llm_extraction_dict(mock_safe_completion):
    mock_client = MockAsyncOpenAI()
    mock_response = MagicMock()
    # Simulate LLM returning a JSON dict string
    mock_response.choices[0].message.content = ' { "actions": ["check_inventory"] } '
    mock_safe_completion.return_value = mock_response

    actions = await extract_actions(
        client=mock_client,
        intent="product_question",
        response_text="Let me check inventory.",
        context_data={},
        model="test-utility",
        logger=logging.getLogger("test")
    )
    assert actions == [{"type": "check_inventory"}]
    mock_safe_completion.assert_awaited_once()

@pytest.mark.asyncio
@patch('agents.llm_components.safe_chat_completion', new_callable=AsyncMock)
async def test_extract_actions_llm_invalid_json(mock_safe_completion):
    mock_client = MockAsyncOpenAI()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = ' ["offer_discount", '
    mock_safe_completion.return_value = mock_response

    actions = await extract_actions(
        client=mock_client,
        intent="complaint",
        response_text="Discount offered.",
        context_data={},
        model="test-utility",
        logger=logging.getLogger("test")
    )
    assert actions == [] # Should return empty on JSON error
    mock_safe_completion.assert_awaited_once()

@pytest.mark.asyncio
@patch('agents.llm_components.safe_chat_completion', new_callable=AsyncMock)
async def test_extract_actions_llm_error(mock_safe_completion):
    mock_client = MockAsyncOpenAI()
    mock_safe_completion.side_effect = Exception("LLM Action Error")

    actions = await extract_actions(
        client=mock_client,
        intent="complaint",
        response_text="Sorry about that.",
        context_data={},
        model="test-utility",
        logger=logging.getLogger("test")
    )
    assert actions == [] # Should return empty on LLM error
    mock_safe_completion.assert_awaited_once()

@pytest.mark.asyncio
@patch('agents.llm_components.safe_chat_completion', new_callable=AsyncMock)
async def test_extract_actions_combined(mock_safe_completion):
    """Test rule-based and LLM actions are combined correctly."""
    mock_client = MockAsyncOpenAI()
    context = {
        "return_eligibility": {"eligible": True},
        "order_details": {"order_id": "RET123"}
    }
    mock_response = MagicMock()
    # LLM suggests an additional action
    mock_response.choices[0].message.content = ' ["log_return_reason"] '
    mock_safe_completion.return_value = mock_response

    actions = await extract_actions(
        client=mock_client,
        intent="return_request",
        response_text="You can return RET123. We logged your reason.",
        context_data=context,
        model="test-utility",
        logger=logging.getLogger("test")
    )
    # Check both actions are present, order might vary depending on execution
    assert len(actions) == 2
    action_types = {a["type"] for a in actions}
    assert action_types == {"provide_return_instructions", "log_return_reason"}
    mock_safe_completion.assert_awaited_once() 