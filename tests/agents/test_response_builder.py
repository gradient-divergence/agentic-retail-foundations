import logging
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, patch

from agents import response_builder as rb

# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


@pytest.fixture
def customer_info_standard():
    return {"name": "Bob", "loyalty_tier": "Standard", "customer_since": "2023-01-01"}

@pytest.fixture
def customer_info_gold():
    return {"name": "Alice", "loyalty_tier": "Gold", "customer_since": "2022-05-10"}

@pytest.fixture
def conversation_history_empty():
    return []

@pytest.fixture
def conversation_history_present():
    return [
        {"role": "customer", "content": "Hi there"},
        {"role": "agent", "content": "Hello! How can I help?"},
    ]

@pytest.fixture
def context_order_status():
    return {
        "order_details": {
            "order_id": "ORD123", "status": "Shipped",
            "items": [{"name": "Widget"}], "estimated_delivery": "2024-03-10"
        }
    }

@pytest.fixture
def context_product_question():
    return {
        "product_details": {"name": "Super Widget", "price": 25.99},
        "inventory": {"status": "In Stock"}
    }

@pytest.fixture
def context_return_eligible():
    return {
        "return_eligibility": {"eligible": True},
        "return_policy": {"return_window_days": 30, "return_methods": ["mail"]}
    }

@pytest.fixture
def context_return_ineligible():
    return {
        "return_eligibility": {"eligible": False, "reason": "Outside window"},
        "return_policy": {"return_window_days": 30, "return_methods": ["mail"]}
    }

# ---------------------------------------------------------------------------
# Tests for build_response_prompt
# ---------------------------------------------------------------------------

def test_build_response_prompt_contains_sections(customer_info_standard):
    prompt = rb.build_response_prompt(
        customer_info=customer_info_standard,
        intent="general_inquiry",
        message="What are your store hours?",
        context_data={},
        conversation_history=[],
    )
    assert "CUSTOMER INFORMATION" in prompt
    assert "INSTRUCTIONS FOR RESPONSE" in prompt
    assert "RELEVANT CONTEXT" in prompt
    assert "RECENT CONVERSATION HISTORY" not in prompt

@pytest.mark.parametrize(
    "test_id, customer_info_fixture, intent, message, context_data_fixture, history_fixture, expected_strings, unexpected_strings",
    [
        # --- General Inquiry --- #
        (
            "general_standard_no_hist_no_ctx", "customer_info_standard", "general_inquiry", "Hours?",
            lambda: {}, "conversation_history_empty",
            ["Name: Bob", "Loyalty tier: Standard", "No specific order or product context", "Address the customer as Bob"],
            ["RECENT CONVERSATION HISTORY"]
        ),
        (
            "general_gold_hist_no_ctx", "customer_info_gold", "general_inquiry", "Hi",
            lambda: {}, "conversation_history_present",
            ["Name: Alice", "Loyalty tier: Gold", "As a valued Gold member", "RECENT CONVERSATION HISTORY", "Customer: Hi there", "Agent: Hello! How can I help?", "Address the customer as Alice"],
            []
        ),
        # --- Order Status --- #
        (
            "order_status_standard_hist_ctx", "customer_info_standard", "order_status", "Track ORD123",
            "context_order_status", "conversation_history_present",
            ["Status: Shipped", "Tracking: Not available", "RECENT CONVERSATION HISTORY"],
            ["No specific order or product context"]
        ),
        (
            "order_status_gold_no_hist_no_ctx", "customer_info_gold", "order_status", "Where order?",
            lambda: {}, "conversation_history_empty",
            ["Name: Alice", "Loyalty tier: Gold", "As a valued Gold member", "No specific order or product context"],
            ["RECENT CONVERSATION HISTORY", "Status: Shipped"]
        ),
        # --- Product Question --- #
        (
            "product_q_standard_no_hist_ctx", "customer_info_standard", "product_question", "Stock?",
            "context_product_question", "conversation_history_empty",
            ["Product Name: Super Widget", "Availability: In Stock"],
            ["RECENT CONVERSATION HISTORY", "No specific order or product context"]
        ),
        # --- Return Request --- #
        (
            "return_eligible_gold_hist_ctx", "customer_info_gold", "return_request", "Return it",
            "context_return_eligible", "conversation_history_present",
            ["Return Eligible: Yes", "Return Methods: mail", "start your return at acmeretail.com/returns", "RECENT CONVERSATION HISTORY", "As a valued Gold member"],
            ["No specific order or product context", "Reason Not Eligible:"]
        ),
        (
            "return_ineligible_standard_no_hist_ctx", "customer_info_standard", "return_request", "Return it",
            "context_return_ineligible", "conversation_history_empty",
            ["Return Eligible: No", "Reason Not Eligible: Outside window"],
            ["RECENT CONVERSATION HISTORY"]
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else ""
)
def test_build_response_prompt_content(
    request,
    test_id, customer_info_fixture, intent, message, context_data_fixture, history_fixture, expected_strings, unexpected_strings
):
    customer_info = request.getfixturevalue(customer_info_fixture)
    context_data = request.getfixturevalue(context_data_fixture) if isinstance(context_data_fixture, str) else context_data_fixture()
    conversation_history = request.getfixturevalue(history_fixture)

    prompt = rb.build_response_prompt(
        customer_info=customer_info,
        intent=intent,
        message=message,
        context_data=context_data,
        conversation_history=conversation_history,
    )

    for expected in expected_strings:
        assert expected in prompt, f"Expected string not found: '{expected}'"

    for unexpected in unexpected_strings:
        assert unexpected not in prompt, f"Unexpected string found: '{unexpected}'"


@pytest.mark.asyncio
async def test_extract_actions_rule_based(monkeypatch):
    async def fake_safe_chat_completion(*args, **kwargs):  # noqa: D401,E501
        return _FakeCompletion("[]")

    monkeypatch.setattr(rb, "safe_chat_completion", fake_safe_chat_completion)

    actions = await rb.extract_actions(
        client=AsyncMock(),
        intent="order_status",
        response_text="Your order is currently delayed. I will escalate this issue.",
        context_data={"order_details": {"order_id": "ABC123", "status": "delayed"}},
        model="dummy",
        logger=logging.getLogger(__name__),
    )
    assert {
        "type": "escalate_issue",
        "reason": "delayed_order",
        "order_id": "ABC123",
    } in actions

@pytest.mark.asyncio
async def test_extract_actions_rule_based_return(monkeypatch):
    """Test the provide_return_instructions rule-based action."""
    # Use patch with new_callable=AsyncMock for safe_chat_completion
    with patch('agents.response_builder.safe_chat_completion', new_callable=AsyncMock) as mock_safe_completion:
        mock_safe_completion.return_value = _FakeCompletion("[]") # LLM returns no actions

        actions = await rb.extract_actions(
            client=AsyncMock(),
            intent="return_request",
            response_text="Sure, you can return order ORD456.",
            context_data={
                "return_eligibility": {"eligible": True},
                "order_details": {"order_id": "ORD456"}
            },
            model="dummy",
            logger=logging.getLogger(__name__),
        )
        # Check if the rule-based action was added
        assert len(actions) == 1
        assert actions[0] == {"type": "provide_return_instructions", "order_id": "ORD456"}
        # Check LLM mock was still called
        mock_safe_completion.assert_awaited_once()

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_id, llm_response_str, rule_based_context, expected_actions",
    [
        # LLM only - valid list
        ("llm_only_list", '[ "action_A", "action_B" ]', {}, [{"type": "action_A"}, {"type": "action_B"}]),
        # LLM only - valid dict wrapper
        ("llm_only_dict", '{"actions": ["action_C"] }', {}, [{"type": "action_C"}]),
        # LLM only - invalid JSON
        ("llm_invalid_json", '[ "action_A", ', {}, []),
        # LLM only - empty list
        ("llm_empty_list", '[]', {}, []),
        # LLM only - not a list/dict
        ("llm_not_list", '"just_a_string"', {}, []),
        # Combined: Rule (escalate) + LLM (unique action)
        (
            "combined_rule_llm_unique", '[ "send_coupon" ]',
            {"order_details": {"order_id": "O1", "status": "lost"}},
            [{"type": "escalate_issue", "reason": "lost_order", "order_id": "O1"}, {"type": "send_coupon"}]
        ),
        # Combined: Rule (return) + LLM (duplicate action type ignored)
        (
            "combined_rule_llm_duplicate", '[ "provide_return_instructions" ]',
            {"return_eligibility": {"eligible": True}, "order_details": {"order_id": "O2"}},
            [{"type": "provide_return_instructions", "order_id": "O2"}] # LLM duplicate ignored
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "" # Use test_id
)
async def test_extract_actions_llm_and_combined(
    test_id: str, llm_response_str: str, rule_based_context: dict, expected_actions: list,
    caplog
):
    """Test LLM-based action extraction and combination with rules."""
    # Use patch with new_callable=AsyncMock for safe_chat_completion
    with patch('agents.response_builder.safe_chat_completion', new_callable=AsyncMock) as mock_safe_completion, \
         caplog.at_level(logging.WARNING):

        mock_safe_completion.return_value = _FakeCompletion(llm_response_str)

        # Determine intent based on rule_based_context for simplicity
        intent = "return_request" if "return_eligibility" in rule_based_context else \
                 "order_status" if "order_details" in rule_based_context else \
                 "general_inquiry"

        actions = await rb.extract_actions(
            client=AsyncMock(),
            intent=intent,
            response_text="Some agent response...",
            context_data=rule_based_context,
            model="dummy",
            logger=logging.getLogger(__name__),
        )

        # Use set comparison as order might not be guaranteed if types differ later
        assert len(actions) == len(expected_actions)
        # Convert to comparable format (e.g., tuples of sorted items)
        action_set = {tuple(sorted(a.items())) for a in actions}
        expected_set = {tuple(sorted(e.items())) for e in expected_actions}
        assert action_set == expected_set

        # Check for warnings on invalid LLM output
        if "invalid_json" in test_id or "not_list" in test_id:
            assert "LLM action extraction did not return a valid list" in caplog.text or \
                   "LLM action extraction returned invalid JSON" in caplog.text
