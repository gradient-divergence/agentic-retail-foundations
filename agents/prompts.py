from __future__ import annotations

"""Prompt builder utilities for `RetailCustomerServiceAgent` and related LLM-based agents.

The functions in this module **only** format the user-visible parts of prompts so they are
reusable from multiple call-sites.  System instructions or model parameters remain in the
calling functions because they may vary depending on the agent's configuration.

All builders return **plain strings** – they do *not* include any role metadata so the
caller is free to place them into either the `system` or `user` role as appropriate.
"""


__all__ = [
    "build_intent_classification_prompt",
    "build_order_id_inference_prompt",
    "build_product_identifier_prompt",
    "build_sentiment_prompt",
    "build_action_extraction_prompt",
]


def build_intent_classification_prompt(message: str) -> str:
    """Return the user prompt used for intent classification."""
    return f"""
        Classify the customer's message into one of the following intents:
        - order_status: Customer is asking about an existing order (e.g., 'where is my package?', 'delivery status')
        - product_question: Customer has a question about a specific product (e.g., 'is the yoga mat latex-free?', 'do you have red shoes?')
        - return_request: Customer wants to return or exchange an item (e.g., 'I want to return this', 'exchange needed')
        - complaint: Customer is expressing dissatisfaction (e.g., 'this is unacceptable', 'very unhappy')
        - general_inquiry: Other general questions (e.g., 'store hours?', 'do you sell gift cards?')
        Customer message: "{message}"
        Intent:
        """


def build_order_id_inference_prompt(message: str, recent_order_ids: list[str]) -> str:
    """Return the user prompt for inferring an order ID from context."""
    return f"""
        Analyze the customer message regarding orders.
        Customer message: "{message}"
        Recent order IDs available for this customer: {recent_order_ids}
        Instructions:
        1. If the message explicitly mentions one of the exact recent order IDs, return that ID.
        2. If the message uses phrases like "my order", "the order", "my recent purchase", "it" (clearly referring to an order) AND only ONE recent order ID is in the list, return that single recent order ID.
        3. If the message uses phrases like "my order", etc., AND MULTIPLE recent order IDs are in the list, return "ambiguous".
        4. If the message seems to mention an order ID but it's NOT in the recent list provided, return "not_recent".
        5. If no order is mentioned or clearly inferable from the context and recent orders, return "not_found".
        Respond ONLY with the exact extracted ID from the list, "ambiguous", "not_recent", or "not_found".
        """


def build_product_identifier_prompt(message: str) -> str:
    """Return the user prompt for extracting a product identifier from the message."""
    return f"""
        Identify the specific product name or product ID the customer is asking about in their message.
        Focus on the main item they are inquiring about.
        Customer message: "{message}"
        Instructions:
        - If a clear product name (e.g., "running shoes", "Yoga Mat", "Wireless Earbuds") or ID (e.g., "P5", "SKU123") is mentioned as the subject of the query, return it exactly as it appears or the most identifiable part.
        - If multiple products are mentioned, return the primary one being asked about.
        - If the message is about a general topic (like orders, returns, store hours) and not a specific product, return "not_found".
        Respond ONLY with the extracted product name/ID string or "not_found".
        """


def build_sentiment_prompt(message: str) -> str:
    """Return the user prompt for sentiment classification."""
    return f'Classify the sentiment: "positive", "neutral", or "negative". Respond with only one word.\nMessage: "{message}"\nSentiment:'


def build_action_extraction_prompt(response: str) -> str:
    """Return the user prompt for extracting actions from an agent response."""
    return f"""
        Analyze the following customer service agent response. Identify any concrete next steps or actions the agent explicitly stated *they* would take or that the *customer* needs to take next based *only* on the agent's text.
        Agent Response: "{response}"
        Examples: "send_email_confirmation", "check_inventory_further", "escalate_to_manager", "customer_needs_to_click_link", "customer_needs_to_provide_info"
        Instructions: List identified actions as a JSON array of simple descriptive strings (e.g., ["action1", "action2"]). If none, return [].
        Actions mentioned or implied in response:
        """
