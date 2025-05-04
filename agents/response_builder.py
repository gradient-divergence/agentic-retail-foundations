from __future__ import annotations

"""Utilities for constructing the *system* prompt used to generate the final
customer-visible response and for extracting follow-up actions from that response.

By centralising this logic we keep `RetailCustomerServiceAgent` free from long
string-building sections and make prompts easier to unit-test in isolation.
"""

from typing import Any
import json
import logging

from openai import OpenAI, AsyncOpenAI

from agents.prompts import build_action_extraction_prompt
from utils.openai_utils import safe_chat_completion

__all__ = [
    "build_response_prompt",
    "extract_actions",
]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_response_prompt(
    *,
    customer_info: dict[str, Any],
    intent: str,
    message: str,
    context_data: dict[str, Any],
    conversation_history: list[dict[str, str]],
    brand_name: str = "ACME Retail",
) -> str:
    """Return the full *system* prompt to be sent to the response model."""
    formatted_history = "\n".join(
        f"{msg['role'].capitalize()}: {msg.get('content', '')}"
        for msg in conversation_history
    )

    system_prompt_parts: list[str] = [
        f"You are a helpful and friendly customer service agent for '{brand_name}'. Your goal is to assist the customer effectively and professionally.",
        "\nCUSTOMER INFORMATION:",
        f"- Name: {customer_info.get('name', 'Valued Customer')}",
        f"- Loyalty tier: {customer_info.get('loyalty_tier', 'Standard')}",
        f"- Customer since: {customer_info.get('customer_since', 'N/A')}",
    ]

    if conversation_history:
        system_prompt_parts.append("\nRECENT CONVERSATION HISTORY (Internal Use Only):")
        system_prompt_parts.append(formatted_history)

    system_prompt_parts.append("\nCURRENT CUSTOMER MESSAGE (Internal Use Only):")
    system_prompt_parts.append(f'"{message}"')

    # --------------------------- Context section ---------------------------
    system_prompt_parts.append("\nRELEVANT CONTEXT FOR RESPONSE (Internal Use Only):")
    context_added = False

    if intent == "order_status" and context_data.get("order_details"):
        order = context_data["order_details"]
        items_str = ", ".join(
            item.get("name", "Unknown Item") for item in order.get("items", [])
        )
        system_prompt_parts.extend(
            [
                f"- Order ID: {order.get('order_id', 'N/A')}",
                f"- Status: {order.get('status', 'N/A')}",
                f"- Items: {items_str}",
                f"- Estimated delivery: {order.get('estimated_delivery', 'N/A')}",
                f"- Tracking: {order.get('tracking_number', 'Not available')}",
            ]
        )
        context_added = True
    elif intent == "product_question" and context_data.get("product_details"):
        product = context_data["product_details"]
        inventory = context_data.get("inventory", {})
        system_prompt_parts.extend(
            [
                f"- Product Name: {product.get('name', 'N/A')}",
                f"- Price: ${product.get('price', 'N/A')}",
                f"- Availability: {inventory.get('status', 'Please check')}",
            ]
        )
        context_added = True
    elif intent == "return_request" and context_data.get("return_eligibility"):
        eligibility = context_data["return_eligibility"]
        policy = context_data.get("return_policy", {})
        is_eligible = eligibility.get("eligible", False)
        system_prompt_parts.append(
            f"- Return Eligible: {'Yes' if is_eligible else 'No'}"
        )
        if not is_eligible:
            system_prompt_parts.append(
                f"- Reason Not Eligible: {eligibility.get('reason', 'Policy timeframe likely exceeded or item non-returnable.')}"
            )
        system_prompt_parts.extend(
            [
                f"- Return Window: {policy.get('return_window_days', 'N/A')} days",
                f"- Return Methods: {', '.join(policy.get('return_methods', []))}",
            ]
        )
        context_added = True

    if not context_added:
        system_prompt_parts.append(
            "- No specific order or product context retrieved for this query."
        )

    # ------------------------ Instructions section ------------------------
    customer_name = customer_info.get("name", "Valued Customer")
    loyalty_tier = customer_info.get("loyalty_tier", "Standard")

    system_prompt_parts.extend(
        [
            "\nINSTRUCTIONS FOR RESPONSE:",
            "1. Be courteous, empathetic, and professional.",
            f"2. Address the customer as {customer_name}.",
            (
                f"3. If the customer is a loyalty member (not 'Standard' tier), acknowledge their status positively (e.g., 'As a valued {loyalty_tier} member...')."
            ),
            "4. Directly answer the customer's query using the RELEVANT CONTEXT provided above.",
            "5. If context is missing or insufficient to answer fully, politely state what you can/cannot confirm and offer to find out more or suggest alternatives (do NOT invent details).",
            "6. For returns, if eligible, explain the next steps clearly (e.g., 'You can start your return at acmeretail.com/returns'). If not eligible, explain why based on the context.",
            "7. Keep responses concise, clear, and easy to understand (approx 2-4 sentences unless detail is required).",
            "8. Maintain a warm, helpful tone consistent with the ACME Retail brand.",
            "9. Do NOT mention the 'internal use only' context sections or the conversation history in your response to the customer.",
            "\nAgent Response:",
        ]
    )

    return "\n".join(system_prompt_parts)


# ---------------------------------------------------------------------------
# Action extraction
# ---------------------------------------------------------------------------


async def extract_actions(
    client: AsyncOpenAI | OpenAI,
    *,
    intent: str,
    response_text: str,
    context_data: dict[str, Any],
    model: str,
    logger: logging.Logger,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
) -> list[dict[str, Any]]:
    """Extract structured actions from the generated response string."""

    actions: list[dict[str, Any]] = []

    # Deterministic, rule-based actions first
    if intent == "return_request" and context_data.get("return_eligibility", {}).get(
        "eligible", False
    ):
        order_id = context_data.get("order_details", {}).get("order_id")
        if order_id:
            actions.append(
                {"type": "provide_return_instructions", "order_id": order_id}
            )
    elif intent == "order_status" and context_data.get("order_details"):
        order_id = context_data["order_details"].get("order_id")
        status = context_data["order_details"].get("status", "").lower()
        if order_id and status in {"delayed", "lost", "investigating", "stuck"}:
            actions.append(
                {
                    "type": "escalate_issue",
                    "reason": f"{status}_order",
                    "order_id": order_id,
                }
            )

    # LLM-based action extraction
    prompt = build_action_extraction_prompt(response_text)
    try:
        # Revert to using messages
        messages = [
            {
                "role": "system",
                "content": "You extract explicitly mentioned actions from agent text. Output ONLY a valid JSON array of strings.",
            },
            {"role": "user", "content": prompt},
        ]
        completion = await safe_chat_completion(
            client,
            model=model,
            messages=messages, # Use messages
            # input=prompt, # Removed input
            # instructions=instructions, # Removed instructions
            logger=logger,
            retry_attempts=retry_attempts,
            retry_backoff=retry_backoff,
            max_tokens=100,
            temperature=0,
        )
        # Use choices[0].message.content
        extracted_text = completion.choices[0].message.content.strip() if completion.choices[0].message.content else "[]"
        try:
            parsed_data = json.loads(extracted_text)
            action_list: list[str] | None = None
            if isinstance(parsed_data, dict):
                # In case the LLM returns a dict wrapper: {"actions": [...]}
                for value in parsed_data.values():
                    if isinstance(value, list):
                        action_list = value
                        break
            elif isinstance(parsed_data, list):
                action_list = parsed_data
            if action_list is not None:
                validated = [a for a in action_list if isinstance(a, str)]
                existing_types = {a["type"] for a in actions}
                for action_type in validated:
                    if action_type not in existing_types:
                        actions.append({"type": action_type})
            else:
                logger.warning(
                    "LLM action extraction did not return a valid list: %s",
                    extracted_text,
                )
        except json.JSONDecodeError:
            logger.warning(
                "LLM action extraction returned invalid JSON: %s", extracted_text
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM action extraction failed: %s", exc)

    return actions
