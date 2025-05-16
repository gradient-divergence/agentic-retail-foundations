# agents/llm_components.py

"""Components specific to the LLM Customer Service Agent, like prompt building and action extraction."""

import json
import logging
from typing import Any

from openai import AsyncOpenAI, OpenAI

from utils.openai_utils import safe_chat_completion

# Assuming prompts and safe_chat_completion are accessible
# Adjust imports based on actual project structure if needed
from .prompts import build_action_extraction_prompt

logger = logging.getLogger(__name__)

# --- Moved from response_builder.py --- #


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
    formatted_history = "\n".join(f"{msg['role'].capitalize()}: {msg.get('content', '')}" for msg in conversation_history)

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
        items_str = ", ".join(item.get("name", "Unknown Item") for item in order.get("items", []))
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
                (
                    f"- Price: ${product.get('price', 'N/A'):.2f}"
                    if isinstance(product.get("price"), (int, float))
                    else f"- Price: {product.get('price', 'N/A')}"
                ),  # Handle price formatting
                f"- Availability: {inventory.get('status', 'Please check')}",
            ]
        )
        context_added = True
    elif intent == "return_request" and context_data.get("return_eligibility"):
        eligibility = context_data["return_eligibility"]
        policy = context_data.get("return_policy", {})
        is_eligible = eligibility.get("eligible", False)
        system_prompt_parts.append(f"- Return Eligible: {'Yes' if is_eligible else 'No'}")
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
        system_prompt_parts.append("- No specific order or product context retrieved for this query.")

    # ------------------------ Instructions section ------------------------
    customer_name = customer_info.get("name", "Valued Customer")
    loyalty_tier = customer_info.get("loyalty_tier", "Standard")

    system_prompt_parts.extend(
        [
            "\nINSTRUCTIONS FOR RESPONSE:",
            "1. Be courteous, empathetic, and professional.",
            f"2. Address the customer as {customer_name}.",
            (
                f"3. If the customer is a loyalty member (tier is not 'Standard', 'Bronze', or 'Guest'), acknowledge their status positively (e.g., 'As a valued {loyalty_tier} member...')."
                if loyalty_tier not in ["Standard", "Bronze", "Guest", None]
                else ""
            ),
            "4. Directly answer the customer's query using the RELEVANT CONTEXT provided above.",
            "5. If context is missing or insufficient to answer fully, politely state what you can/cannot confirm and offer to find out more or suggest alternatives (do NOT invent details).",
            "6. For returns, if eligible, explain the next steps clearly (e.g., 'You can start your return at acmeretail.com/returns'). If not eligible, explain why based on the context.",
            "7. Keep responses concise, clear, and easy to understand (approx 2-4 sentences unless detail is required).",
            "8. Maintain a warm, helpful tone consistent with the ACME Retail brand.",
            "9. Do NOT mention the 'internal use only' context sections or the conversation history in your response to the customer.",
            "\nAgent Response:",  # Added delimiter
        ]
    )
    # Filter out empty strings that might result from conditional formatting (like loyalty tier)
    return "\n".join(filter(None, system_prompt_parts))


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
    if intent == "return_request" and context_data.get("return_eligibility", {}).get("eligible", False):
        order_id = context_data.get("order_details", {}).get("order_id")
        if order_id:
            actions.append({"type": "provide_return_instructions", "order_id": order_id})
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
    # Ensure client is AsyncOpenAI if called from async context
    if not isinstance(client, AsyncOpenAI):
        logger.warning("extract_actions received non-async client, LLM extraction might fail or block.")
        # Or raise TypeError

    prompt = build_action_extraction_prompt(response_text)
    try:
        messages = [
            {
                "role": "system",
                "content": 'You extract explicitly mentioned actions from agent text. Output ONLY a valid JSON array of strings representing action types (e.g. ["escalate_issue", "offer_discount"]). If no actions are mentioned, output an empty array [].',
            },
            {"role": "user", "content": prompt},
        ]
        completion = await safe_chat_completion(
            client,  # type: ignore # Expect Async Client here
            model=model,
            messages=messages,
            logger=logger,
            retry_attempts=retry_attempts,
            retry_backoff=retry_backoff,
            max_tokens=100,
            temperature=0,
        )

        extracted_text = (
            completion.choices[0].message.content.strip() if completion and completion.choices and completion.choices[0].message.content else "[]"
        )
        try:
            # Attempt to find JSON array within the response, even if there's prefix/suffix text
            import re

            match = re.search(r"\[.*?\]", extracted_text)
            json_str = match.group(0) if match else extracted_text

            parsed_data = json.loads(json_str)
            action_list: list[str] | None = None

            if isinstance(parsed_data, list):
                action_list = parsed_data
            elif isinstance(parsed_data, dict):
                # Handle {"actions": [...]} case
                for value in parsed_data.values():
                    if isinstance(value, list):
                        action_list = value
                        break

            if action_list is not None:
                validated = [a for a in action_list if isinstance(a, str)]
                existing_types = {a["type"] for a in actions}
                for action_type in validated:
                    if action_type not in existing_types:
                        actions.append({"type": action_type})
                logger.debug(f"LLM extracted actions: {validated}")
            else:
                logger.warning(
                    "LLM action extraction did not return a valid list/dict: %s",
                    extracted_text,
                )
        except json.JSONDecodeError:
            logger.warning("LLM action extraction returned invalid JSON: %s", extracted_text)
        except Exception as inner_e:
            logger.error("Error processing LLM action extraction result: %s", inner_e)

    except Exception as exc:
        logger.error("LLM action extraction call failed: %s", exc)

    logger.debug(f"Final combined actions: {actions}")
    return actions


# --- New function for response generation --- #


async def generate_agent_response(
    client: AsyncOpenAI,
    *,
    model: str,
    system_prompt: str,
    logger: logging.Logger,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
    max_tokens: int = 250,
    temperature: float = 0.7,
) -> str:
    """
    Generates the customer-facing response using the LLM.

    Args:
        client: The AsyncOpenAI client.
        model: The name of the response generation model.
        system_prompt: The fully constructed system prompt including context,
                       history, instructions, etc.
        logger: Logger instance.
        retry_attempts: Number of retry attempts.
        retry_backoff: Initial backoff delay for retries.
        max_tokens: Max tokens for the response.
        temperature: Sampling temperature.

    Returns:
        The generated response message string, or an empty string if generation fails.
    """
    if not isinstance(client, AsyncOpenAI):
        logger.error("generate_agent_response requires an AsyncOpenAI client.")
        return ""

    try:
        completion = await safe_chat_completion(
            client,
            model=model,
            messages=[{"role": "system", "content": system_prompt}],
            logger=logger,
            retry_attempts=retry_attempts,
            retry_backoff=retry_backoff,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=None,
        )
        if completion and completion.choices and completion.choices[0].message.content:
            generated_message = completion.choices[0].message.content.strip()
            logger.debug(f"LLM generated response: '{generated_message[:100]}...'")
            return generated_message
        else:
            logger.warning("LLM response generation returned no content.")
            return ""
    except Exception as e:
        logger.error(f"LLM response generation failed: {e}", exc_info=True)
        return ""  # Return empty string on failure
