"""Natural Language Processing Utilities for Agentic Systems.

This module provides functions for common NLP tasks needed by retail agents,
such as intent classification, entity extraction (order IDs, product IDs),
and sentiment analysis, leveraging OpenAI's models.
"""

import logging

from openai import AsyncOpenAI

from agents.prompts import (
    build_intent_classification_prompt,
    build_order_id_inference_prompt,
    build_product_identifier_prompt,
    build_sentiment_prompt,
)
from utils.openai_utils import safe_chat_completion

__all__ = [
    "classify_intent",
    "extract_order_id_llm",
    "extract_product_id",
    "sentiment_analysis",
]

logger_nlp = logging.getLogger(__name__)


async def classify_intent(client: AsyncOpenAI, *, message: str) -> str:
    """Return one of the allowed intents inferred by the LLM."""
    # Ensure client is AsyncOpenAI
    if not isinstance(client, AsyncOpenAI):
        logger_nlp.error("Expected AsyncOpenAI client for classify_intent")
        return "unknown"  # Or raise TypeError

    prompt = build_intent_classification_prompt(message)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intent classification system. Respond with only one of "
                "the following intents: order_status, product_question, "
                "return_request, complaint, general_inquiry"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    completion = await safe_chat_completion(
        client,
        model="gpt-3.5-turbo",
        messages=messages,
        logger=logger_nlp,
        max_tokens=15,
        temperature=0,
    )
    # Handle potential None completion
    if completion and completion.choices and completion.choices[0].message.content:
        intent = completion.choices[0].message.content.strip().lower()
        # Validate against allowed intents
        allowed_intents = {
            "order_status",
            "product_question",
            "return_request",
            "complaint",
            "general_inquiry",
        }
        if intent in allowed_intents:
            return intent
        else:
            logger_nlp.warning(f"LLM returned unexpected intent: {intent}")
            return "unknown"  # Fallback should be unknown, not general_inquiry
    logger_nlp.warning("LLM completion failed or gave empty content.")
    return "unknown"  # Default if something went wrong


async def extract_order_id_llm(
    client: AsyncOpenAI,
    *,
    message: str,
    recent_order_ids: list[str],
    model: str,
    logger: logging.Logger,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
) -> str | None:
    """Extract an order ID using LLM, returning None if unable."""
    # Ensure client is AsyncOpenAI
    if not isinstance(client, AsyncOpenAI):
        pass  # Or raise TypeError("Expected AsyncOpenAI client for async function")

    prompt = build_order_id_inference_prompt(message, recent_order_ids)
    messages = [
        {
            "role": "system",
            "content": "You extract order IDs. Respond ONLY with the ID or 'None'. "
            "If multiple recent IDs are mentioned and it's ambiguous which "
            "one the user means, respond 'ambiguous'. If the message "
            "mentions an ID but it's not in the recent list, respond "
            "'not_recent'.",
        },
        {"role": "user", "content": prompt},
    ]
    completion = await safe_chat_completion(
        client,
        model=model,
        messages=messages,
        logger=logger,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
        max_tokens=20,  # Allow reasonable length for IDs
        temperature=0,
    )
    if completion and completion.choices and completion.choices[0].message.content:
        result = completion.choices[0].message.content.strip()
        # Handle empty string after strip
        if not result:
            return None
        # Handle explicit non-extraction cases
        if result.lower() in ["none", "ambiguous", "not_recent", "not_found"]:
            return None
        # Basic validation (e.g., alphanumeric, length) could be added here
        return result
    return None


async def extract_product_id(
    client: AsyncOpenAI,
    *,
    message: str,
    model: str,
    logger: logging.Logger,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
) -> str | None:
    """Extract a product identifier (SKU, name fragment) using LLM."""
    # Ensure client is AsyncOpenAI
    if not isinstance(client, AsyncOpenAI):
        pass  # Or raise TypeError("Expected AsyncOpenAI client for async function")

    prompt = build_product_identifier_prompt(message)
    messages = [
        {
            "role": "system",
            "content": "You extract product names or SKUs. Respond ONLY with the identifier or 'None'.",
        },
        {"role": "user", "content": prompt},
    ]
    completion = await safe_chat_completion(
        client,
        model=model,
        messages=messages,
        logger=logger,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
        max_tokens=50,  # Allow for longer product names
        temperature=0,
    )
    if completion and completion.choices and completion.choices[0].message.content:
        result = completion.choices[0].message.content.strip()
        # Handle empty string after strip
        if not result:
            return None
        # Basic check to filter out conversational fillers if the model fails
        # strict instruction
        if result.lower() != "none" and len(result.split()) < 10:  # Avoid long sentences
            return result
    return None


async def sentiment_analysis(
    client: AsyncOpenAI,
    *,
    message: str,
    model: str,
    logger: logging.Logger,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
) -> str:
    """Classify sentiment as positive, neutral, or negative."""
    # Ensure client is AsyncOpenAI
    if not isinstance(client, AsyncOpenAI):
        pass  # Or raise TypeError("Expected AsyncOpenAI client for async function")

    prompt = build_sentiment_prompt(message)
    messages = [
        {
            "role": "system",
            "content": "You classify text sentiment. Respond only: positive, neutral, or negative.",
        },
        {"role": "user", "content": prompt},
    ]
    completion = await safe_chat_completion(
        client,
        model=model,
        messages=messages,
        logger=logger,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
        max_tokens=10,
        temperature=0,
    )
    if completion and completion.choices and completion.choices[0].message.content:
        sentiment = completion.choices[0].message.content.strip().lower()
        if sentiment in ["positive", "neutral", "negative"]:
            return sentiment
        else:
            logger.warning(f"LLM returned unexpected sentiment: {sentiment}")
    return "neutral"  # Default to neutral if classification fails
