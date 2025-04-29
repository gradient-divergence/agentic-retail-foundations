from __future__ import annotations

"""NLP-related helpers that rely on LLM calls but **do not** have any business logic.

The goal is to keep the `RetailCustomerServiceAgent` lean by extracting reusable
functionality here.  The helpers expect an initialised `openai.OpenAI` client and
are pure in the sense that they do not mutate external state.
"""

import logging

from openai import OpenAI

from utils.openai_utils import safe_chat_completion
from agents.prompts import (
    build_intent_classification_prompt,
    build_order_id_inference_prompt,
    build_product_identifier_prompt,
    build_sentiment_prompt,
)

__all__ = [
    "classify_intent",
    "extract_order_id_via_llm",
    "extract_product_identifier",
    "analyze_sentiment",
]


async def classify_intent(
    client: OpenAI,
    *,
    message: str,
    model: str,
    logger: logging.Logger,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
) -> str:
    """Return one of the allowed intents inferred by the LLM."""
    prompt = build_intent_classification_prompt(message)
    completion = await safe_chat_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an intent classification system. Respond with only one of the following intents: "
                    "order_status, product_question, return_request, complaint, general_inquiry"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        logger=logger,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
        max_tokens=15,
        temperature=0,
        stop=["\n"],
    )
    intent_result = completion.choices[0].message.content.strip().lower()
    valid_intents = [
        "order_status",
        "product_question",
        "return_request",
        "complaint",
        "general_inquiry",
    ]
    return intent_result if intent_result in valid_intents else "general_inquiry"


async def extract_order_id_via_llm(
    client: OpenAI,
    *,
    message: str,
    recent_order_ids: list[str],
    model: str,
    logger: logging.Logger,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
) -> str:
    """Return best-guess order ID or helper keywords from the LLM."""
    prompt = build_order_id_inference_prompt(message, recent_order_ids)
    completion = await safe_chat_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You extract specific order references based on recent orders.",
            },
            {"role": "user", "content": prompt},
        ],
        logger=logger,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
        max_tokens=25,
        temperature=0,
    )
    return completion.choices[0].message.content.strip()


async def extract_product_identifier(
    client: OpenAI,
    *,
    message: str,
    model: str,
    logger: logging.Logger,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
) -> str | None:
    """Return a product identifier (name or ID) or ``None`` if not found."""
    prompt = build_product_identifier_prompt(message)
    completion = await safe_chat_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You extract specific product identifiers from text.",
            },
            {"role": "user", "content": prompt},
        ],
        logger=logger,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
        max_tokens=40,
        temperature=0,
    )
    result = completion.choices[0].message.content.strip()
    if result.lower() == "not_found" or len(result) < 2:
        return None
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    return result


async def analyze_sentiment(
    client: OpenAI,
    *,
    message: str,
    model: str,
    logger: logging.Logger,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
) -> str:
    """Classify sentiment as *positive*, *neutral*, or *negative*."""
    prompt = build_sentiment_prompt(message)
    completion = await safe_chat_completion(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You classify text sentiment. Respond only: positive, neutral, or negative.",
            },
            {"role": "user", "content": prompt},
        ],
        logger=logger,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
        max_tokens=10,
        temperature=0,
    )
    sentiment = (
        completion.choices[0]
        .message.content.strip()
        .lower()
        .replace(".", "")
        .replace(",", "")
    )
    return sentiment if sentiment in {"positive", "neutral", "negative"} else "neutral"
