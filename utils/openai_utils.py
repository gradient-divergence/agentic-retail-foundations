import asyncio
import logging
from collections.abc import Iterable
from typing import Any

# Use AsyncOpenAI for async operations
from openai import AsyncOpenAI, OpenAI

# Import specific message param types
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
)  # ChatCompletionSystemMessageParam,; ChatCompletionUserMessageParam,  # Use the base MessageParam type hint

# Define a type alias for the expected message structure
# ChatMessage = Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]

__all__ = ["safe_chat_completion"]


async def safe_chat_completion(
    client: AsyncOpenAI | OpenAI,
    *,
    model: str,
    messages: Iterable[dict[str, Any]],
    logger: logging.Logger | None = None,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
    **kwargs,
) -> ChatCompletion:
    """Safely invoke OpenAI chat completion endpoint with retries.

    Parameters
    ----------
    client:
        An initialised ``openai.OpenAI`` or ``openai.AsyncOpenAI`` client instance.
    model:
        The model name to call (e.g. ``"gpt-4o"``).
    messages:
        The messages for the chat completion endpoint.
    logger:
        Optional logger for diagnostics; if omitted a module-level logger is used.
    retry_attempts:
        How many times to retry on *any* exception.
    retry_backoff:
        Base back-off (in seconds); the delay grows exponentially
        (``backoff * 2**(attempt-1)``).
    **kwargs:
        Additional keyword arguments forwarded to ``client.chat.completions.create``.

    Returns
    -------
    ChatCompletion
        The raw response object returned by the OpenAI SDK.

    Raises
    ------
    Exception
        Re-raises the last encountered exception if all retry attempts fail.
    """
    if client is None:
        raise RuntimeError("OpenAI client is not initialised.")

    # Ensure client is async for async call
    if not isinstance(client, AsyncOpenAI):
        # Handle sync client case - either raise error or use asyncio.to_thread
        # For now, let's raise an error as this function is async
        raise TypeError("Sync OpenAI client provided to async safe_chat_completion.")

    logger = logger or logging.getLogger(__name__)
    start_ts: float
    last_exc: Exception | None = None
    # Cast messages just before use
    typed_messages: Iterable[ChatCompletionMessageParam] = messages  # type: ignore[assignment]

    for attempt in range(1, retry_attempts + 1):
        try:
            start_ts = asyncio.get_event_loop().time()
            # Now we know client is AsyncOpenAI
            completion = await client.chat.completions.create(
                model=model,
                messages=typed_messages,  # Use casted messages
                **kwargs,
            )
            latency = asyncio.get_event_loop().time() - start_ts
            logger.debug(
                "OpenAI completions.create call succeeded | model=%s | latency=%.2fs",
                model,
                latency,
            )
            assert isinstance(completion, ChatCompletion)
            return completion
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("OpenAI call failed (attempt %s/%s): %s", attempt, retry_attempts, exc)
            # Only sleep if there are more retries left
            if attempt < retry_attempts:
                await asyncio.sleep(retry_backoff * (2 ** (attempt - 1)))

    # All retries exhausted
    assert last_exc is not None  # for type checkers
    raise last_exc
