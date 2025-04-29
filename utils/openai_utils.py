import asyncio
import logging
from typing import Any

from openai import OpenAI

__all__ = ["safe_chat_completion"]


async def safe_chat_completion(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    logger: logging.Logger | None = None,
    retry_attempts: int = 3,
    retry_backoff: float = 1.0,
    **kwargs,
) -> Any:
    """Safely invoke ``client.chat.responses.create`` with retries.

    Parameters
    ----------
    client:
        An initialised ``openai.OpenAI`` client instance.
    model:
        The model name to call (e.g. ``"gpt-4o"``).
    messages:
        The chat messages payload passed verbatim to the OpenAI endpoint.
    logger:
        Optional logger for diagnostics; if omitted a module-level logger is used.
    retry_attempts:
        How many times to retry on *any* exception.
    retry_backoff:
        Base back-off (in seconds); the delay grows exponentially (``backoff * 2**(attempt-1)``).
    **kwargs:
        Additional keyword arguments forwarded to ``client.chat.responses.create``.

    Returns
    -------
    Any
        The raw completion object returned by the OpenAI SDK.

    Raises
    ------
    Exception
        Re-raises the last encountered exception if all retry attempts fail.
    """
    if client is None:
        raise RuntimeError("OpenAI client is not initialised.")

    logger = logger or logging.getLogger(__name__)
    start_ts: float
    last_exc: Exception | None = None
    for attempt in range(1, retry_attempts + 1):
        try:
            start_ts = asyncio.get_event_loop().time()
            # Older SDKs expose `chat.responses.create`; newer ones use `chat.completions.create`.
            # Attempt `responses` first per project preference, then fall back.
            try:
                completion = await asyncio.to_thread(
                    client.chat.responses.create,
                    model=model,
                    messages=messages,
                    **kwargs,
                )
            except AttributeError:
                # Fall back to newer naming
                completion = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=messages,
                    **kwargs,
                )
            latency = asyncio.get_event_loop().time() - start_ts
            logger.debug(
                "OpenAI call succeeded | model=%s | tokens_n/a | latency=%.2fs",
                model,
                latency,
            )
            return completion
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "OpenAI call failed (attempt %s/%s): %s", attempt, retry_attempts, exc
            )
            await asyncio.sleep(retry_backoff * (2 ** (attempt - 1)))

    # All retries exhausted
    assert last_exc is not None  # for type checkers
    raise last_exc
