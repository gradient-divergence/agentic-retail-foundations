import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Assume openai is installed and its types are available
# You might need `pip install openai` in your dev environment if not already present
# and potentially `types-openai` if strict type checking requires it,
# although `openai` includes its own types now.
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice

from utils.openai_utils import safe_chat_completion


# Helper to create a mock ChatCompletion response
def create_mock_completion(content: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-mock",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=content,
                    role="assistant",
                ),
                # logprobs=None, # Assuming logprobs are not needed for basic tests
            )
        ],
        created=1677652288,  # Example timestamp
        model="gpt-mock",
        object="chat.completion",
        # system_fingerprint=None, # Optional field
        # usage=CompletionUsage(completion_tokens=10, prompt_tokens=5, total_tokens=15) # Optional
    )


@pytest.mark.asyncio
async def test_safe_chat_completion_success(caplog):
    """Test successful call to safe_chat_completion without retries."""
    # Mock the client structure: client.chat.completions.create
    mock_create_method = AsyncMock(return_value=create_mock_completion("Test response content"))
    mock_client = MagicMock(spec=AsyncOpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = mock_create_method

    model = "gpt-test"
    messages = [{"role": "user", "content": "Test message"}]
    test_kwargs = {"temperature": 0.7}

    with caplog.at_level(logging.DEBUG):
        completion = await safe_chat_completion(
            client=mock_client,
            model=model,
            messages=messages,
            logger=logging.getLogger("test_logger"),
            retry_attempts=3,
            retry_backoff=0.1,
            **test_kwargs,
        )

    # Assert create was called once with the correct arguments
    mock_create_method.assert_called_once_with(
        model=model,
        messages=messages,
        temperature=0.7,  # Check that kwargs are passed through
    )

    # Assert the returned completion is the mocked response
    assert completion == mock_create_method.return_value
    assert completion.choices[0].message.content == "Test response content"

    # Assert success log message
    assert "OpenAI completions.create call succeeded" in caplog.text
    assert f"model={model}" in caplog.text
    # Check log level if needed: assert caplog.records[0].levelname == "DEBUG"


@pytest.mark.asyncio
async def test_safe_chat_completion_retry_on_failure(caplog):
    """Test that the function retries on failure and succeeds on the second attempt."""
    # Mock the client structure and side effect on create
    mock_success_response = create_mock_completion("Success after retry")
    mock_create_method = AsyncMock(
        side_effect=[
            TimeoutError("API timed out"),  # First call fails
            mock_success_response,  # Second call succeeds
        ]
    )
    mock_client = MagicMock(spec=AsyncOpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = mock_create_method

    model = "gpt-retry"
    messages = [{"role": "user", "content": "Retry test"}]
    retry_attempts = 3
    retry_backoff = 0.1  # Use a small backoff for testing speed

    # Mock asyncio.sleep to check if it's called and with what delay
    with (
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        caplog.at_level(logging.WARNING),
    ):  # Capture warning logs
        completion = await safe_chat_completion(
            client=mock_client,
            model=model,
            messages=messages,
            logger=logging.getLogger("retry_test_logger"),
            retry_attempts=retry_attempts,
            retry_backoff=retry_backoff,
        )

    # Assert create was called twice (failed once, succeeded once)
    assert mock_client.chat.completions.create.call_count == 2

    # Assert asyncio.sleep was called once with the initial backoff
    mock_sleep.assert_called_once_with(retry_backoff * (2 ** (1 - 1)))  # backoff * 2**(attempt-1)

    # Assert the final result is the successful response
    assert completion == mock_success_response

    # Assert warning log for the failed attempt
    assert "OpenAI call failed (attempt 1/3): API timed out" in caplog.text
    # Ensure no success DEBUG log was emitted by checking number of records
    # or specific content
    # (Assuming the test logger doesn't capture the DEBUG log from the
    # original function)


@pytest.mark.asyncio
async def test_safe_chat_completion_failure_after_retries(caplog):
    """Test that the function raises the last exception after all retries fail."""
    # Mock the client structure and side effect on create
    persistent_error = TimeoutError("Persistent API timeout")
    mock_create_method = AsyncMock(side_effect=persistent_error)
    mock_client = MagicMock(spec=AsyncOpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = mock_create_method

    model = "gpt-fail"
    messages = [{"role": "user", "content": "Failure test"}]
    retry_attempts = 3
    retry_backoff = 0.05  # Very small backoff for test speed

    # Mock asyncio.sleep to check calls without actually sleeping
    with (
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        pytest.raises(TimeoutError) as excinfo,
        caplog.at_level(logging.WARNING),
    ):
        await safe_chat_completion(
            client=mock_client,
            model=model,
            messages=messages,
            logger=logging.getLogger("fail_test_logger"),
            retry_attempts=retry_attempts,
            retry_backoff=retry_backoff,
        )

    # Assert the correct exception was raised
    assert excinfo.value is persistent_error

    # Assert create was called retry_attempts times
    assert mock_client.chat.completions.create.call_count == retry_attempts

    # Assert asyncio.sleep was called retry_attempts - 1 times
    assert mock_sleep.call_count == retry_attempts - 1
    # Check sleep durations (optional but good)
    expected_sleeps = [retry_backoff * (2 ** (i - 1)) for i in range(1, retry_attempts)]
    actual_sleeps = [call.args[0] for call in mock_sleep.call_args_list]
    assert actual_sleeps == expected_sleeps

    # Assert warning logs for all failed attempts
    assert f"OpenAI call failed (attempt 1/{retry_attempts})" in caplog.text
    assert f"OpenAI call failed (attempt 2/{retry_attempts})" in caplog.text
    assert f"OpenAI call failed (attempt 3/{retry_attempts})" in caplog.text
    assert str(persistent_error) in caplog.text


@pytest.mark.asyncio
async def test_safe_chat_completion_invalid_client():
    """Test passing invalid client types raises appropriate errors."""
    model = "gpt-invalid-client"
    messages = [{"role": "user", "content": "Invalid client test"}]

    # Test with None client
    with pytest.raises(RuntimeError, match="OpenAI client is not initialised."):
        await safe_chat_completion(
            client=None,  # type: ignore [arg-type]
            model=model,
            messages=messages,
        )

    # Test with sync client
    sync_client = OpenAI()  # Create a sync client instance
    with pytest.raises(TypeError, match="Sync OpenAI client provided to async safe_chat_completion."):
        await safe_chat_completion(
            client=sync_client,  # Pass the sync client
            model=model,
            messages=messages,
        )


# Placeholder for future tests
# async def test_safe_chat_completion_kwargs_forwarding(): ... # Covered partly above
