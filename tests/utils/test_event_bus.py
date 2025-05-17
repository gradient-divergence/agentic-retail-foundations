import logging
from unittest.mock import AsyncMock

import pytest

from models.enums import AgentType  # Required for RetailEvent
from models.events import RetailEvent  # Required for type hints and publish tests later
from utils.event_bus import EventBus


# Test Initialization
def test_event_bus_initialization():
    """Test that the EventBus initializes with an empty subscribers dict."""
    bus = EventBus()
    assert bus.subscribers == {}


# Test Subscription Logic
@pytest.mark.asyncio
async def test_subscribe_single_callback():
    """Test subscribing a single callback."""
    bus = EventBus()
    mock_callback = AsyncMock()
    event_type = "test_event"

    bus.subscribe(event_type, mock_callback)

    assert event_type in bus.subscribers
    assert len(bus.subscribers[event_type]) == 1
    assert bus.subscribers[event_type][0] is mock_callback


@pytest.mark.asyncio
async def test_subscribe_multiple_callbacks_same_event():
    """Test subscribing multiple different callbacks to the same event."""
    bus = EventBus()
    mock_callback1 = AsyncMock(name="cb1")
    mock_callback2 = AsyncMock(name="cb2")
    event_type = "test_event"

    bus.subscribe(event_type, mock_callback1)
    bus.subscribe(event_type, mock_callback2)

    assert event_type in bus.subscribers
    assert len(bus.subscribers[event_type]) == 2
    assert mock_callback1 in bus.subscribers[event_type]
    assert mock_callback2 in bus.subscribers[event_type]


@pytest.mark.asyncio
async def test_subscribe_multiple_callbacks_different_events():
    """Test subscribing callbacks to different events."""
    bus = EventBus()
    mock_callback1 = AsyncMock(name="cb1")
    mock_callback2 = AsyncMock(name="cb2")
    event_type1 = "event_a"
    event_type2 = "event_b"

    bus.subscribe(event_type1, mock_callback1)
    bus.subscribe(event_type2, mock_callback2)

    assert event_type1 in bus.subscribers
    assert len(bus.subscribers[event_type1]) == 1
    assert bus.subscribers[event_type1][0] is mock_callback1

    assert event_type2 in bus.subscribers
    assert len(bus.subscribers[event_type2]) == 1
    assert bus.subscribers[event_type2][0] is mock_callback2


@pytest.mark.asyncio
async def test_subscribe_duplicate_callback(caplog):
    """Test that subscribing the exact same callback twice is ignored."""
    bus = EventBus()
    mock_callback = AsyncMock(name="cb_duplicate")
    event_type = "test_event"

    with caplog.at_level(logging.WARNING):
        bus.subscribe(event_type, mock_callback)
        bus.subscribe(event_type, mock_callback)  # Attempt duplicate subscription

    assert event_type in bus.subscribers
    assert len(bus.subscribers[event_type]) == 1  # Should only be one
    assert bus.subscribers[event_type][0] is mock_callback
    assert "already subscribed" in caplog.text  # Check for warning log


def test_subscribe_non_callable():
    """Test that subscribing a non-callable raises TypeError."""
    bus = EventBus()
    non_callable = "not a function"
    event_type = "test_event"

    with pytest.raises(TypeError, match="Callback must be a callable async function."):
        bus.subscribe(event_type, non_callable)  # type: ignore [arg-type]

    assert event_type not in bus.subscribers  # Ensure nothing was added


# Test Unsubscription Logic
@pytest.mark.asyncio
async def test_unsubscribe_single_callback():
    """Test unsubscribing a specific callback leaves others."""
    bus = EventBus()
    mock_callback1 = AsyncMock(name="cb1")
    mock_callback2 = AsyncMock(name="cb2")
    event_type = "test_event"

    bus.subscribe(event_type, mock_callback1)
    bus.subscribe(event_type, mock_callback2)

    bus.unsubscribe(event_type, mock_callback1)

    assert event_type in bus.subscribers
    assert len(bus.subscribers[event_type]) == 1
    assert bus.subscribers[event_type][0] is mock_callback2
    assert mock_callback1 not in bus.subscribers[event_type]


@pytest.mark.asyncio
async def test_unsubscribe_last_callback():
    """Test unsubscribing the last callback removes the event type."""
    bus = EventBus()
    mock_callback = AsyncMock()
    event_type = "test_event"

    bus.subscribe(event_type, mock_callback)
    bus.unsubscribe(event_type, mock_callback)

    assert event_type not in bus.subscribers


@pytest.mark.asyncio
async def test_unsubscribe_nonexistent_callback(caplog):
    """Test unsubscribing a callback not subscribed to the event logs warning."""
    bus = EventBus()
    mock_callback1 = AsyncMock(name="cb1")
    mock_callback2 = AsyncMock(name="cb2_not_subscribed")
    event_type = "test_event"

    bus.subscribe(event_type, mock_callback1)

    with caplog.at_level(logging.WARNING):
        bus.unsubscribe(event_type, mock_callback2)  # Attempt to unsubscribe cb2

    assert event_type in bus.subscribers  # cb1 should still be there
    assert len(bus.subscribers[event_type]) == 1
    assert "Callback AsyncMock not found" in caplog.text  # Use class name


@pytest.mark.asyncio
async def test_unsubscribe_from_nonexistent_event_type():
    """Test unsubscribing from an event type with no subscribers."""
    bus = EventBus()
    mock_callback = AsyncMock()
    event_type = "nonexistent_event"

    # No error should be raised
    bus.unsubscribe(event_type, mock_callback)

    assert event_type not in bus.subscribers


# Test Publishing Logic


# Helper to create a dummy RetailEvent for testing
def create_test_event(event_type: str, payload: dict | None = None) -> RetailEvent:
    return RetailEvent(
        event_type=event_type,
        payload=payload if payload is not None else {},
        source=AgentType.TEST_AGENT,  # Use a dummy source
    )


@pytest.mark.asyncio
async def test_publish_calls_correct_subscribers():
    """Test that publish calls all and only the correct subscribers."""
    bus = EventBus()
    mock_callback_a1 = AsyncMock(name="cb_a1")
    mock_callback_a2 = AsyncMock(name="cb_a2")
    mock_callback_b1 = AsyncMock(name="cb_b1")

    event_a = "event_a"
    event_b = "event_b"

    bus.subscribe(event_a, mock_callback_a1)
    bus.subscribe(event_a, mock_callback_a2)
    bus.subscribe(event_b, mock_callback_b1)

    test_event_a = create_test_event(event_a, {"data": "value_a"})
    await bus.publish(test_event_a)

    # Check calls for event_a subscribers
    mock_callback_a1.assert_called_once_with(test_event_a)
    mock_callback_a2.assert_called_once_with(test_event_a)

    # Check that event_b subscriber was not called
    mock_callback_b1.assert_not_called()


@pytest.mark.asyncio
async def test_publish_no_subscribers():
    """Test publishing an event with no subscribers."""
    bus = EventBus()
    test_event = create_test_event("no_subscriber_event")

    # Should complete without errors
    await bus.publish(test_event)


@pytest.mark.asyncio
async def test_publish_with_callback_exception(caplog):
    """Test that publish handles exceptions in callbacks gracefully."""
    bus = EventBus()
    mock_callback_ok = AsyncMock(name="cb_ok")
    failing_callback = AsyncMock(name="cb_fail", side_effect=ValueError("Callback failed!"))

    event_type = "mixed_event"
    bus.subscribe(event_type, mock_callback_ok)
    bus.subscribe(event_type, failing_callback)

    test_event = create_test_event(event_type)

    with caplog.at_level(logging.ERROR):
        await bus.publish(test_event)

    # Verify the successful callback was still called
    mock_callback_ok.assert_called_once_with(test_event)

    # Verify the failing callback was called
    failing_callback.assert_called_once_with(test_event)

    # Verify the error was logged
    assert "Error in subscriber callback 'AsyncMock'" in caplog.text  # Use class name
    assert "Callback failed!" in caplog.text  # Check the specific exception message too


@pytest.mark.asyncio
async def test_publish_invalid_event_object(caplog):
    """Test publishing an object that is not a RetailEvent."""
    bus = EventBus()
    mock_callback = AsyncMock(name="cb1")
    event_type = "test_event"
    bus.subscribe(event_type, mock_callback)

    invalid_event = {"event_type": event_type, "payload": {}}  # A dict, not RetailEvent

    with caplog.at_level(logging.ERROR):
        await bus.publish(invalid_event)  # type: ignore [arg-type]

    # Verify error log
    assert f"Attempted to publish invalid event type: {type(invalid_event)}" in caplog.text

    # Verify callback was not called
    mock_callback.assert_not_called()
