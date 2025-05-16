import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Models and utils needed
from agents.base import BaseAgent
from models.enums import AgentType, OrderStatus
from models.events import RetailEvent
from models.fulfillment import Order
from utils.event_bus import EventBus

# --- Test Fixtures --- #


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Provides a mocked EventBus."""
    return AsyncMock(spec=EventBus)


@pytest.fixture
def base_agent(mock_event_bus: AsyncMock) -> BaseAgent:
    """Provides a BaseAgent instance with a mocked event bus."""
    return BaseAgent(
        agent_id="base_test_agent_001",
        agent_type=AgentType.ORDER_MANAGER,  # Example type
        event_bus=mock_event_bus,
    )


# --- Test Initialization --- #


def test_base_agent_initialization(base_agent: BaseAgent, mock_event_bus: AsyncMock):
    """Test BaseAgent initialization."""
    assert base_agent.agent_id == "base_test_agent_001"
    assert base_agent.agent_type == AgentType.ORDER_MANAGER
    assert base_agent.event_bus is mock_event_bus


# --- Test publish_event --- #


@pytest.mark.asyncio
async def test_publish_event_success(base_agent: BaseAgent, mock_event_bus: AsyncMock):
    """Test successful event publishing."""
    event_type = "test.event"
    payload = {"data": "value"}

    await base_agent.publish_event(event_type, payload)

    # Assert publish was called once
    mock_event_bus.publish.assert_called_once()

    # Check the event object passed to publish
    call_args = mock_event_bus.publish.call_args
    published_event = call_args.args[0]

    assert isinstance(published_event, RetailEvent)
    assert published_event.event_type == event_type
    assert published_event.payload == payload
    assert published_event.source == base_agent.agent_type


@pytest.mark.asyncio
async def test_publish_event_no_bus(base_agent: BaseAgent, mock_event_bus: AsyncMock, caplog):
    """Test publishing when event_bus is None."""
    base_agent.event_bus = None  # Manually remove event bus
    event_type = "test.event"
    payload = {"data": "value"}

    with caplog.at_level(logging.ERROR):
        await base_agent.publish_event(event_type, payload)

    # Assert publish was NOT called
    mock_event_bus.publish.assert_not_called()
    # Assert error was logged
    assert f"Agent {base_agent.agent_id} has no event bus" in caplog.text


# --- Test handle_exception --- #


# Mock the Order class methods needed
@pytest.fixture
def mock_order() -> MagicMock:
    order = MagicMock(spec=Order)
    order.order_id = "ORD123"
    # Configure update_status to be an async mock if needed, but it's called directly
    # If BaseAgent awaited it, we'd need AsyncMock here.
    order.update_status = MagicMock()
    return order


@pytest.mark.asyncio
@patch.object(BaseAgent, "publish_event", new_callable=AsyncMock)
async def test_handle_exception_no_order(mock_publish, base_agent: BaseAgent, caplog):
    """Test handle_exception without an order object."""
    exception = ValueError("Something went wrong")
    context = {"step": "processing"}

    with caplog.at_level(logging.ERROR):
        await base_agent.handle_exception(exception, context)

    # Check error log
    assert f"Exception in {base_agent.agent_type.value} agent" in caplog.text
    assert str(exception) in caplog.text

    # Check system.exception event published without order_id
    mock_publish.assert_called_once()
    call_args = mock_publish.call_args
    event_type = call_args.args[0]
    payload = call_args.args[1]

    assert event_type == "system.exception"
    error_details = payload.get("error_details", {})
    assert error_details["error_type"] == "ValueError"
    assert error_details["error_message"] == str(exception)
    assert error_details["context"] == context
    assert error_details["agent_id"] == base_agent.agent_id
    assert "order_id" not in error_details


@pytest.mark.asyncio
@patch.object(BaseAgent, "publish_event", new_callable=AsyncMock)
async def test_handle_exception_with_order_success(mock_publish, base_agent: BaseAgent, mock_order: MagicMock, caplog):
    """Test handle_exception with an order, where status update succeeds."""
    exception = TypeError("Bad type")
    context = {"data": {"key": "value"}}

    with caplog.at_level(logging.ERROR):
        await base_agent.handle_exception(exception, context, order=mock_order)

    # Check error log
    assert f"Exception in {base_agent.agent_type.value} agent" in caplog.text
    assert str(exception) in caplog.text

    # Check order status update was called
    mock_order.update_status.assert_called_once()
    # Check args passed to update_status (status, source_agent, details_dict)
    update_args = mock_order.update_status.call_args.args
    assert update_args[0] == OrderStatus.EXCEPTION
    assert update_args[1] == base_agent.agent_type
    assert isinstance(update_args[2], dict)
    assert update_args[2]["error_type"] == "TypeError"

    # Check system.exception event published *with* order_id
    mock_publish.assert_called_once()
    call_args = mock_publish.call_args
    event_type = call_args.args[0]
    payload = call_args.args[1]
    error_details = payload.get("error_details", {})

    assert event_type == "system.exception"
    assert error_details["error_type"] == "TypeError"
    assert error_details["order_id"] == mock_order.order_id  # Check order_id is present
    assert error_details["context"] == context


@pytest.mark.asyncio
@patch.object(BaseAgent, "publish_event", new_callable=AsyncMock)
async def test_handle_exception_with_order_update_fails(mock_publish, base_agent: BaseAgent, mock_order: MagicMock, caplog):
    """Test handle_exception when the order.update_status call itself fails."""
    exception = KeyError("Missing key")
    context = {"file": "input.txt"}
    update_error = ConnectionError("DB connection failed")

    # Make the update_status mock raise an error
    mock_order.update_status.side_effect = update_error

    with caplog.at_level(logging.ERROR):
        await base_agent.handle_exception(exception, context, order=mock_order)

    # Check initial error log
    assert f"Exception in {base_agent.agent_type.value} agent" in caplog.text
    assert str(exception) in caplog.text

    # Check order status update was called
    mock_order.update_status.assert_called_once()

    # Check the SECOND error log (failure to update)
    assert f"Failed to update order status during exception handling for order {mock_order.order_id}" in caplog.text
    assert str(update_error) in caplog.text

    # Check system.exception event published (should still happen)
    mock_publish.assert_called_once()
    call_args = mock_publish.call_args
    event_type = call_args.args[0]
    payload = call_args.args[1]
    error_details = payload.get("error_details", {})

    assert event_type == "system.exception"
    assert error_details["error_type"] == "KeyError"
    # order_id might or might not be present depending on exact execution flow before exception
    # Let's just check it doesn't contain the update_error details
    assert error_details["context"] == context
    assert "DB connection failed" not in error_details.get("error_message", "")
