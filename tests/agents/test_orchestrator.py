import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch
import logging
from datetime import datetime, timedelta

# Module to test
from agents.orchestrator import MasterOrchestrator
from agents.base import BaseAgent # Needed for patching handle_exception_event

# Dependent models/classes
from utils.event_bus import EventBus
from models.events import RetailEvent
from models.enums import AgentType, OrderStatus

# --- Test Fixtures --- #

@pytest.fixture
def mock_event_bus() -> MagicMock:
    # Use MagicMock as subscribe is sync
    bus = MagicMock(spec=EventBus)
    bus.subscribe = MagicMock()
    bus.publish = AsyncMock() # Publish is async
    return bus

@pytest.fixture
def orchestrator(mock_event_bus: MagicMock) -> MasterOrchestrator:
    # Prevent BaseAgent.__init__ trying to call register_event_handlers again
    with patch.object(BaseAgent, 'register_event_handlers'):
        agent = MasterOrchestrator(agent_id="master_01", event_bus=mock_event_bus)
    # Manually call register_event_handlers on the instance AFTER init
    # This ensures the instance's methods are bound before subscribe is called
    agent.register_event_handlers()
    return agent

# --- Test Initialization --- #

def test_orchestrator_initialization(orchestrator: MasterOrchestrator, mock_event_bus: MagicMock):
    """Test orchestrator initialization and event handler registration."""
    assert orchestrator.agent_id == "master_01"
    assert orchestrator.agent_type == AgentType.MASTER
    assert orchestrator.event_bus is mock_event_bus
    assert orchestrator.orders == {}

    # Check that subscribe was called for expected events
    assert mock_event_bus.subscribe.call_count >= 10 # Check a reasonable number
    mock_event_bus.subscribe.assert_any_call(
        "order.created", orchestrator.handle_order_event
    )
    mock_event_bus.subscribe.assert_any_call(
        "fulfillment.shipped", orchestrator.handle_order_event
    )
    mock_event_bus.subscribe.assert_any_call(
        "order.exception", orchestrator.handle_order_event
    )

# --- Test handle_order_event --- #

@pytest.mark.asyncio
@patch('agents.orchestrator.datetime') # Mock datetime for timestamps
@patch.object(MasterOrchestrator, 'handle_exception_event', new_callable=AsyncMock)
async def test_handle_order_event_tracking_and_status(
    mock_handle_exc, mock_dt, orchestrator: MasterOrchestrator, caplog
):
    """Test that events update order tracking (history, status, timestamp)."""
    fixed_now = datetime(2024, 1, 20, 10, 0, 0)
    mock_dt.now.return_value = fixed_now # Used for logging potentially

    order_id = "ORD_TRACK_1"
    event1_ts = (fixed_now - timedelta(minutes=5)).isoformat()
    event2_ts = fixed_now.isoformat()

    event1 = RetailEvent(
        event_type="order.created",
        payload={"order_id": order_id, "customer_id": "C1"},
        source=AgentType.ORDER_INGESTION,
        timestamp=event1_ts
    )
    event2 = RetailEvent(
        event_type="order.allocated",
        payload={"order_id": order_id, "items": []},
        source=AgentType.INVENTORY,
        timestamp=event2_ts
    )

    with caplog.at_level(logging.INFO):
        await orchestrator.handle_order_event(event1)
        await orchestrator.handle_order_event(event2)

    # Verify order tracking state
    assert order_id in orchestrator.orders
    order_state = orchestrator.orders[order_id]
    assert len(order_state["events"]) == 2
    assert order_state["events"][0]["event_type"] == "order.created"
    assert order_state["events"][0]["timestamp"] == event1_ts
    assert order_state["events"][1]["event_type"] == "order.allocated"
    assert order_state["events"][1]["timestamp"] == event2_ts
    assert order_state["last_update"] == event2_ts
    assert order_state["current_status"] == OrderStatus.ALLOCATED.value # Mapped from event type

    # Verify logging
    assert f"Order {order_id} - Event: order.created" in caplog.text
    assert f"Order {order_id} - Event: order.allocated" in caplog.text
    assert f"Status: {OrderStatus.ALLOCATED.value}" in caplog.text

    # Verify exception handler NOT called for these events
    mock_handle_exc.assert_not_awaited()

@pytest.mark.asyncio
@patch.object(MasterOrchestrator, 'handle_exception_event', new_callable=AsyncMock)
async def test_handle_order_event_calls_exception_handler(
    mock_handle_exc, orchestrator: MasterOrchestrator
):
    """Test that order.exception events are routed to the specific handler."""
    order_id = "ORD_EXC_1"
    exception_event = RetailEvent(
        event_type="order.exception",
        payload={"order_id": order_id, "error_details": {"type": "TestError"}},
        source=AgentType.FULFILLMENT,
        timestamp=datetime.now().isoformat()
    )

    await orchestrator.handle_order_event(exception_event)

    # Verify exception handler WAS called
    mock_handle_exc.assert_awaited_once_with(exception_event)

@pytest.mark.asyncio
async def test_handle_order_event_missing_order_id(orchestrator: MasterOrchestrator, caplog):
    """Test handling events missing order_id."""
    event_no_oid = RetailEvent(
        event_type="system.startup", # Event type doesn't typically have order_id
        payload={"component": "event_bus"},
        source=AgentType.SYSTEM
    )
    event_order_no_oid = RetailEvent(
        event_type="order.validated", # Should have order_id
        payload={"customer_id": "C1"},
        source=AgentType.VALIDATION
    )

    with caplog.at_level(logging.WARNING):
        await orchestrator.handle_order_event(event_no_oid)
        await orchestrator.handle_order_event(event_order_no_oid)

    # No order state should be created
    assert len(orchestrator.orders) == 0
    # Warning should only be logged for the order-related event
    assert "Event missing order_id: system.startup" not in caplog.text
    assert "Event missing order_id: order.validated" in caplog.text

# --- Test Exception Handling & Recovery --- #

@pytest.mark.asyncio
@patch.object(MasterOrchestrator, '_apply_recovery_strategy', new_callable=AsyncMock)
async def test_handle_exception_event(
    mock_apply_recovery, orchestrator: MasterOrchestrator, caplog
):
    """Test handling of order.exception events."""
    order_id = "ORD_EXC_2"
    error_details = {"error_type": "PaymentError", "error_message": "Card declined", "context": {"stage": "charge"}}
    source = AgentType.PAYMENT
    event = RetailEvent(
        event_type="order.exception",
        payload={"order_id": order_id, "error_details": error_details},
        source=source
    )

    # Pre-populate order state to check update
    orchestrator.orders[order_id] = {"current_status": OrderStatus.PROCESSING.value, "events": []}

    with caplog.at_level(logging.ERROR):
        await orchestrator.handle_exception_event(event)

    # Verify error log
    assert f"Order {order_id} - Exception: PaymentError from {source.value}" in caplog.text

    # Verify order state updated
    assert orchestrator.orders[order_id]["current_status"] == "exception"
    assert orchestrator.orders[order_id]["exception_details"] == error_details

    # Verify recovery strategy called
    mock_apply_recovery.assert_awaited_once_with(order_id, event)

@pytest.mark.asyncio
async def test_handle_exception_event_missing_order_id(
    orchestrator: MasterOrchestrator, caplog
):
    """Test handle_exception_event ignores events without order_id."""
    error_details = {"error_type": "UnknownError"}
    event = RetailEvent(
        event_type="order.exception",
        payload={"error_details": error_details}, # No order_id
        source=AgentType.SYSTEM
    )
    with patch.object(orchestrator, '_apply_recovery_strategy') as mock_recovery:
        with caplog.at_level(logging.ERROR):
            await orchestrator.handle_exception_event(event)

        assert "Exception event missing order_id" in caplog.text
        mock_recovery.assert_not_called()

@pytest.mark.asyncio
@patch.object(MasterOrchestrator, '_handle_inventory_allocation_failure', new_callable=AsyncMock)
@patch.object(MasterOrchestrator, '_handle_payment_failure', new_callable=AsyncMock)
@patch.object(MasterOrchestrator, '_escalate_to_human', new_callable=AsyncMock)
# Also mock publish_event to prevent downstream errors during this routing test
@patch.object(MasterOrchestrator, 'publish_event', new_callable=AsyncMock)
async def test_apply_recovery_strategy(
    mock_publish, # Add mock_publish argument
    mock_escalate, mock_handle_payment, mock_handle_inventory,
    orchestrator: MasterOrchestrator
):
    """Test _apply_recovery_strategy routes to correct handlers."""
    order_id = "ORD_RECOV_1"

    # Case 1: Inventory allocation failure
    event_inv = RetailEvent(
        "order.exception", {"order_id": order_id, "error_details": {"context": {"stage": "allocation"}}},
        source=AgentType.INVENTORY
    )
    await orchestrator._apply_recovery_strategy(order_id, event_inv)
    mock_handle_inventory.assert_awaited_once_with(order_id)
    mock_handle_payment.assert_not_awaited()
    mock_escalate.assert_not_awaited()
    mock_handle_inventory.reset_mock()

    # Case 2: Payment failure
    event_pay = RetailEvent(
        "order.exception", {"order_id": order_id, "error_details": {"error_type": "CardExpired"}},
        source=AgentType.PAYMENT
    )
    await orchestrator._apply_recovery_strategy(order_id, event_pay)
    mock_handle_inventory.assert_not_awaited()
    mock_handle_payment.assert_awaited_once_with(order_id, "CardExpired")
    mock_escalate.assert_not_awaited()
    mock_handle_payment.reset_mock()

    # Case 3: Unknown source/context -> Escalate
    event_unknown = RetailEvent(
        "order.exception", {"order_id": order_id, "error_details": {}},
        source=AgentType.FULFILLMENT # Example other source
    )
    await orchestrator._apply_recovery_strategy(order_id, event_unknown)
    mock_handle_inventory.assert_not_awaited()
    mock_handle_payment.assert_not_awaited()
    mock_escalate.assert_awaited_once_with(order_id, {})
    mock_escalate.reset_mock()

    # Case 4: No event provided (e.g., internal trigger) -> Escalate
    await orchestrator._apply_recovery_strategy(order_id, None)
    mock_handle_inventory.assert_not_awaited()
    mock_handle_payment.assert_not_awaited()
    mock_escalate.assert_awaited_once_with(order_id, {})

@pytest.mark.asyncio
@patch.object(MasterOrchestrator, 'publish_event', new_callable=AsyncMock)
async def test_recovery_helpers_publish_correct_events(mock_publish, orchestrator: MasterOrchestrator):
    """Test that the recovery helper methods publish the correct events."""
    order_id = "ORD_RECOV_HELPERS"

    # Test Inventory Allocation Failure
    mock_publish.reset_mock()
    await orchestrator._handle_inventory_allocation_failure(order_id)
    mock_publish.assert_awaited_once_with(
        "inventory.reallocation_requested",
        {
            "order_id": order_id,
            "allow_substitutions": True,
            "try_alternative_methods": True,
        }
    )

    # Test Payment Failure (Retry)
    mock_publish.reset_mock()
    await orchestrator._handle_payment_failure(order_id, "InsufficientFunds")
    mock_publish.assert_awaited_once_with(
        "payment.retry_requested",
        {"order_id": order_id, "retry_count": 1, "delay_minutes": 5}
    )

    # Test Payment Failure (Alternative)
    mock_publish.reset_mock()
    await orchestrator._handle_payment_failure(order_id, "CardInvalid")
    mock_publish.assert_awaited_once_with(
        "payment.alternative_requested",
        {"order_id": order_id, "original_error": "CardInvalid"}
    )

    # Test Escalation
    mock_publish.reset_mock()
    error_details = {"type": "WeirdError", "msg": "Something odd happened"}
    await orchestrator._escalate_to_human(order_id, error_details)
    mock_publish.assert_awaited_once_with(
        "support.ticket_created",
        {
            "order_id": order_id,
            "error_details": error_details,
            "priority": "high",
            "queue": "order_exceptions",
        }
    )

# Placeholder tests
# async def test_check_for_stalled_orders...(): ... 