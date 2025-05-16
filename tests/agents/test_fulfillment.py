import logging
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from agents.base import BaseAgent  # To patch methods if needed

# Module to test
from agents.fulfillment import FulfillmentAgent
from models.enums import AgentType, FulfillmentMethod, OrderStatus
from models.events import RetailEvent
from models.fulfillment import Order, OrderLineItem
from utils.event_bus import EventBus

# --- Test Fixtures --- #


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    bus = AsyncMock(spec=EventBus)
    bus.subscribe = MagicMock()  # Use MagicMock for sync subscribe call
    bus.publish = AsyncMock()  # Use AsyncMock for async publish call
    return bus


@pytest.fixture
def fulfillment_agent(mock_event_bus: AsyncMock) -> FulfillmentAgent:
    # Need to patch BaseAgent.__init__ to prevent double registration if necessary,
    # or ensure subscribe mock handles it. Let's assume subscribe mock is enough.
    return FulfillmentAgent(agent_id="fulfill_01", event_bus=mock_event_bus)


# --- Test Initialization --- #


def test_fulfillment_agent_initialization(fulfillment_agent: FulfillmentAgent, mock_event_bus: MagicMock):
    """Test agent initialization and event handler registration."""
    assert fulfillment_agent.agent_id == "fulfill_01"
    assert fulfillment_agent.agent_type == AgentType.FULFILLMENT
    assert fulfillment_agent.event_bus is mock_event_bus

    # Check if event handlers were registered
    # (Based on register_event_handlers implementation)
    expected_calls = [
        call("order.allocated", fulfillment_agent.handle_order_allocated),
        call("order.payment_processed", fulfillment_agent.handle_payment_processed),
    ]
    mock_event_bus.subscribe.assert_has_calls(expected_calls, any_order=True)
    # Add checks for other commented-out subscriptions if they get implemented


# --- Test Event Handlers --- #


@pytest.mark.asyncio
async def test_handle_order_allocated_success(fulfillment_agent: FulfillmentAgent, mock_event_bus: AsyncMock):
    """Test handling the order.allocated event successfully."""
    order_id = "ORD_ALLOC_1"
    order_total = 99.99
    event = RetailEvent(
        event_type="order.allocated",
        payload={"order_id": order_id, "order_total": order_total},
        source=AgentType.INVENTORY,  # Example source
    )

    await fulfillment_agent.handle_order_allocated(event)

    # Verify the correct payment request event was published
    mock_event_bus.publish.assert_awaited_once()
    call_args = mock_event_bus.publish.call_args
    published_event: RetailEvent = call_args.args[0]

    assert published_event.event_type == "payment.request_processing"
    assert published_event.source == fulfillment_agent.agent_type
    assert published_event.payload["order_id"] == order_id
    assert published_event.payload["amount"] == order_total


@pytest.mark.asyncio
async def test_handle_order_allocated_missing_order_id(fulfillment_agent: FulfillmentAgent, mock_event_bus: AsyncMock, caplog):
    """Test handling order.allocated event with missing order_id."""
    event = RetailEvent(
        event_type="order.allocated",
        payload={"order_total": 50.0},  # Missing order_id
        source=AgentType.INVENTORY,
    )

    with caplog.at_level(logging.ERROR):
        await fulfillment_agent.handle_order_allocated(event)

    # Verify error logged and no event published
    assert "Missing order_id in allocated order event" in caplog.text
    mock_event_bus.publish.assert_not_awaited()


@pytest.mark.asyncio
@patch.object(FulfillmentAgent, "publish_event", new_callable=AsyncMock)
@patch.object(BaseAgent, "handle_exception", new_callable=AsyncMock)  # Patch base handler
async def test_handle_order_allocated_publish_error(
    mock_handle_exception,
    mock_publish_event,
    fulfillment_agent: FulfillmentAgent,
    mock_event_bus: AsyncMock,
    caplog,
):
    """Test error handling if publishing payment request fails."""
    order_id = "ORD_ALLOC_ERR"
    event = RetailEvent(
        event_type="order.allocated",
        payload={"order_id": order_id, "order_total": 10.0},
        source=AgentType.INVENTORY,
    )
    publish_error = ConnectionError("Event bus down")
    mock_publish_event.side_effect = publish_error

    with caplog.at_level(logging.ERROR):
        await fulfillment_agent.handle_order_allocated(event)

    # Verify publish was attempted
    mock_publish_event.assert_awaited_once()
    # Verify error logged by handler
    assert f"Error initiating payment for order {order_id}" in caplog.text
    # Verify base exception handler was called
    mock_handle_exception.assert_awaited_once()
    exc_call_args = mock_handle_exception.call_args.kwargs
    assert exc_call_args["exception"] is publish_error
    assert exc_call_args["context"] == {
        "stage": "payment_initiation",
        "order_id": order_id,
    }
    assert exc_call_args["order"] is None


# --- Test Helper Methods --- #

# Need models for creating test Order/OrderLineItem


def test_group_items_by_fulfillment_mixed(fulfillment_agent: FulfillmentAgent):
    """Test grouping items with different methods and locations."""
    # Create mock OrderLineItems (Add price)
    item1 = OrderLineItem(
        product_id="P1",
        quantity=1,
        price=10.0,
        fulfillment_method=FulfillmentMethod.SHIP_FROM_STORE,
        fulfillment_location_id="S1",
    )
    item2 = OrderLineItem(
        product_id="P2",
        quantity=2,
        price=5.0,
        fulfillment_method=FulfillmentMethod.SHIP_FROM_STORE,
        fulfillment_location_id="S1",
    )
    item3 = OrderLineItem(
        product_id="P3",
        quantity=1,
        price=20.0,
        fulfillment_method=FulfillmentMethod.SHIP_FROM_WAREHOUSE,
        fulfillment_location_id="W1",
    )
    item4 = OrderLineItem(
        product_id="P4",
        quantity=1,
        price=15.0,
        fulfillment_method=FulfillmentMethod.PICKUP_IN_STORE,
        fulfillment_location_id="S2",
    )
    item5_missing = OrderLineItem(product_id="P5", quantity=1, price=1.0)  # Missing details, still needs price

    order = Order(
        order_id="GRP_TEST_1",
        customer_id="C1",
        items=[item1, item2, item3, item4, item5_missing],
        status=OrderStatus.CREATED,  # Use correct enum
    )

    groups = fulfillment_agent._group_items_by_fulfillment(order)

    # Expected groups: (SHIP_FROM_STORE, S1), (SHIP_FROM_WAREHOUSE, W1),
    # (PICKUP_IN_STORE, S2)
    assert len(groups) == 3

    # Check group 1 (SHIP_FROM_STORE, S1)
    group1 = next(g for g in groups if g[0] == FulfillmentMethod.SHIP_FROM_STORE and g[1] == "S1")
    assert group1 is not None
    assert len(group1[2]) == 2  # Items P1, P2
    assert item1 in group1[2]
    assert item2 in group1[2]

    # Check group 2 (SHIP_FROM_WAREHOUSE, W1)
    group2 = next(g for g in groups if g[0] == FulfillmentMethod.SHIP_FROM_WAREHOUSE and g[1] == "W1")
    assert group2 is not None
    assert len(group2[2]) == 1  # Item P3
    assert item3 in group2[2]

    # Check group 3 (PICKUP_IN_STORE, S2)
    group3 = next(g for g in groups if g[0] == FulfillmentMethod.PICKUP_IN_STORE and g[1] == "S2")
    assert group3 is not None
    assert len(group3[2]) == 1  # Item P4
    assert item4 in group3[2]

    # Item 5 should have been skipped


def test_group_items_by_fulfillment_single_group(fulfillment_agent: FulfillmentAgent):
    """Test grouping when all items have the same method/location."""
    item1 = OrderLineItem(
        product_id="P1",
        quantity=1,
        price=10.0,
        fulfillment_method=FulfillmentMethod.PICKUP_IN_STORE,
        fulfillment_location_id="S1",
    )
    item2 = OrderLineItem(
        product_id="P2",
        quantity=1,
        price=12.0,
        fulfillment_method=FulfillmentMethod.PICKUP_IN_STORE,
        fulfillment_location_id="S1",
    )
    order = Order(
        order_id="GRP_TEST_2",
        customer_id="C2",
        items=[item1, item2],
        status=OrderStatus.CREATED,
    )

    groups = fulfillment_agent._group_items_by_fulfillment(order)

    assert len(groups) == 1
    method, location, items = groups[0]
    assert method == FulfillmentMethod.PICKUP_IN_STORE
    assert location == "S1"
    assert len(items) == 2
    assert item1 in items
    assert item2 in items


def test_group_items_by_fulfillment_empty(fulfillment_agent: FulfillmentAgent):
    """Test grouping with an order that has no items."""
    order = Order(order_id="GRP_TEST_3", customer_id="C3", items=[], status=OrderStatus.CREATED)
    groups = fulfillment_agent._group_items_by_fulfillment(order)
    assert groups == []


def test_group_items_by_fulfillment_invalid_item(fulfillment_agent: FulfillmentAgent, caplog):
    """Test grouping skips items with missing fulfillment details."""
    item_valid = OrderLineItem(
        product_id="P1",
        quantity=1,
        price=10.0,
        fulfillment_method=FulfillmentMethod.SHIP_FROM_STORE,
        fulfillment_location_id="S1",
    )
    item_invalid = OrderLineItem(product_id="P_INVALID", quantity=1, price=1.0)  # Missing details, still needs price
    order = Order(
        order_id="GRP_TEST_4",
        customer_id="C4",
        items=[item_valid, item_invalid],
        status=OrderStatus.CREATED,
    )

    with caplog.at_level(logging.ERROR):
        groups = fulfillment_agent._group_items_by_fulfillment(order)

    assert len(groups) == 1  # Only the valid item forms a group
    assert groups[0][2] == [item_valid]
    assert "Item P_INVALID in order GRP_TEST_4 missing fulfillment details." in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_id, method, expected_target_agent",
    [
        ("ship_from_store", FulfillmentMethod.SHIP_FROM_STORE, AgentType.STORE_OPS),
        ("pickup", FulfillmentMethod.PICKUP_IN_STORE, AgentType.STORE_OPS),
        (
            "delivery_from_store",
            FulfillmentMethod.DELIVERY_FROM_STORE,
            AgentType.STORE_OPS,
        ),
        (
            "ship_from_warehouse",
            FulfillmentMethod.SHIP_FROM_WAREHOUSE,
            AgentType.WAREHOUSE,
        ),
    ],
    ids=lambda x: x[0] if isinstance(x, str) else "",  # Use test_id
)
@patch.object(FulfillmentAgent, "publish_event", new_callable=AsyncMock)
async def test_initiate_fulfillment_request_success(
    mock_publish,
    fulfillment_agent: FulfillmentAgent,
    test_id: str,
    method: FulfillmentMethod,
    expected_target_agent: AgentType,
):
    """Test initiating fulfillment request for different methods."""
    item1 = OrderLineItem(product_id="P1", quantity=1, price=10.0)
    order = Order(
        order_id="INIT_TEST_1",
        customer_id="C1",
        items=[item1],
        status=OrderStatus.CREATED,
    )
    location = "LOC_A" if method != FulfillmentMethod.SHIP_FROM_WAREHOUSE else "WH_B"
    items_list = [item1]

    await fulfillment_agent._initiate_fulfillment_request(order, method, location, items_list)

    # Verify publish was called correctly
    mock_publish.assert_awaited_once()
    call_args = mock_publish.call_args
    event_type = call_args.args[0]
    payload = call_args.args[1]

    assert event_type == "fulfillment.requested"
    assert payload["order_id"] == order.order_id
    assert payload["fulfillment_group_id"] == f"{order.order_id}-{method.value}-{location}"
    assert payload["fulfillment_method"] == method.value
    assert payload["location_id"] == location
    assert payload["target_agent_type"] == expected_target_agent.value
    assert len(payload["items"]) == 1
    assert payload["items"][0]["product_id"] == item1.product_id
    assert payload["items"][0]["quantity"] == item1.quantity
    assert payload["customer_id"] == order.customer_id


@pytest.mark.asyncio
@patch.object(FulfillmentAgent, "publish_event", new_callable=AsyncMock)
async def test_initiate_fulfillment_request_unknown_method(mock_publish, fulfillment_agent: FulfillmentAgent, caplog):
    """Test initiating fulfillment with an unknown method logs error."""
    item1 = OrderLineItem(product_id="P1", quantity=1, price=10.0)
    order = Order(
        order_id="INIT_TEST_2",
        customer_id="C1",
        items=[item1],
        status=OrderStatus.CREATED,
    )
    location = "LOC_X"
    items_list = [item1]
    unknown_method = MagicMock(spec=FulfillmentMethod)
    unknown_method.value = "UNKNOWN_METHOD"

    with caplog.at_level(logging.ERROR):
        await fulfillment_agent._initiate_fulfillment_request(order, unknown_method, location, items_list)

    # Verify error logged and publish not called
    assert "Cannot determine target agent for unknown fulfillment method" in caplog.text
    mock_publish.assert_not_awaited()


# --- Test handle_payment_processed --- #


@pytest.mark.asyncio
@patch.object(FulfillmentAgent, "_get_order", new_callable=AsyncMock)
@patch.object(FulfillmentAgent, "_group_items_by_fulfillment")  # Sync helper
@patch.object(FulfillmentAgent, "_initiate_fulfillment_request", new_callable=AsyncMock)
@patch.object(FulfillmentAgent, "publish_event", new_callable=AsyncMock)  # Mock publish directly
async def test_handle_payment_processed_success(
    mock_publish,
    mock_initiate_request,
    mock_group_items,
    mock_get_order,
    fulfillment_agent: FulfillmentAgent,
    mock_event_bus: AsyncMock,
):
    """Test successful handling of payment processed event."""
    order_id = "ORD_PAY_1"
    event = RetailEvent(
        event_type="order.payment_processed",
        payload={"order_id": order_id, "transaction_id": "txn_123"},
        source=AgentType.PAYMENT,  # Example source
    )

    # Setup Mocks
    mock_order = MagicMock(spec=Order)
    mock_order.order_id = order_id
    mock_order.customer_id = "CUST_MOCK"
    mock_order.delivery_address = {}
    mock_get_order.return_value = mock_order

    # Mock grouping result - ensure mocked items are OrderLineItem if structure matters
    group1_items = [MagicMock(spec=OrderLineItem, product_id="P1", quantity=1, price=10.0)]  # Add price if spec checks
    group2_items = [MagicMock(spec=OrderLineItem, product_id="P2", quantity=1, price=20.0)]
    mock_groups = [
        (FulfillmentMethod.SHIP_FROM_STORE, "S1", group1_items),
        (FulfillmentMethod.SHIP_FROM_WAREHOUSE, "W1", group2_items),
    ]
    mock_group_items.return_value = mock_groups

    # Execute
    await fulfillment_agent.handle_payment_processed(event)

    # Assertions
    mock_get_order.assert_awaited_once_with(order_id)
    mock_group_items.assert_called_once_with(mock_order)

    # Check initiate fulfillment called for each group
    assert mock_initiate_request.await_count == len(mock_groups)
    mock_initiate_request.assert_has_awaits(
        [
            call(mock_order, FulfillmentMethod.SHIP_FROM_STORE, "S1", group1_items),
            call(mock_order, FulfillmentMethod.SHIP_FROM_WAREHOUSE, "W1", group2_items),
        ],
        any_order=True,
    )

    # Check status update event published
    # It should be the last call to publish_event
    # Note: Using ANY for details dict as precise content isn't critical here
    mock_publish.assert_awaited_with(
        "order.status_update_request",
        {
            "order_id": order_id,
            "new_status": OrderStatus.PICKING.value,
            "details": {"fulfillment_groups": len(mock_groups)},
        },
    )


@pytest.mark.asyncio
@patch.object(FulfillmentAgent, "_get_order", new_callable=AsyncMock, return_value=None)
@patch.object(FulfillmentAgent, "_group_items_by_fulfillment")
@patch.object(FulfillmentAgent, "_initiate_fulfillment_request", new_callable=AsyncMock)
@patch.object(FulfillmentAgent, "publish_event", new_callable=AsyncMock)
async def test_handle_payment_processed_order_not_found(
    mock_publish,
    mock_initiate_request,
    mock_group_items,
    mock_get_order,
    fulfillment_agent: FulfillmentAgent,
    caplog,
):
    """Test handling payment processed when the order cannot be found."""
    order_id = "ORD_PAY_NF"
    event = RetailEvent(
        event_type="order.payment_processed",
        payload={"order_id": order_id},
        source=AgentType.PAYMENT,
    )

    with caplog.at_level(logging.ERROR):
        await fulfillment_agent.handle_payment_processed(event)

    # Verify get_order was called
    mock_get_order.assert_awaited_once_with(order_id)
    # Verify error logged
    assert f"Order {order_id} not found" in caplog.text
    # Verify no further actions taken
    mock_group_items.assert_not_called()
    mock_initiate_request.assert_not_awaited()
    mock_publish.assert_not_awaited()


# Placeholder tests
# async def test_handle_payment_processed...(): ...
