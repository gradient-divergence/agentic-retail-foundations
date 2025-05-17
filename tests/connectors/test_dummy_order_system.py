import logging
from datetime import datetime
from unittest.mock import patch

import pytest

# Module to test
from connectors.dummy_order_system import DummyOrderSystem


@pytest.fixture
def order_system() -> DummyOrderSystem:
    """Provides a DummyOrderSystem instance for testing."""
    return DummyOrderSystem()


# --- Test get_recent_orders --- #


@pytest.mark.asyncio
async def test_get_recent_orders_found(order_system):
    """Test retrieving recent orders for a known customer."""
    cid = "C123"
    orders = await order_system.get_recent_orders(cid, limit=2)
    assert len(orders) == 2
    # Verify sorted by date desc (ORD988 is more recent than ORD987)
    assert orders[0]["order_id"] == "ORD988"
    assert orders[1]["order_id"] == "ORD987"
    # Check limit works
    orders_all = await order_system.get_recent_orders(cid, limit=5)
    assert len(orders_all) == 2  # Only 2 orders exist for C123


@pytest.mark.asyncio
async def test_get_recent_orders_not_found(order_system):
    """Test retrieving orders for an unknown customer."""
    cid = "C_UNKNOWN"
    orders = await order_system.get_recent_orders(cid)
    assert orders == []


# --- Test get_order_details --- #


@pytest.mark.asyncio
async def test_get_order_details_found(order_system):
    """Test retrieving details for a known order."""
    oid = "ORD987"
    details = await order_system.get_order_details(oid)
    assert details is not None
    assert details["order_id"] == oid
    assert details["customer_id"] == "C123"


@pytest.mark.asyncio
async def test_get_order_details_not_found(order_system):
    """Test retrieving details for an unknown order."""
    oid = "ORD_UNKNOWN"
    details = await order_system.get_order_details(oid)
    assert details is None


# --- Test check_return_eligibility --- #


# Need to patch datetime.now for consistent results
@pytest.mark.asyncio
@patch("connectors.dummy_order_system.datetime")
async def test_check_return_eligibility_delivered_within_window(mock_dt, order_system):
    """Test eligibility for a delivered order within the 30-day window."""
    oid = "ORD987"  # Delivered on 2023-10-21
    fixed_now = datetime(2023, 11, 10)  # Within 30 days
    mock_dt.now.return_value = fixed_now
    mock_dt.fromisoformat.side_effect = datetime.fromisoformat  # Allow real fromisoformat

    result = await order_system.check_return_eligibility(oid)
    assert result == {"eligible": True}


@pytest.mark.asyncio
@patch("connectors.dummy_order_system.datetime")
async def test_check_return_eligibility_delivered_outside_window(mock_dt, order_system):
    """Test eligibility for a delivered order outside the 30-day window."""
    oid = "ORD987"  # Delivered on 2023-10-21
    fixed_now = datetime(2023, 11, 25)  # More than 30 days
    mock_dt.now.return_value = fixed_now
    mock_dt.fromisoformat.side_effect = datetime.fromisoformat

    result = await order_system.check_return_eligibility(oid)
    assert result == {"eligible": False, "reason": "Past 30-day return window."}


@pytest.mark.asyncio
async def test_check_return_eligibility_shipped(order_system):
    """Test eligibility for a shipped but not delivered order."""
    oid = "ORD988"  # Status Shipped
    result = await order_system.check_return_eligibility(oid)
    assert result == {"eligible": False, "reason": "Not yet delivered."}


@pytest.mark.asyncio
async def test_check_return_eligibility_other_status(order_system):
    """Test eligibility for an order with another status (e.g., Delayed)."""
    oid = "ORD999"  # Status Delayed
    result = await order_system.check_return_eligibility(oid)
    assert result == {"eligible": False, "reason": "Status is 'Delayed'."}


@pytest.mark.asyncio
async def test_check_return_eligibility_order_not_found(order_system):
    """Test eligibility for an unknown order ID."""
    oid = "ORD_UNKNOWN"
    result = await order_system.check_return_eligibility(oid)
    assert result == {"eligible": False, "reason": "Order not found."}


# --- Test get_return_policy --- #


def test_get_return_policy(order_system):
    """Test retrieving the static return policy."""
    policy = order_system.get_return_policy()
    assert isinstance(policy, dict)
    assert "return_window_days" in policy
    assert policy["return_window_days"] == 30  # Check a known value


# --- Test report_visual_audit --- #


@pytest.mark.asyncio
async def test_report_visual_audit(order_system, caplog):
    """Test the dummy report_visual_audit method logs correctly."""
    issue = {"section_id": "D5", "anomaly": "Spill detected"}
    with caplog.at_level(logging.INFO):
        await order_system.report_visual_audit(issue)

    assert "DUMMY: Received visual audit: D5" in caplog.text
