import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock, call
from datetime import datetime, timedelta
import logging
from typing import Any
import asyncio

# Module to test
from agents.sensor import SensorDataProcessor

# --- Fixtures --- #

@pytest.fixture
def mock_inventory_system() -> AsyncMock:
    return AsyncMock()

@pytest.fixture
def mock_alert_system() -> AsyncMock:
    return AsyncMock()

@pytest.fixture
def sensor_processor(mock_inventory_system, mock_alert_system) -> SensorDataProcessor:
    # Disable FastAPI/uvicorn parts for unit testing core logic
    with patch('agents.sensor.FastAPI'): # Prevent FastAPI init
        processor = SensorDataProcessor(
            store_id="S_TEST",
            inventory_system=mock_inventory_system,
            alert_system=mock_alert_system
        )
    return processor

# --- Test Initialization --- #

def test_sensor_processor_initialization(sensor_processor, mock_inventory_system, mock_alert_system):
    """Test initialization sets defaults and stores dependencies."""
    assert sensor_processor.store_id == "S_TEST"
    assert sensor_processor.inventory_system is mock_inventory_system
    assert sensor_processor.alert_system is mock_alert_system
    assert isinstance(sensor_processor.confidence_thresholds, dict)
    assert sensor_processor.confidence_thresholds["rfid"] == 0.85 # Check default
    assert sensor_processor.recent_readings == {}
    assert sensor_processor.product_state == {}
    assert sensor_processor.discrepancies == {}
    # assert isinstance(sensor_processor.app, MagicMock) # Check FastAPI was mocked

# --- Test process_sensor_message Dispatching --- #

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_process_rfid_reading', new_callable=AsyncMock)
@patch.object(SensorDataProcessor, '_process_smart_shelf_reading', new_callable=AsyncMock)
@patch.object(SensorDataProcessor, '_process_environmental_reading', new_callable=AsyncMock)
@patch.object(SensorDataProcessor, '_process_price_tag_reading', new_callable=AsyncMock)
async def test_process_sensor_message_dispatch(
    mock_process_tag, mock_process_env, mock_process_shelf, mock_process_rfid,
    sensor_processor: SensorDataProcessor
):
    """Test process_sensor_message calls the correct handler based on sensor_type."""
    base_msg = {"sensor_id": "sensor123", "timestamp": datetime.now().isoformat()}

    # RFID
    msg_rfid = {**base_msg, "sensor_type": "rfid", "data": "..."}
    await sensor_processor.process_sensor_message(msg_rfid)
    mock_process_rfid.assert_awaited_once_with(msg_rfid)
    mock_process_shelf.assert_not_awaited()
    mock_process_env.assert_not_awaited()
    mock_process_tag.assert_not_awaited()
    mock_process_rfid.reset_mock()

    # Smart Shelf
    msg_shelf = {**base_msg, "sensor_type": "smart_shelf", "data": "..."}
    await sensor_processor.process_sensor_message(msg_shelf)
    mock_process_rfid.assert_not_awaited()
    mock_process_shelf.assert_awaited_once_with(msg_shelf)
    mock_process_env.assert_not_awaited()
    mock_process_tag.assert_not_awaited()
    mock_process_shelf.reset_mock()

    # Environmental
    msg_env = {**base_msg, "sensor_type": "environmental", "data": "..."}
    await sensor_processor.process_sensor_message(msg_env)
    mock_process_rfid.assert_not_awaited()
    mock_process_shelf.assert_not_awaited()
    mock_process_env.assert_awaited_once_with(msg_env)
    mock_process_tag.assert_not_awaited()
    mock_process_env.reset_mock()

    # Digital Price Tag
    msg_tag = {**base_msg, "sensor_type": "digital_price_tag", "data": "..."}
    await sensor_processor.process_sensor_message(msg_tag)
    mock_process_rfid.assert_not_awaited()
    mock_process_shelf.assert_not_awaited()
    mock_process_env.assert_not_awaited()
    mock_process_tag.assert_awaited_once_with(msg_tag)
    mock_process_tag.reset_mock()

    # Unknown type
    msg_unknown = {**base_msg, "sensor_type": "unknown_sensor", "data": "..."}
    await sensor_processor.process_sensor_message(msg_unknown)
    mock_process_rfid.assert_not_awaited()
    mock_process_shelf.assert_not_awaited()
    mock_process_env.assert_not_awaited()
    mock_process_tag.assert_not_awaited()

@pytest.mark.asyncio
async def test_process_sensor_message_invalid_sensor_id(sensor_processor: SensorDataProcessor, capsys):
    """Test message processing skips if sensor_id is invalid or missing."""
    # Patch process methods to ensure they are NOT called
    with patch.object(SensorDataProcessor, '_process_rfid_reading') as mock_process_rfid:
        # Missing sensor_id
        msg_missing = {"sensor_type": "rfid", "timestamp": datetime.now().isoformat()}
        await sensor_processor.process_sensor_message(msg_missing)
        mock_process_rfid.assert_not_called()
        captured_missing = capsys.readouterr()
        assert "Invalid or missing sensor_id" in captured_missing.out

        # Invalid sensor_id type
        msg_invalid = {"sensor_id": 123, "sensor_type": "rfid", "timestamp": datetime.now().isoformat()}
        await sensor_processor.process_sensor_message(msg_invalid)
        mock_process_rfid.assert_not_called()
        captured_invalid = capsys.readouterr()
        assert "Invalid or missing sensor_id" in captured_invalid.out

@pytest.mark.asyncio
# Patch datetime to control time for pruning
@patch('agents.sensor.datetime')
async def test_process_sensor_message_recent_readings(mock_dt, sensor_processor: SensorDataProcessor):
    """Test that recent_readings list is updated and pruned."""
    sensor_id = "s1"
    fixed_now = datetime(2024, 1, 12, 12, 0, 0)
    mock_dt.now.return_value = fixed_now
    # Ensure fromisoformat still works
    mock_dt.fromisoformat.side_effect = datetime.fromisoformat

    # Messages with controlled timestamps relative to fixed_now
    ts_now = fixed_now.isoformat()
    ts_1h_ago = (fixed_now - timedelta(hours=1)).isoformat()
    ts_2d_ago = (fixed_now - timedelta(days=2)).isoformat()

    msg1 = {"sensor_id": sensor_id, "sensor_type": "rfid", "timestamp": ts_1h_ago, "data": 1}
    msg2 = {"sensor_id": sensor_id, "sensor_type": "rfid", "timestamp": ts_now, "data": 2}
    msg_old = {"sensor_id": sensor_id, "sensor_type": "rfid", "timestamp": ts_2d_ago, "data": 0}

    # Process all messages
    with patch.object(SensorDataProcessor, '_process_rfid_reading') as mock_process_rfid:
        await sensor_processor.process_sensor_message(msg_old)
        # Pruning happens here based on msg_old's timestamp vs fixed_now -> msg_old IS older than 24h
        # but recent_readings[sensor_id] is created *inside* this call, so pruning an empty list? No, it's added first.
        # Let's trace: process(msg_old) -> recent_readings={'s1': [msg_old]} -> prune(cutoff=24h ago) -> recent_readings={'s1': []} ? No, prune compares item ts > cutoff
        # item ts = 2d ago. cutoff = now - 24h. So msg_old ts is NOT > cutoff. It should be removed.
        # So after first call, recent_readings[s1] should be empty.

        await sensor_processor.process_sensor_message(msg1)
        # msg1 ts = 1h ago. cutoff = now - 24h. msg1 ts > cutoff. Should be kept. recent_readings={'s1': [msg1]}

        await sensor_processor.process_sensor_message(msg2)
        # msg2 ts = now. cutoff = now - 24h. msg2 ts > cutoff. Should be kept.
        # Before adding msg2, list is [msg1]. Prune checks msg1 -> msg1 kept.
        # Add msg2 -> recent_readings={'s1': [msg1, msg2]}

    # Check final state
    assert sensor_id in sensor_processor.recent_readings
    assert len(sensor_processor.recent_readings[sensor_id]) == 2
    assert msg_old not in sensor_processor.recent_readings[sensor_id]
    assert msg1 in sensor_processor.recent_readings[sensor_id]
    assert msg2 in sensor_processor.recent_readings[sensor_id]

# --- Test Specific Processors --- #

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_handle_inventory_discrepancy', new_callable=AsyncMock)
async def test_process_rfid_reading_success_match(
    mock_handle_disc, sensor_processor: SensorDataProcessor, mock_inventory_system: AsyncMock
):
    """Test RFID processing when detected matches expected."""
    location = {"zone": "A", "section": "1"}
    timestamp = datetime.now().isoformat()
    product_list = [{"product_id": "P1"}, {"product_id": "P2"}]
    message = {
        "sensor_id": "rfid01", "sensor_type": "rfid", "timestamp": timestamp,
        "location": location, "confidence": 0.9, "detected_products": product_list
    }
    expected_location_str = "A.1"

    # Mock inventory system to return expected products
    mock_inventory_system.get_expected_products = AsyncMock(return_value={"P1", "P2"})
    mock_inventory_system.update_product_locations = AsyncMock()

    await sensor_processor._process_rfid_reading(message)

    # Verify inventory system calls
    mock_inventory_system.get_expected_products.assert_awaited_once_with(
        sensor_processor.store_id, expected_location_str
    )
    mock_inventory_system.update_product_locations.assert_awaited_once()
    # Check args for location update
    update_call_args = mock_inventory_system.update_product_locations.call_args.args
    assert update_call_args[0] == sensor_processor.store_id
    expected_update_payload = [
        {"product_id": "P1", "location": expected_location_str, "last_seen": timestamp, "confidence": 0.9},
        {"product_id": "P2", "location": expected_location_str, "last_seen": timestamp, "confidence": 0.9},
    ]
    assert update_call_args[1] == expected_update_payload

    # Verify no discrepancy handled
    mock_handle_disc.assert_not_awaited()

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_handle_inventory_discrepancy', new_callable=AsyncMock)
async def test_process_rfid_reading_missing_and_unexpected(
    mock_handle_disc, sensor_processor: SensorDataProcessor, mock_inventory_system: AsyncMock
):
    """Test RFID processing with both missing and unexpected items."""
    location = {"zone": "B", "section": "2"}
    timestamp = datetime.now().isoformat()
    detected_list = [{"product_id": "P1"}, {"product_id": "P3"}] # P2 missing, P3 unexpected
    message = {
        "sensor_id": "rfid02", "sensor_type": "rfid", "timestamp": timestamp,
        "location": location, "confidence": 0.9, "detected_products": detected_list
    }
    expected_location_str = "B.2"

    # Mock inventory system
    mock_inventory_system.get_expected_products = AsyncMock(return_value={"P1", "P2"}) # Expect P1, P2
    mock_inventory_system.update_product_locations = AsyncMock()

    await sensor_processor._process_rfid_reading(message)

    # Verify inventory system calls (update locations only for detected)
    mock_inventory_system.get_expected_products.assert_awaited_once_with(
        sensor_processor.store_id, expected_location_str
    )
    mock_inventory_system.update_product_locations.assert_awaited_once()
    update_call_args = mock_inventory_system.update_product_locations.call_args.args
    expected_update_payload = [
        {"product_id": "P1", "location": expected_location_str, "last_seen": timestamp, "confidence": 0.9},
        {"product_id": "P3", "location": expected_location_str, "last_seen": timestamp, "confidence": 0.9},
    ]
    # Use set comparison for list of dicts if order isn't guaranteed
    assert len(update_call_args[1]) == len(expected_update_payload)
    assert set(frozenset(d.items()) for d in update_call_args[1]) == set(frozenset(d.items()) for d in expected_update_payload)


    # Verify discrepancy calls
    assert mock_handle_disc.call_count == 2
    mock_handle_disc.assert_has_awaits([
        call(expected_location_str, ["P2"], "missing", "rfid"), # Missing P2
        call(expected_location_str, ["P3"], "unexpected", "rfid") # Unexpected P3
    ], any_order=True)

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_handle_inventory_discrepancy', new_callable=AsyncMock)
async def test_process_rfid_reading_low_confidence(
    mock_handle_disc, sensor_processor: SensorDataProcessor, mock_inventory_system: AsyncMock
):
    """Test RFID processing ignores readings below confidence threshold."""
    message = {
        "sensor_id": "rfid03", "sensor_type": "rfid", "timestamp": datetime.now().isoformat(),
        "location": {"zone": "C", "section": "1"}, "confidence": 0.5, # Below default 0.85
        "detected_products": [{"product_id": "P1"}]
    }

    await sensor_processor._process_rfid_reading(message)

    # Verify no calls were made
    mock_inventory_system.get_expected_products.assert_not_awaited()
    mock_inventory_system.update_product_locations.assert_not_awaited()
    mock_handle_disc.assert_not_awaited()

# --- Test Smart Shelf Processing --- #

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_handle_inventory_discrepancy', new_callable=AsyncMock)
async def test_process_smart_shelf_no_discrepancy(
    mock_handle_disc, sensor_processor: SensorDataProcessor, mock_inventory_system: AsyncMock
):
    """Test smart shelf reading when weight difference is below threshold."""
    message = {
        "sensor_id": "shelf01", "sensor_type": "smart_shelf", "timestamp": datetime.now().isoformat(),
        "shelf_id": "SH01", "location": {"zone": "A", "section": "1"},
        "current_weight_grams": 980.0,
        "expected_weight_grams": 1000.0,
        "product_info": {"product_id": "P_SHELF", "unit_weight_grams": 50.0}
    }
    # Difference = 20g, Threshold = 50.0 * 0.5 = 25g. Should be ignored.

    await sensor_processor._process_smart_shelf_reading(message)

    # Verify no discrepancy or update calls made
    mock_handle_disc.assert_not_awaited()
    mock_inventory_system.update_product_quantity.assert_not_awaited()

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_handle_inventory_discrepancy', new_callable=AsyncMock)
async def test_process_smart_shelf_low_stock(
    mock_handle_disc, sensor_processor: SensorDataProcessor, mock_inventory_system: AsyncMock
):
    """Test smart shelf reading indicating low stock."""
    location = {"zone": "A", "section": "1"}
    shelf_id = "SH01"
    product_id = "P_LOW"
    timestamp = datetime.now().isoformat()
    message = {
        "sensor_id": "shelf01", "sensor_type": "smart_shelf", "timestamp": timestamp,
        "shelf_id": shelf_id, "location": location,
        "current_weight_grams": 120.0, # Estimated 2 units (120/50=2.4->2)
        "expected_weight_grams": 500.0, # Expected 10 units
        "product_info": {"product_id": product_id, "unit_weight_grams": 50.0}
    }
    expected_location_str = "A.1.SH01"
    expected_units = 10
    estimated_units = 2

    await sensor_processor._process_smart_shelf_reading(message)

    # Verify discrepancy call for low_stock
    mock_handle_disc.assert_awaited_once_with(
        expected_location_str, [product_id], "low_stock", "smart_shelf",
        {"expected_units": expected_units, "estimated_units": estimated_units, "confidence": 0.9}
    )
    # Verify inventory update call
    mock_inventory_system.update_product_quantity.assert_awaited_once_with(
        sensor_processor.store_id, product_id, estimated_units, expected_location_str,
        timestamp, source="smart_shelf"
    )

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_handle_inventory_discrepancy', new_callable=AsyncMock)
async def test_process_smart_shelf_out_of_stock(
    mock_handle_disc, sensor_processor: SensorDataProcessor, mock_inventory_system: AsyncMock
):
    """Test smart shelf reading indicating out of stock."""
    location = {"zone": "A", "section": "1"}
    shelf_id = "SH01"
    product_id = "P_OOS"
    timestamp = datetime.now().isoformat()
    message = {
        "sensor_id": "shelf01", "sensor_type": "smart_shelf", "timestamp": timestamp,
        "shelf_id": shelf_id, "location": location,
        "current_weight_grams": 10.0, # Estimated 0 units (10/50=0.2->0)
        "expected_weight_grams": 500.0, # Expected 10 units
        "product_info": {"product_id": product_id, "unit_weight_grams": 50.0}
    }
    expected_location_str = "A.1.SH01"
    expected_units = 10
    estimated_units = 0

    await sensor_processor._process_smart_shelf_reading(message)

    # Verify discrepancy call for out_of_stock
    mock_handle_disc.assert_awaited_once_with(
        expected_location_str, [product_id], "out_of_stock", "smart_shelf",
        {"expected_units": expected_units, "estimated_units": estimated_units, "confidence": 0.9}
    )
    # Verify inventory update call
    mock_inventory_system.update_product_quantity.assert_awaited_once_with(
        sensor_processor.store_id, product_id, estimated_units, expected_location_str,
        timestamp, source="smart_shelf"
    )

@pytest.mark.asyncio
async def test_process_smart_shelf_invalid_weights(sensor_processor: SensorDataProcessor, capsys):
    """Test smart shelf processing with invalid weight data."""
    message = {
        "sensor_id": "shelf01", "sensor_type": "smart_shelf", "timestamp": datetime.now().isoformat(),
        "shelf_id": "SH01", "location": {"zone": "A", "section": "1"},
        "current_weight_grams": "heavy", # Invalid
        "expected_weight_grams": 1000.0,
        "product_info": {"product_id": "P_SHELF", "unit_weight_grams": 50.0}
    }
    await sensor_processor._process_smart_shelf_reading(message)
    captured = capsys.readouterr()
    assert "Invalid weight values" in captured.out

# --- Test Environmental Processing --- #

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_id, sensor_type, value, unit, location, expected_alert, expected_priority, expect_quality_check",
    [
        # Temperature: Refrigerated - OK
        ("temp_ref_ok", "temperature", 4.0, "C", {"zone": "Ref", "section": "1", "zone_type": "refrigerated"}, False, None, False),
        # Temperature: Refrigerated - Medium Alert
        ("temp_ref_med", "temperature", 6.0, "C", {"zone": "Ref", "section": "1", "zone_type": "refrigerated"}, True, "medium", True),
        # Temperature: Refrigerated - High Alert
        ("temp_ref_high", "temperature", 9.0, "C", {"zone": "Ref", "section": "1", "zone_type": "refrigerated"}, True, "high", True),
        # Temperature: Frozen - OK
        ("temp_frz_ok", "temperature", -18.0, "C", {"zone": "Frz", "section": "A", "zone_type": "frozen"}, False, None, False),
        # Temperature: Frozen - Medium Alert
        ("temp_frz_med", "temperature", -12.0, "C", {"zone": "Frz", "section": "A", "zone_type": "frozen"}, True, "medium", True),
        # Temperature: Frozen - High Alert
        ("temp_frz_high", "temperature", -8.0, "C", {"zone": "Frz", "section": "A", "zone_type": "frozen"}, True, "high", True),
        # Temperature: Ambient - OK (no specific checks)
        ("temp_amb_ok", "temperature", 22.0, "C", {"zone": "Amb", "section": "5", "zone_type": "ambient"}, False, None, False),
        # Humidity: Produce - OK
        ("hum_prod_ok", "humidity", 85.0, "%", {"zone": "Prod", "section": "P1", "zone_type": "produce"}, False, None, False),
        # Humidity: Produce - Low Alert
        ("hum_prod_low", "humidity", 75.0, "%", {"zone": "Prod", "section": "P1", "zone_type": "produce"}, True, "medium", False),
        # Humidity: Produce - High Alert
        ("hum_prod_high", "humidity", 98.0, "%", {"zone": "Prod", "section": "P1", "zone_type": "produce"}, True, "medium", False),
        # Humidity: Ambient - OK (no specific checks)
        ("hum_amb_ok", "humidity", 50.0, "%", {"zone": "Amb", "section": "5", "zone_type": "ambient"}, False, None, False),
        # Invalid value type
        ("invalid_value", "temperature", "hot", "C", {"zone": "Ref", "section": "1", "zone_type": "refrigerated"}, False, None, False),
    ],
    ids=lambda x: x if isinstance(x, str) else "" # Use test_id
)
async def test_process_environmental_reading(
    test_id: str, sensor_type: str, value: Any, unit: str, location: dict,
    expected_alert: bool, expected_priority: str | None, expect_quality_check: bool,
    sensor_processor: SensorDataProcessor, mock_alert_system: AsyncMock, mock_inventory_system: AsyncMock,
    capsys # Capture print warnings for invalid value
):
    """Test environmental sensor processing logic and alerting."""
    timestamp = datetime.now().isoformat()
    message = {
        "sensor_id": f"env_{test_id}", "sensor_type": "environmental", "timestamp": timestamp,
        "environmental_type": sensor_type, "value": value, "unit": unit, "location": location
    }

    await sensor_processor._process_environmental_reading(message)

    if expected_alert:
        mock_alert_system.send_alert.assert_awaited_once()
        call_args = mock_alert_system.send_alert.call_args.kwargs
        assert call_args["alert_type"] == "environmental"
        assert call_args["priority"] == expected_priority
        assert call_args["location"] == f"{location.get('zone')}.{location.get('section')}"
        details = call_args["details"]
        assert details["sensor_type"] == sensor_type
        assert details["value"] == value
        assert details["unit"] == unit
        assert details["threshold_exceeded"] is True
    else:
        mock_alert_system.send_alert.assert_not_awaited()

    if expect_quality_check:
        mock_inventory_system.flag_products_for_quality_check.assert_awaited_once_with(
            sensor_processor.store_id,
            location=f"{location.get('zone')}.{location.get('section')}",
            reason=f"Temperature threshold exceeded: {value}{unit}",
            timestamp=timestamp
        )
    else:
        mock_inventory_system.flag_products_for_quality_check.assert_not_awaited()

    if test_id == "invalid_value":
        captured = capsys.readouterr()
        assert "Warning: Invalid or missing numeric value" in captured.out

# --- Test Price Tag Processing --- #

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_request_price_tag_update', new_callable=AsyncMock)
async def test_process_price_tag_reading_low_battery(
    mock_request_update, sensor_processor: SensorDataProcessor, mock_alert_system: AsyncMock, mock_inventory_system: AsyncMock
):
    """Test processing price tag with low battery triggers maintenance alert."""
    tag_id = "TAG001"
    product_id = "P_TAG1"
    location = {"zone": "Z", "section": "S1"}
    message = {
        "sensor_id": "tag_sensor_1", "sensor_type": "digital_price_tag", "timestamp": datetime.now().isoformat(),
        "tag_id": tag_id, "product_id": product_id, "location": location,
        "price_displayed": 19.99, "battery_level": 15 # Low battery
    }

    # Mock inventory system to return a price (doesn't matter if it matches for this test)
    mock_inventory_system.get_current_price = AsyncMock(return_value=19.99)

    await sensor_processor._process_price_tag_reading(message)

    # Verify maintenance alert sent
    mock_alert_system.send_alert.assert_awaited_once()
    call_args = mock_alert_system.send_alert.call_args.kwargs
    assert call_args["alert_type"] == "maintenance"
    assert call_args["priority"] == "low"
    assert call_args["location"] == "Z.S1"
    details = call_args["details"]
    assert details["device_id"] == tag_id
    assert details["battery_level"] == 15
    assert details["product_id"] == product_id

    # Verify price check still happened
    mock_inventory_system.get_current_price.assert_awaited_once_with(sensor_processor.store_id, product_id)
    # Verify price update NOT requested (as price matches in this specific case)
    mock_request_update.assert_not_awaited()

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_request_price_tag_update', new_callable=AsyncMock)
async def test_process_price_tag_reading_price_mismatch(
    mock_request_update, sensor_processor: SensorDataProcessor, mock_alert_system: AsyncMock, mock_inventory_system: AsyncMock
):
    """Test processing price tag with a price mismatch."""
    tag_id = "TAG002"
    product_id = "P_TAG2"
    location = {"zone": "Z", "section": "S2"}
    displayed_price = 25.50
    expected_price = 24.99
    message = {
        "sensor_id": "tag_sensor_2", "sensor_type": "digital_price_tag", "timestamp": datetime.now().isoformat(),
        "tag_id": tag_id, "product_id": product_id, "location": location,
        "price_displayed": displayed_price, "battery_level": 90 # Battery OK
    }

    # Mock inventory system to return the expected price
    mock_inventory_system.get_current_price = AsyncMock(return_value=expected_price)

    await sensor_processor._process_price_tag_reading(message)

    # Verify price discrepancy alert sent
    mock_alert_system.send_alert.assert_awaited_once()
    call_args = mock_alert_system.send_alert.call_args.kwargs
    assert call_args["alert_type"] == "price_discrepancy"
    assert call_args["priority"] == "medium"
    assert call_args["location"] == "Z.S2"
    details = call_args["details"]
    assert details["product_id"] == product_id
    assert details["displayed_price"] == displayed_price
    assert details["expected_price"] == expected_price
    assert details["tag_id"] == tag_id

    # Verify price update WAS requested
    mock_request_update.assert_awaited_once_with(tag_id, product_id, expected_price)

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_request_price_tag_update', new_callable=AsyncMock)
async def test_process_price_tag_reading_price_match(
    mock_request_update, sensor_processor: SensorDataProcessor, mock_alert_system: AsyncMock, mock_inventory_system: AsyncMock
):
    """Test processing price tag when price matches expected."""
    tag_id = "TAG003"
    product_id = "P_TAG3"
    location = {"zone": "Z", "section": "S3"}
    price = 30.00
    message = {
        "sensor_id": "tag_sensor_3", "sensor_type": "digital_price_tag", "timestamp": datetime.now().isoformat(),
        "tag_id": tag_id, "product_id": product_id, "location": location,
        "price_displayed": price, "battery_level": 80 # Battery OK
    }
    mock_inventory_system.get_current_price = AsyncMock(return_value=price)

    await sensor_processor._process_price_tag_reading(message)

    # Verify NO alert sent
    mock_alert_system.send_alert.assert_not_awaited()
    # Verify price update NOT requested
    mock_request_update.assert_not_awaited()

@pytest.mark.asyncio
async def test_process_price_tag_invalid_price_data(
    sensor_processor: SensorDataProcessor, mock_alert_system: AsyncMock, mock_inventory_system: AsyncMock, capsys
):
    """Test handling invalid price data type."""
    message = {
        "sensor_id": "tag_sensor_4", "sensor_type": "digital_price_tag", "timestamp": datetime.now().isoformat(),
        "tag_id": "TAG004", "product_id": "P_TAG4", "location": {},
        "price_displayed": "Unknown", "battery_level": 90
    }
    mock_inventory_system.get_current_price = AsyncMock(return_value=10.0)
    await sensor_processor._process_price_tag_reading(message)
    captured = capsys.readouterr()
    assert "Warning: Could not compare prices" in captured.out
    mock_alert_system.send_alert.assert_not_awaited()

# --- Test Discrepancy Handling --- #

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_calculate_discrepancy_confidence')
async def test_handle_inventory_discrepancy_first_detection(
    mock_calc_conf, sensor_processor: SensorDataProcessor,
    mock_inventory_system: AsyncMock, mock_alert_system: AsyncMock
):
    """Test first detection of a discrepancy creates a record but doesn't alert/report yet."""
    location = "A.1"
    product_id = "P1"
    discrepancy_type = "missing"
    source = "rfid"
    details = {"info": "detail1"}

    # Simulate low confidence initially
    mock_calc_conf.return_value = 0.6

    await sensor_processor._handle_inventory_discrepancy(
        location, [product_id], discrepancy_type, source, details
    )

    # Verify record created
    discrepancy_key = f"{product_id}:{location}:{discrepancy_type}"
    assert discrepancy_key in sensor_processor.discrepancies
    record = sensor_processor.discrepancies[discrepancy_key]
    assert record["product_id"] == product_id
    assert record["location"] == location
    assert record["type"] == discrepancy_type
    assert record["detection_count"] == 1
    assert record["sources"] == [source]
    assert record["details"] == details
    assert "first_detected" in record
    assert "last_updated" in record

    # Verify confidence was calculated
    mock_calc_conf.assert_called_once_with(record)

    # Verify no report/alert sent yet
    mock_inventory_system.report_inventory_issue.assert_not_awaited()
    mock_alert_system.send_alert.assert_not_awaited()

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_calculate_discrepancy_confidence')
async def test_handle_inventory_discrepancy_subsequent_detection(
    mock_calc_conf, sensor_processor: SensorDataProcessor,
    mock_inventory_system: AsyncMock, mock_alert_system: AsyncMock
):
    """Test subsequent detection updates count/timestamp/sources."""
    location = "A.1"
    product_id = "P1"
    discrepancy_type = "missing"
    source1 = "rfid"
    source2 = "camera"
    details1 = {"info": "detail1"}
    details2 = {"extra": "detail2"}

    # Simulate low confidence initially
    mock_calc_conf.return_value = 0.7

    # First detection
    await sensor_processor._handle_inventory_discrepancy(
        location, [product_id], discrepancy_type, source1, details1
    )
    discrepancy_key = f"{product_id}:{location}:{discrepancy_type}"
    record1 = sensor_processor.discrepancies[discrepancy_key].copy()

    # Ensure some time passes for timestamp check
    await asyncio.sleep(0.01)

    # Second detection from different source
    await sensor_processor._handle_inventory_discrepancy(
        location, [product_id], discrepancy_type, source2, details2
    )

    record2 = sensor_processor.discrepancies[discrepancy_key]
    assert record2["detection_count"] == 2
    assert record2["sources"] == [source1, source2]
    assert record2["last_updated"] > record1["last_updated"]
    assert record2["first_detected"] == record1["first_detected"]
    assert record2["details"] == {**details1, **details2} # Details should merge/update

    # Verify confidence calculated again
    assert mock_calc_conf.call_count == 2
    mock_calc_conf.assert_called_with(record2)

    # Verify no report/alert sent yet (still below threshold)
    mock_inventory_system.report_inventory_issue.assert_not_awaited()
    mock_alert_system.send_alert.assert_not_awaited()

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_calculate_discrepancy_confidence')
async def test_handle_inventory_discrepancy_reaches_confidence_threshold(
    mock_calc_conf, sensor_processor: SensorDataProcessor,
    mock_inventory_system: AsyncMock, mock_alert_system: AsyncMock
):
    """Test reaching confidence threshold triggers report/alert."""
    location = "B.2"
    product_id = "P2"
    discrepancy_type = "out_of_stock"
    source = "smart_shelf"
    details = {"expected": 5, "estimated": 0}

    # Simulate confidence reaching threshold on first detection
    mock_calc_conf.return_value = 0.95

    await sensor_processor._handle_inventory_discrepancy(
        location, [product_id], discrepancy_type, source, details
    )

    discrepancy_key = f"{product_id}:{location}:{discrepancy_type}"
    record = sensor_processor.discrepancies[discrepancy_key]

    # Verify report issue called
    mock_inventory_system.report_inventory_issue.assert_awaited_once_with(
        sensor_processor.store_id, product_id, location,
        discrepancy_type, details, 0.95
    )
    # Verify alert called for out_of_stock
    mock_alert_system.send_alert.assert_awaited_once()
    alert_args = mock_alert_system.send_alert.call_args.kwargs
    assert alert_args["alert_type"] == "inventory"
    assert alert_args["priority"] == "high" # Confidence >= 0.95
    assert alert_args["location"] == location
    assert alert_args["details"]["product_id"] == product_id
    assert alert_args["details"]["issue"] == "out_of_stock"
    assert alert_args["details"]["confidence"] == 0.95

@pytest.mark.asyncio
@patch.object(SensorDataProcessor, '_calculate_discrepancy_confidence', return_value=0.8) # Confidence below threshold
async def test_handle_inventory_discrepancy_reaches_count_threshold(
    mock_calc_conf, sensor_processor: SensorDataProcessor,
    mock_inventory_system: AsyncMock, mock_alert_system: AsyncMock
):
    """Test reaching detection count threshold triggers report/alert."""
    location = "C.3"
    product_id = "P3"
    discrepancy_type = "low_stock"
    source = "rfid"
    details = {"info": "some details"}

    # Simulate 3 detections
    await sensor_processor._handle_inventory_discrepancy(location, [product_id], discrepancy_type, source, details)
    await sensor_processor._handle_inventory_discrepancy(location, [product_id], discrepancy_type, source, details)
    await sensor_processor._handle_inventory_discrepancy(location, [product_id], discrepancy_type, source, details)

    discrepancy_key = f"{product_id}:{location}:{discrepancy_type}"
    record = sensor_processor.discrepancies[discrepancy_key]
    assert record["detection_count"] == 3

    # Verify report issue called (count >= 3)
    mock_inventory_system.report_inventory_issue.assert_awaited_once_with(
        sensor_processor.store_id, product_id, location,
        discrepancy_type, details, 0.8 # Confidence value passed
    )
    # Verify alert NOT called for low_stock
    mock_alert_system.send_alert.assert_not_awaited()

# --- Test Confidence Calculation --- #

def test_calculate_discrepancy_confidence(sensor_processor: SensorDataProcessor):
    """Test the confidence calculation logic."""
    # Base confidence = 0.5
    # 1 source (rfid=0.85) -> + (0.85-0.7)*0.5 = 0.075
    # Count 1 -> + 0
    record1 = {"sources": ["rfid"], "detection_count": 1}
    assert sensor_processor._calculate_discrepancy_confidence(record1) == pytest.approx(0.5 + 0.075)

    # 2 sources (rfid=0.85, shelf=0.75) -> +0.15 + (0.85-0.7)*0.5 + (0.75-0.7)*0.5 = 0.15 + 0.075 + 0.025 = 0.25
    # Count 3 -> + 0.1
    record2 = {"sources": ["rfid", "smart_shelf"], "detection_count": 3}
    assert sensor_processor._calculate_discrepancy_confidence(record2) == pytest.approx(0.85)

    # 3 sources (rfid=0.85, shelf=0.75, cv=0.80) -> +0.3 + 0.075 + 0.025 + (0.8-0.7)*0.5 = 0.3 + 0.075 + 0.025 + 0.05 = 0.45
    # Count 5 -> + 0.2
    record3 = {"sources": ["rfid", "smart_shelf", "computer_vision"], "detection_count": 5}
    # Expected = 0.5 + 0.45 + 0.2 = 1.15 -> Capped at 0.99
    assert sensor_processor._calculate_discrepancy_confidence(record3) == pytest.approx(0.99)

# Placeholder tests for maintenance
# def test_clean_old_discrepancies(): ... 