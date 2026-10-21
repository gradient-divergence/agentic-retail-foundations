# region book:sensor-processor-imports
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel, Field, JsonValue, ValidationError

from models.sensors import READING, is_finite_number

logger = logging.getLogger(__name__)


class SensorMessage(BaseModel):
    sensor_id: str | None = None
    sensor_type: str | None = None
    timestamp: str | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> SensorMessage:
        fields = ("sensor_id", "sensor_type", "timestamp")
        return cls(**{k: data.get(k) if isinstance(data.get(k), str) else None for k in fields}, payload=data)


class SensorBatchPayload(BaseModel):
    readings: list[dict[str, JsonValue]] = Field(default_factory=list)


class SensorBatchResult(BaseModel):
    status: str
    count: int
    rejected: int


class SensorDataProcessor:
    def __init__(
        self,
        store_id: str,
        inventory_system,
        alert_system,
        confidence_thresholds: dict[str, float] | None = None,
    ):
        self.store_id = store_id
        self.inventory_system = inventory_system
        self.alert_system = alert_system
        self.confidence_thresholds = confidence_thresholds or {
            "rfid": 0.85,
            "smart_shelf": 0.75,
            "computer_vision": 0.80,
        }
        self.recent_readings = {}
        self.seen_events = {}
        self.processing_lock = asyncio.Lock()
        self.last_event_time = {}
        self.product_state = {}
        self.discrepancies = {}
        self.app = FastAPI()
        self.setup_routes()
        self.active_connections = set()

    # endregion book:sensor-processor-imports

    # region book:sensor-processor-setup-routes
    def setup_routes(self):
        """Configure API endpoints for sensor data ingestion"""

        @self.app.websocket("/sensor-stream")
        async def sensor_stream_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.add(websocket)

            try:
                while True:
                    data = await websocket.receive_text()
                    await self.process_sensor_message(SensorMessage.from_dict(json.loads(data)))
            except Exception as exc:
                logger.info(f"WebSocket error: {exc}")
            finally:
                self.active_connections.remove(websocket)

        @self.app.post("/sensor-batch")
        async def sensor_batch_endpoint(data: SensorBatchPayload) -> SensorBatchResult:
            accepted = sum(
                [await self.process_sensor_message(SensorMessage.from_dict(r)) for r in data.readings]
            )
            rejected = len(data.readings) - accepted
            return SensorBatchResult(status="processed", count=accepted, rejected=rejected)

    # endregion book:sensor-processor-setup-routes

    # region book:sensor-processor-process-message
    async def process_sensor_message(self, message: SensorMessage | dict[str, JsonValue]):
        """Process the validated envelope or the public JSON reading format."""
        reading = self._validate_sensor_message(message)
        if reading is None:
            return False
        async with self.processing_lock:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(hours=24)
            sensor_id, sensor_type = reading.sensor_id, reading.sensor_type
            if (
                reading.timestamp <= cutoff
                or reading.timestamp > now + timedelta(minutes=1)
                or reading.timestamp <= self.last_event_time.get(sensor_id, cutoff)
            ):
                return False
            self.seen_events = {key: time for key, time in self.seen_events.items() if time > cutoff}
            if reading.event_id in self.seen_events or reading.confidence < self.confidence_thresholds.get(
                sensor_type, 0.75
            ):
                return False
            payload = reading.model_dump(mode="json")
            if sensor_type == "rfid":
                await self._process_rfid_reading(payload)
            elif sensor_type == "smart_shelf":
                await self._process_smart_shelf_reading(payload)
            elif sensor_type == "environmental":
                await self._process_environmental_reading(payload)
            elif sensor_type == "digital_price_tag":
                await self._process_price_tag_reading(payload)
            # A failed connector call leaves this event and timestamp eligible for retry.
            self.recent_readings[sensor_id] = [payload]
            self.seen_events[reading.event_id] = reading.timestamp
            self.last_event_time[sensor_id] = reading.timestamp
            return True

    # endregion book:sensor-processor-process-message

    @staticmethod
    def _validate_sensor_message(message: SensorMessage | dict[str, JsonValue]):
        if isinstance(message, dict):
            message = SensorMessage.from_dict(message)
        payload = {
            **message.payload,
            "sensor_id": message.sensor_id,
            "sensor_type": message.sensor_type,
            "timestamp": message.timestamp,
        }
        try:
            return READING.validate_python(payload)
        except ValidationError as exc:
            logger.warning(
                "sensor.reading_rejected", extra={"invalid_fields": [error["loc"] for error in exc.errors()]}
            )
            return None

    # region book:sensor-processor-rfid
    async def _process_rfid_reading(self, message: dict[str, JsonValue]):
        """Process RFID reader data"""
        reader_location = message.get("location", {})
        confidence = message.get("confidence", 1.0)
        if confidence < self.confidence_thresholds.get("rfid", 0.85):
            return
        detected_products = message.get("detected_products", [])
        detected_ids = set(item.get("product_id") for item in detected_products)
        expected_location = f"{reader_location.get('zone')}.{reader_location.get('section')}"
        expected_ids = await self.inventory_system.get_expected_products(self.store_id, expected_location)
        missing_ids = expected_ids - detected_ids
        if missing_ids:
            await self._handle_inventory_discrepancy(
                expected_location, list(missing_ids), "missing", "rfid", event_id=message.get("event_id")
            )

        unexpected_ids = detected_ids - expected_ids
        if unexpected_ids:
            await self._handle_inventory_discrepancy(
                expected_location,
                list(unexpected_ids),
                "unexpected",
                "rfid",
                event_id=message.get("event_id"),
            )

        # Each RFID message is a complete SKU-presence snapshot for this location.
        for key, record in list(self.discrepancies.items()):
            if (
                record["location"] == expected_location
                and record["type"] in ("missing", "unexpected")
                and record["product_id"] not in missing_ids | unexpected_ids
            ):
                await self._clear_discrepancy(key, {}, confidence)

        await self.inventory_system.update_product_locations(
            self.store_id,
            [
                {
                    "product_id": product.get("product_id"),
                    "location": expected_location,
                    "last_seen": message.get("timestamp"),
                    "confidence": confidence,
                }
                for product in detected_products
            ],
        )

    # endregion book:sensor-processor-rfid

    # region book:sensor-processor-smart-shelf
    async def _process_smart_shelf_reading(self, message: dict[str, JsonValue]):
        """Process weight-sensing shelf data"""
        if message.get("confidence", 0) < self.confidence_thresholds.get("smart_shelf", 0.75):
            return
        product = message["product_info"]
        tare, unit_weight = message["tare_weight_grams"], product["unit_weight_grams"]
        estimated_units = max(0, round((message["current_weight_grams"] - tare) / unit_weight))
        expected_units = max(0, round((message["expected_weight_grams"] - tare) / unit_weight))
        location = message["location"]
        address = f"{location['zone']}.{location['section']}.{message['shelf_id']}"
        if estimated_units < expected_units:
            await self._handle_inventory_discrepancy(
                address,
                [product["product_id"]],
                "low_stock" if estimated_units else "out_of_stock",
                "smart_shelf",
                {"expected_units": expected_units, "estimated_units": estimated_units},
                event_id=message.get("event_id"),
            )
        else:
            key = f"{product['product_id']}:{address}"
            if key in self.discrepancies:
                await self._clear_discrepancy(
                    key, {"estimated_units": estimated_units}, message["confidence"]
                )
        # endregion book:sensor-processor-smart-shelf

        # region book:sensor-processor-smart-shelf-update
        # Weight counting assumes known tare and homogeneous units; this is an observation.
        self.product_state[f"{product['product_id']}:{address}"] = {
            "estimated_units": estimated_units,
            "timestamp": message["timestamp"],
            "source": "smart_shelf",
        }

    # endregion book:sensor-processor-smart-shelf-update

    # region book:sensor-processor-environmental-joined
    async def _process_environmental_reading(self, message: dict[str, JsonValue]):
        """Process environmental sensor data"""
        sensor_type = message.get("environmental_type")
        value, unit = self._environmental_value(message)
        if value is None:
            return
        location = message.get("location", {})
        threshold_exceeded = False
        alert_priority = "info"

        if sensor_type == "temperature":
            # Illustrative Celsius thresholds; deployment needs product-specific operating limits.
            limits = {"refrigerated": (5, 8), "frozen": (-15, -10)}.get(location.get("zone_type", "ambient"))
            if limits and value > limits[0]:
                threshold_exceeded = True
                alert_priority = "high" if value > limits[1] else "medium"
        elif sensor_type == "humidity":
            if location.get("zone_type") == "produce" and (value < 80 or value > 95):
                threshold_exceeded = True
                alert_priority = "medium"

        if threshold_exceeded:
            await self.alert_system.send_alert(
                alert_type="environmental",
                priority=alert_priority,
                location=f"{location.get('zone')}.{location.get('section')}",
                details={
                    "sensor_type": sensor_type,
                    "value": value,
                    "unit": unit,
                    "threshold_exceeded": True,
                },
            )
            if sensor_type == "temperature" and location.get("zone_type") in [
                "refrigerated",
                "frozen",
                "produce",
            ]:
                await self.inventory_system.flag_products_for_quality_check(
                    self.store_id,
                    location=f"{location.get('zone')}.{location.get('section')}",
                    reason=f"Temperature threshold exceeded: {value}{unit}",
                    timestamp=message.get("timestamp"),
                )

    # endregion book:sensor-processor-environmental-joined

    @staticmethod
    def _environmental_value(message: dict[str, JsonValue]):
        value, unit = message.get("value"), message.get("unit")
        if not is_finite_number(value):
            logger.warning("Invalid or missing numeric value")
            return None, unit
        if message.get("environmental_type") == "temperature" and unit == "F":
            value = (value - 32) * 5 / 9
            unit = "C"
        return value, unit

    # region book:sensor-processor-price-tag-joined
    async def _process_price_tag_reading(self, message: dict[str, JsonValue]):
        """Process digital price tag status updates"""
        tag_id = message.get("tag_id")
        product_id = message.get("product_id")
        price_displayed = message.get("price_displayed")
        battery_level = message.get("battery_level", 100)
        location = message.get("location", {})

        if not is_finite_number(price_displayed):
            logger.warning("Could not compare prices")
            return
        if battery_level < 20:
            await self.alert_system.send_alert(
                alert_type="maintenance",
                priority="low",
                location=f"{location.get('zone')}.{location.get('section')}",
                details={
                    "device_type": "digital_price_tag",
                    "device_id": tag_id,
                    "battery_level": battery_level,
                    "product_id": product_id,
                },
            )

        expected_price = await self.inventory_system.get_current_price(self.store_id, product_id)
        if price_displayed != expected_price:
            await self.alert_system.send_alert(
                alert_type="price_discrepancy",
                priority="medium",
                location=f"{location.get('zone')}.{location.get('section')}",
                details={
                    "product_id": product_id,
                    "displayed_price": price_displayed,
                    "expected_price": expected_price,
                    "tag_id": tag_id,
                },
            )
            await self._request_price_tag_update(tag_id, product_id, expected_price)

    # endregion book:sensor-processor-price-tag-joined

    async def _clear_discrepancy(self, key, details, confidence):
        record = self.discrepancies[key]
        if record.get("reported"):
            await self.inventory_system.report_inventory_issue(
                self.store_id, record["product_id"], record["location"], "resolved", details, confidence
            )
        del self.discrepancies[key]  # Retain the incident when recovery delivery fails.

    # region book:sensor-processor-discrepancy-handler
    async def _handle_inventory_discrepancy(
        self,
        location: str,
        product_ids: list[str],
        discrepancy_type: str,
        source: str,
        details: dict[str, JsonValue] | None = None,
        event_id: str | None = None,
    ):
        """Repeated distinct readings provide a heuristic confirmation score, not probability."""
        timestamp = datetime.now().isoformat()
        for product_id in product_ids:
            discrepancy_key = f"{product_id}:{location}"
            # A product/location has one current state, even after partial recovery.
            if (
                discrepancy_key not in self.discrepancies
                or self.discrepancies[discrepancy_key]["type"] != discrepancy_type
            ):
                self.discrepancies[discrepancy_key] = {
                    "product_id": product_id,
                    "location": location,
                    "type": discrepancy_type,
                    "first_detected": timestamp,
                    "last_updated": timestamp,
                    "detection_count": 1,
                    "sources": [source],
                    "details": details or {},
                    "event_ids": [event_id] if event_id else [],
                }
            else:
                record = self.discrepancies[discrepancy_key]
                record["last_updated"] = timestamp
                if event_id is None or event_id not in record["event_ids"]:
                    record["detection_count"] += 1
                    if event_id is not None:
                        record["event_ids"].append(event_id)
                if source not in record["sources"]:
                    record["sources"].append(source)
                record["details"].update(details or {})

            # endregion book:sensor-processor-discrepancy-handler

            # region book:sensor-processor-discrepancy-action
            record = self.discrepancies[discrepancy_key]
            confidence_score = self._calculate_discrepancy_confidence(record)
            if confidence_score >= 0.9 or record["detection_count"] >= 3:
                if discrepancy_type in ["missing", "out_of_stock", "low_stock"]:
                    if not record.get("reported"):
                        await self.inventory_system.report_inventory_issue(
                            self.store_id,
                            product_id,
                            location,
                            discrepancy_type,
                            record["details"],
                            confidence_score,
                        )
                        record["reported"] = True
                    if discrepancy_type == "out_of_stock" and not record.get("alerted"):
                        await self.alert_system.send_alert(
                            alert_type="inventory",
                            priority="high" if confidence_score >= 0.95 else "medium",
                            location=location,
                            details={
                                "product_id": product_id,
                                "issue": "out_of_stock",
                                "confidence": confidence_score,
                            },
                        )
                        record["alerted"] = True

    # endregion book:sensor-processor-discrepancy-action

    # region book:sensor-processor-discrepancy-confidence
    def _calculate_discrepancy_confidence(self, discrepancy_record: dict[str, JsonValue]) -> float:
        """Calculate confidence score for a discrepancy based on sources and frequency"""
        confidence = 0.5
        source_count = len(discrepancy_record["sources"])
        if source_count >= 3:
            confidence += 0.3
        elif source_count == 2:
            confidence += 0.15
        detection_count = discrepancy_record["detection_count"]
        if detection_count >= 5:
            confidence += 0.2
        elif detection_count >= 3:
            confidence += 0.1
        for source in discrepancy_record["sources"]:
            source_confidence = self.confidence_thresholds.get(source, 0.7)
            confidence += (source_confidence - 0.7) * 0.5
        return min(0.99, confidence)

    # endregion book:sensor-processor-discrepancy-confidence

    # region book:sensor-processor-request-price-update
    async def _request_price_tag_update(self, tag_id: str, product_id: str, price: float):
        """Request update for a digital price tag"""
        logger.info(f"Requesting price update for tag {tag_id}, product {product_id} to ${price}")

    # endregion book:sensor-processor-request-price-update

    # region book:sensor-processor-run
    async def run(self):
        """Run the main processing loop"""
        import uvicorn

        server = uvicorn.Server(uvicorn.Config(self.app, host="0.0.0.0", port=8080))
        maintenance_task = asyncio.create_task(self._run_maintenance_loop())
        try:
            await server.serve()
        finally:
            maintenance_task.cancel()
            with suppress(asyncio.CancelledError):
                await maintenance_task

    # endregion book:sensor-processor-run

    # region book:sensor-processor-maintenance-loop
    async def _run_maintenance_loop(self):
        """Run periodic maintenance tasks"""
        while True:
            await self._clean_old_discrepancies()
            await self._cross_validate_sources()
            await asyncio.sleep(300)

    # endregion book:sensor-processor-maintenance-loop

    # region book:sensor-processor-clean-discrepancies
    async def _clean_old_discrepancies(self):
        """Remove old resolved discrepancies"""
        now = datetime.now()
        to_remove = []
        for key, record in self.discrepancies.items():
            last_updated = datetime.fromisoformat(record["last_updated"])
            if (now - last_updated).total_seconds() > 86400:
                to_remove.append(key)
        for key in to_remove:
            del self.discrepancies[key]

    # endregion book:sensor-processor-clean-discrepancies

    # region book:sensor-processor-cross-validate
    async def _cross_validate_sources(self):
        """Cross-validate data between different sensor sources"""
        pass


# endregion book:sensor-processor-cross-validate
