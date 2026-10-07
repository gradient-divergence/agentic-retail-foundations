# region book:sensor-processor-imports
from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel, Field, JsonValue


class SensorMessage(BaseModel):
    sensor_id: str | None = None
    sensor_type: str | None = None
    timestamp: str | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> SensorMessage:
        return cls(
            sensor_id=str(data.get("sensor_id")) if data.get("sensor_id") else None,
            sensor_type=str(data.get("sensor_type")) if data.get("sensor_type") else None,
            timestamp=str(data.get("timestamp")) if data.get("timestamp") else None,
            payload=data,
        )


class SensorBatchPayload(BaseModel):
    readings: list[dict[str, JsonValue]] = Field(default_factory=list)


class SensorBatchResult(BaseModel):
    status: str
    count: int


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
                print(f"WebSocket error: {exc}")
            finally:
                self.active_connections.remove(websocket)

        @self.app.post("/sensor-batch")
        async def sensor_batch_endpoint(data: SensorBatchPayload) -> SensorBatchResult:
            """Endpoint for batch uploads of sensor data"""
            for reading in data.readings:
                await self.process_sensor_message(SensorMessage.from_dict(reading))
            return SensorBatchResult(status="processed", count=len(data.readings))

    # endregion book:sensor-processor-setup-routes

    # region book:sensor-processor-process-message
    async def process_sensor_message(self, message: SensorMessage):
        """Process an incoming sensor reading"""
        payload = {**message.payload, "timestamp": message.timestamp or message.payload.get("timestamp")}
        sensor_id = message.sensor_id or payload.get("sensor_id")
        sensor_type = message.sensor_type or str(payload.get("sensor_type"))
        if not isinstance(sensor_id, str):
            return
        try:
            datetime.fromisoformat(payload.get("timestamp", ""))
        except (TypeError, ValueError):
            return
        if sensor_id not in self.recent_readings:
            self.recent_readings[sensor_id] = []

        self.recent_readings[sensor_id].append(payload)
        cutoff = datetime.now().astimezone() - timedelta(hours=24)
        self.recent_readings[sensor_id] = [
            reading
            for reading in self.recent_readings[sensor_id]
            if datetime.fromisoformat(reading["timestamp"]).astimezone() > cutoff
        ]

        if sensor_type == "rfid":
            await self._process_rfid_reading(payload)
        elif sensor_type == "smart_shelf":
            await self._process_smart_shelf_reading(payload)
        elif sensor_type == "environmental":
            await self._process_environmental_reading(payload)
        elif sensor_type == "digital_price_tag":
            await self._process_price_tag_reading(payload)

    # endregion book:sensor-processor-process-message

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
            await self._handle_inventory_discrepancy(expected_location, list(missing_ids), "missing", "rfid")

        unexpected_ids = detected_ids - expected_ids
        if unexpected_ids:
            await self._handle_inventory_discrepancy(
                expected_location, list(unexpected_ids), "unexpected", "rfid"
            )

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
        shelf_id = message.get("shelf_id")
        location = message.get("location", {})
        current_weight = message.get("current_weight_grams")
        expected_weight = message.get("expected_weight_grams")
        product_info = message.get("product_info", {})
        unit_weight = product_info.get("unit_weight_grams", 1)
        if (
            not isinstance(current_weight, int | float)
            or not isinstance(expected_weight, int | float)
            or not isinstance(unit_weight, int | float)
            or unit_weight <= 0
        ):
            return

        weight_diff = abs(current_weight - expected_weight)
        weight_threshold = unit_weight * 0.5

        if weight_diff > weight_threshold:
            estimated_units = max(0, round(current_weight / unit_weight))
            expected_units = max(0, round(expected_weight / unit_weight))

            if estimated_units < expected_units:
                discrepancy_type = "low_stock" if estimated_units > 0 else "out_of_stock"
                await self._handle_inventory_discrepancy(
                    f"{location.get('zone')}.{location.get('section')}.{shelf_id}",
                    [product_info.get("product_id")],
                    discrepancy_type,
                    "smart_shelf",
                    {
                        "expected_units": expected_units,
                        "estimated_units": estimated_units,
                        "confidence": 0.9,
                    },
                )

            # endregion book:sensor-processor-smart-shelf

            # region book:sensor-processor-smart-shelf-update
            await self.inventory_system.update_product_quantity(
                self.store_id,
                product_info.get("product_id"),
                estimated_units,
                f"{location.get('zone')}.{location.get('section')}.{shelf_id}",
                message.get("timestamp"),
                source="smart_shelf",
            )

    # endregion book:sensor-processor-smart-shelf-update

    # region book:sensor-processor-environmental
    async def _process_environmental_reading(self, message: dict[str, JsonValue]):
        """Process environmental sensor data"""
        sensor_type = message.get("environmental_type")
        value = message.get("value")
        unit = message.get("unit")
        location = message.get("location", {})
        threshold_exceeded = False
        alert_priority = "info"

        if sensor_type == "temperature":
            zone_type = location.get("zone_type", "ambient")
            if zone_type == "refrigerated" and value > 5:
                threshold_exceeded = True
                alert_priority = "high" if value > 8 else "medium"
            elif zone_type == "frozen" and value > -15:
                threshold_exceeded = True
                alert_priority = "high" if value > -10 else "medium"
        elif sensor_type == "humidity":
            if location.get("zone_type") == "produce" and (value < 80 or value > 95):
                threshold_exceeded = True
                alert_priority = "medium"

        # endregion book:sensor-processor-environmental

        # region book:sensor-processor-environmental-alert
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

    # endregion book:sensor-processor-environmental-alert

    # region book:sensor-processor-price-tag
    async def _process_price_tag_reading(self, message: dict[str, JsonValue]):
        """Process digital price tag status updates"""
        tag_id = message.get("tag_id")
        product_id = message.get("product_id")
        price_displayed = message.get("price_displayed")
        battery_level = message.get("battery_level", 100)
        location = message.get("location", {})

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

        # endregion book:sensor-processor-price-tag

        # region book:sensor-processor-price-verify
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

    # endregion book:sensor-processor-price-verify

    # region book:sensor-processor-discrepancy-handler
    async def _handle_inventory_discrepancy(
        self,
        location: str,
        product_ids: list[str],
        discrepancy_type: str,
        source: str,
        details: dict[str, JsonValue] | None = None,
    ):
        """Handle detected inventory discrepancies"""
        timestamp = datetime.now().isoformat()
        for product_id in product_ids:
            discrepancy_key = f"{product_id}:{location}:{discrepancy_type}"
            if discrepancy_key not in self.discrepancies:
                self.discrepancies[discrepancy_key] = {
                    "product_id": product_id,
                    "location": location,
                    "type": discrepancy_type,
                    "first_detected": timestamp,
                    "last_updated": timestamp,
                    "detection_count": 1,
                    "sources": [source],
                    "details": details or {},
                }
            else:
                record = self.discrepancies[discrepancy_key]
                record["last_updated"] = timestamp
                record["detection_count"] += 1
                if source not in record["sources"]:
                    record["sources"].append(source)
                if details:
                    record["details"].update(details)

            # endregion book:sensor-processor-discrepancy-handler

            # region book:sensor-processor-discrepancy-action
            record = self.discrepancies[discrepancy_key]
            confidence_score = self._calculate_discrepancy_confidence(record)
            if confidence_score >= 0.9 or record["detection_count"] >= 3:
                if discrepancy_type in ["missing", "out_of_stock", "low_stock"]:
                    await self.inventory_system.report_inventory_issue(
                        self.store_id,
                        product_id,
                        location,
                        discrepancy_type,
                        record["details"],
                        confidence_score,
                    )
                    if discrepancy_type == "out_of_stock":
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
        print(f"Requesting price update for tag {tag_id}, product {product_id} to ${price}")

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
