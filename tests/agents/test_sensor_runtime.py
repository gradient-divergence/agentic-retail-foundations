import asyncio
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[2]


class FakeFastAPI:
    def websocket(self, path):
        return lambda handler: handler

    def post(self, path):
        return lambda handler: handler


@pytest.fixture(params=["agents/sensor.py", "demos/sensor_processor_demo.py"])
def sensor_module(request, monkeypatch):
    # Keep route setup separate from the async lifecycle checks.
    monkeypatch.setitem(sys.modules, "fastapi", SimpleNamespace(FastAPI=FakeFastAPI, WebSocket=object))
    return runpy.run_path(str(ROOT / request.param))


@pytest.mark.parametrize("failure", [False, True])
def test_server_shares_loop_and_cleans_maintenance_task(sensor_module, monkeypatch, failure):
    async def check():
        loop = asyncio.get_running_loop()
        processor = sensor_module["SensorDataProcessor"]("S1", AsyncMock(), AsyncMock())
        events = []

        async def maintenance():
            events.append("maintenance started")
            try:
                await asyncio.Event().wait()
            finally:
                events.append("maintenance stopped")

        class Server:
            def __init__(self, config):
                assert config.app is processor.app
                assert config.host == "0.0.0.0"
                assert config.port == 8080

            async def serve(self):
                assert asyncio.get_running_loop() is loop
                await asyncio.sleep(0)
                events.append("server finished")
                if failure:
                    raise RuntimeError("server failed")

        def blocking_run(*args, **kwargs):
            raise AssertionError("run() must use the server on the caller's async loop")

        processor._run_maintenance_loop = maintenance
        monkeypatch.setitem(
            sys.modules,
            "uvicorn",
            SimpleNamespace(
                Config=lambda app, **kwargs: SimpleNamespace(app=app, **kwargs),
                Server=Server,
                run=blocking_run,
            ),
        )
        if failure:
            with pytest.raises(RuntimeError, match="server failed"):
                await processor.run()
        else:
            await processor.run()
        assert events == ["maintenance started", "server finished", "maintenance stopped"]
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(check())


def test_server_cancellation_stops_maintenance(sensor_module, monkeypatch):
    async def check():
        processor = sensor_module["SensorDataProcessor"]("S1", AsyncMock(), AsyncMock())
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def maintenance():
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        class Server:
            def __init__(self, config):
                pass

            async def serve(self):
                started.set()
                await asyncio.Event().wait()

        processor._run_maintenance_loop = maintenance
        monkeypatch.setitem(
            sys.modules,
            "uvicorn",
            SimpleNamespace(Config=lambda app, **kwargs: SimpleNamespace(app=app, **kwargs), Server=Server),
        )
        task = asyncio.create_task(processor.run())
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stopped.is_set()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(check())


def test_timezone_aware_sensor_timestamp_is_accepted(sensor_module):
    processor = sensor_module["SensorDataProcessor"]("S1", AsyncMock(), AsyncMock())
    reading = {
        "sensor_id": "SENSOR1",
        "sensor_type": "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    message = (
        sensor_module["SensorMessage"].from_dict(reading) if "SensorMessage" in sensor_module else reading
    )
    asyncio.run(processor.process_sensor_message(message))
    assert processor.recent_readings == {"SENSOR1": [reading]}


@pytest.mark.parametrize("timestamp", [None, "bad timestamp"])
def test_invalid_sensor_timestamp_does_not_poison_buffer(sensor_module, timestamp):
    processor = sensor_module["SensorDataProcessor"]("S1", AsyncMock(), AsyncMock())
    reading = {"sensor_id": "SENSOR1", "sensor_type": "unknown", "timestamp": timestamp}
    message = (
        sensor_module["SensorMessage"].from_dict(reading) if "SensorMessage" in sensor_module else reading
    )
    asyncio.run(processor.process_sensor_message(message))
    assert processor.recent_readings == {}


def test_invalid_unit_weight_does_not_update_inventory(sensor_module):
    inventory = AsyncMock()
    processor = sensor_module["SensorDataProcessor"]("S1", inventory, AsyncMock())
    reading = {
        "current_weight_grams": 500,
        "expected_weight_grams": 1000,
        "product_info": {"product_id": "P1", "unit_weight_grams": 0},
    }
    asyncio.run(processor._process_smart_shelf_reading(reading))
    inventory.update_product_quantity.assert_not_awaited()
