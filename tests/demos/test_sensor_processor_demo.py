import asyncio
import runpy
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_printed_import_region_defines_sensor_message_without_missing_forward_reference(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastapi", SimpleNamespace(FastAPI=object, WebSocket=object))
    path = Path(__file__).resolve().parents[2] / "demos/sensor_processor_demo.py"
    source = path.read_text()
    region = source.split("# region book:sensor-processor-imports\n", 1)[1]
    region = region.split("# endregion book:sensor-processor-imports", 1)[0]
    module = ModuleType("book_sensor_imports")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = module.__dict__
    exec(compile(region, "sensor-processor-imports", "exec", dont_inherit=True), namespace)
    message = namespace["SensorMessage"].from_dict({"sensor_id": "S1"})
    assert message.sensor_id == "S1"


def test_sensor_message_envelope_timestamp_is_used_without_mutating_payload(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastapi", SimpleNamespace(FastAPI=object, WebSocket=object))
    path = Path(__file__).resolve().parents[2] / "demos/sensor_processor_demo.py"
    module = runpy.run_path(str(path))
    processor_type = module["SensorDataProcessor"]
    processor = processor_type.__new__(processor_type)
    processor.recent_readings = {}
    timestamp = datetime.now().isoformat()
    message = module["SensorMessage"](sensor_id="S1", sensor_type="unknown", timestamp=timestamp)
    asyncio.run(processor.process_sensor_message(message))
    assert processor.recent_readings == {"S1": [{"timestamp": timestamp}]}
    assert message.payload == {}
