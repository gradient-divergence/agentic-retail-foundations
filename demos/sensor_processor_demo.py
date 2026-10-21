"""Book sensor entry point; canonical validated processing lives in agents.sensor."""

from agents.sensor import SensorDataProcessor, SensorMessage

__all__ = ["SensorDataProcessor", "SensorMessage"]


async def run_shelf_recovery() -> dict:
    """Two current shelf readings update observations, never the inventory ledger."""
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    writes = []

    async def observe(*args, **kwargs):
        pass

    async def write(*args, **kwargs):
        writes.append(args)

    inventory = SimpleNamespace(report_inventory_issue=observe, update_product_quantity=write)
    processor = SensorDataProcessor("store-1", inventory, SimpleNamespace(send_alert=observe))
    now = datetime.now(timezone.utc)
    reading = dict(
        sensor_id="shelf-1",
        sensor_type="smart_shelf",
        timestamp=now.isoformat(),
        location={"zone": "Z", "section": "A"},
        shelf_id="1",
        confidence=0.9,
        expected_weight_grams=1000,
        tare_weight_grams=0,
        product_info={"product_id": "P", "unit_weight_grams": 100},
    )
    await processor.process_sensor_message(
        SensorMessage.from_dict({**reading, "event_id": "empty", "current_weight_grams": 0})
    )
    initial = processor.product_state["P:Z.A.1"]["estimated_units"]
    await processor.process_sensor_message(
        SensorMessage.from_dict(
            {
                **reading,
                "event_id": "recovered",
                "timestamp": (now + timedelta(seconds=1)).isoformat(),
                "current_weight_grams": 1000,
            }
        )
    )
    return dict(
        initial_units=initial,
        recovered_units=processor.product_state["P:Z.A.1"]["estimated_units"],
        open_discrepancies=len(processor.discrepancies),
        ledger_writes=len(writes),
    )


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(run_shelf_recovery()))
