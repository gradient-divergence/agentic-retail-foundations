import asyncio

import pytest

from capstone.event_bus import EventBus as CapstoneEventBus
from capstone.schemas import EventEnvelope
from models.enums import AgentType
from models.events import RetailEvent
from utils.event_bus import EventBus


@pytest.fixture(params=["retail", "capstone"])
def bus_event(request):
    if request.param == "retail":
        return EventBus(), RetailEvent(
            event_type="received", payload={"quantity": 2}, source=AgentType.SYSTEM
        )
    return CapstoneEventBus(), EventEnvelope(event_type="received", payload={"quantity": 2}, trace_id="trace")


def test_replay_and_concurrent_publish_deliver_once_per_subscriber(bus_event):
    bus, event = bus_event
    deliveries = []

    async def first(received):
        await asyncio.sleep(0)
        deliveries.append(("first", received.event_id))

    async def second(received):
        deliveries.append(("second", received.event_id))

    async def scenario():
        bus.subscribe(event.event_type, first)
        bus.subscribe(event.event_type, first)
        bus.subscribe(event.event_type, second)
        await asyncio.gather(bus.publish(event), bus.publish(event.model_copy()))
        await bus.publish(event)

    asyncio.run(scenario())
    assert deliveries.count(("first", event.event_id)) == 1
    assert deliveries.count(("second", event.event_id)) == 1


def test_replay_retries_only_failed_subscriber(bus_event):
    bus, event = bus_event
    calls = {"ok": 0, "retry": 0}

    async def ok(_):
        calls["ok"] += 1

    async def retry(_):
        calls["retry"] += 1
        if calls["retry"] == 1:
            raise RuntimeError("temporary failure")

    async def scenario():
        bus.subscribe(event.event_type, ok)
        bus.subscribe(event.event_type, retry)
        await bus.publish(event)
        await bus.publish(event)
        await bus.publish(event)

    asyncio.run(scenario())
    assert calls == {"ok": 1, "retry": 2}


def test_subscription_changes_do_not_corrupt_error_reporting(bus_event, caplog):
    bus, event = bus_event

    async def removes_itself(_):
        bus.subscribers[event.event_type].remove(removes_itself)
        raise RuntimeError("expected handler failure")

    bus.subscribe(event.event_type, removes_itself)
    asyncio.run(bus.publish(event))
    assert "expected handler failure" in caplog.text
