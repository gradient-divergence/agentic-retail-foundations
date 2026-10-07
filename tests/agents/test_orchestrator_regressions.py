import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from agents.orchestrator import MasterOrchestrator
from models.enums import AgentType
from models.events import RetailEvent
from utils.event_bus import EventBus


@pytest.mark.parametrize("tz", [None, timezone.utc])
def test_stalled_order_check_accepts_stored_event_datetime(tz):
    orchestrator = MasterOrchestrator("master", EventBus())
    event = RetailEvent(
        event_type="order.created",
        payload={"order_id": "order"},
        source=AgentType.CUSTOMER,
        timestamp=datetime.now(tz) - timedelta(hours=1),
    )
    asyncio.run(orchestrator.handle_order_event(event))
    orchestrator.publish_event = AsyncMock()
    asyncio.run(orchestrator._check_for_stalled_orders())
    assert orchestrator.publish_event.await_args.args[0] == "order.stalled_alert"


def test_stalled_event_is_routed_to_recovery():
    bus = EventBus()
    orchestrator = MasterOrchestrator("master", bus)
    orchestrator._apply_recovery_strategy = AsyncMock()
    event = RetailEvent(event_type="order.stalled", payload={"order_id": "order"}, source=AgentType.SYSTEM)
    asyncio.run(bus.publish(event))
    orchestrator._apply_recovery_strategy.assert_awaited_once_with("order", event)
