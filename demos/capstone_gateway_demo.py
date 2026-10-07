from __future__ import annotations

import asyncio

from pydantic import BaseModel

from capstone.event_bus import EventBus
from capstone.orchestrator import CapstoneOrchestrator
from capstone.policies import PolicyEvaluator
from capstone.schemas import EventEnvelope, ToolCall
from capstone.tools import ToolSpec
from capstone.tracing import TraceContext


class InventoryReservation(BaseModel):
    sku: str
    quantity: int
    status: str


async def handle_order_created(event: EventEnvelope) -> None:
    print(f"Event received: {event.event_type} for order {event.payload['order_id']}")


def reserve_inventory(sku: str, quantity: int) -> InventoryReservation:
    return InventoryReservation(sku=sku, quantity=quantity, status="reserved")


async def main() -> None:
    bus = EventBus()
    bus.subscribe("order.created", handle_order_created)

    trace = TraceContext(route="capstone/order", tenant_id="retail-demo")
    event = EventEnvelope(
        event_type="order.created",
        payload={"order_id": "ORDER-1001", "sku": "SKU-123", "quantity": 2},
        trace_id=trace.trace_id,
    )
    await bus.publish(event)

    tool = ToolSpec(
        name="reserve_inventory",
        description="Reserve inventory for an order.",
        permission="write",
        schema={
            "type": "object",
            "properties": {
                "sku": {"type": "string", "minLength": 1},
                "quantity": {"type": "integer", "minimum": 1},
            },
            "required": ["sku", "quantity"],
        },
        handler=reserve_inventory,
    )
    gateway = CapstoneOrchestrator([tool], PolicyEvaluator())
    call = ToolCall(name="reserve_inventory", args={"sku": "SKU-123", "quantity": 2})
    result, audit = gateway.handle_tool_call(call, trace)

    print("Tool result:", result.model_dump())
    print("Audit record:", audit.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
