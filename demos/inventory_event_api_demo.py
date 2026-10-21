from __future__ import annotations

import json
import logging
import uuid

# region book:inventory-event-imports
from datetime import datetime
from enum import Enum
from typing import Literal

try:
    import redis
except ModuleNotFoundError:
    raise SystemExit("Install the streaming extra: uv sync --extra streaming") from None
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field, JsonValue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inventory-service")

app = FastAPI(title="Retail Inventory Event Service")
redis_client = redis.Redis(host="redis", port=6379, db=0)
# endregion book:inventory-event-imports


# region book:inventory-event-type-enum-joined
class EventType(str, Enum):
    """Types of inventory events"""

    RECEIVED = "inventory.received"
    SOLD = "inventory.sold"
    ADJUSTED = "inventory.adjusted"
    TRANSFERRED = "inventory.transferred"
    DISPATCHED = "inventory.dispatched"
    RESERVED = "inventory.reserved"
    RELEASED = "inventory.released"


class InventoryEvent(BaseModel):
    """Base model for all inventory events"""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    product_id: str
    location_id: str
    quantity: int
    user_id: str | None = None
    reference_id: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


# endregion book:inventory-event-type-enum-joined


# region book:inventory-event-received-joined
class InventoryReceived(InventoryEvent):
    """Event for receiving inventory"""

    event_type: Literal[EventType.RECEIVED] = EventType.RECEIVED
    supplier_id: str
    purchase_order_id: str | None = None


class InventorySold(InventoryEvent):
    """POS sale or fulfillment against an order-specific reservation."""

    event_type: Literal[EventType.SOLD] = EventType.SOLD
    order_id: str
    customer_id: str | None = None
    reservation_id: str | None = None


class InventoryAdjusted(InventoryEvent):
    """Event for manual inventory adjustments"""

    event_type: Literal[EventType.ADJUSTED] = EventType.ADJUSTED
    reason_code: str
    notes: str | None = None


class InventoryTransferred(InventoryEvent):
    """Physical receipt of a previously dispatched transfer."""

    event_type: Literal[EventType.TRANSFERRED] = EventType.TRANSFERRED
    source_location_id: str
    destination_location_id: str
    transfer_id: str


# endregion book:inventory-event-received-joined


# region book:inventory-current-state-joined
class InventoryCurrentState(BaseModel):
    """Represents current inventory state (read model)"""

    product_id: str
    location_id: str
    quantity_available: int
    quantity_reserved: int
    last_updated: datetime


class InventoryStateResponse(BaseModel):
    product_id: str
    locations: dict[str, InventoryCurrentState]


inventory_state: dict[str, dict[str, InventoryCurrentState]] = {}


async def publish_event(event: InventoryEvent) -> None:
    """Publish inventory event to Redis stream"""
    try:
        event_data = event.model_dump()
        event_json = json.dumps(event_data, default=str)
        stream_key = f"streams:{event.event_type}"
        redis_client.xadd(stream_key, {"data": event_json})
        redis_client.xadd("streams:inventory.all", {"data": event_json})
        logger.info(f"Published {event.event_type} event: {event.event_id}")
    except Exception as exc:
        logger.error(f"Failed to publish event: {str(exc)}")
        raise


# endregion book:inventory-current-state-joined


class InventoryReserved(InventoryEvent):
    event_type: Literal[EventType.RESERVED] = EventType.RESERVED
    reservation_id: str
    order_id: str


class InventoryReleased(InventoryReserved):
    event_type: Literal[EventType.RELEASED] = EventType.RELEASED


class InventoryDispatched(InventoryTransferred):
    event_type: Literal[EventType.DISPATCHED] = EventType.DISPATCHED


reservations: dict[tuple[str, str, str], dict] = {}
in_transit: dict[str, dict] = {}
applied_events: dict[str, str] = {}
received_transfers: set[str] = set()


def _require_inventory(condition, message):
    if not condition:
        raise HTTPException(400, message)


def _apply_inventory_delta(event: InventoryEvent, current: InventoryCurrentState) -> None:
    """Process-local atomic projection; production needs a transactional outbox/store."""

    _require_inventory(
        type(event.quantity) is int and event.quantity != 0, "Quantity must be a nonzero integer"
    )
    if event.event_type != EventType.ADJUSTED:
        _require_inventory(event.quantity > 0, "Quantity must be positive")
    key = (event.product_id, event.location_id, getattr(event, "reservation_id", None))
    if event.event_type == EventType.RECEIVED:
        current.quantity_available += event.quantity
    elif event.event_type in {EventType.RESERVED, EventType.RELEASED}:
        _require_inventory(isinstance(event, InventoryReserved), "Reservation-linked event required")
        if event.event_type == EventType.RESERVED:
            _require_inventory(key not in reservations, "Reservation already exists")
            _require_inventory(current.quantity_available >= event.quantity, "Insufficient available stock")
            reservations[key] = dict(order_id=event.order_id, remaining=event.quantity)
            current.quantity_available -= event.quantity
            current.quantity_reserved += event.quantity
        else:
            reservation = reservations.get(key)
            _require_inventory(
                reservation
                and reservation["order_id"] == event.order_id
                and reservation["remaining"] >= event.quantity,
                "Reservation does not cover release",
            )
            reservation["remaining"] -= event.quantity
            current.quantity_reserved -= event.quantity
            current.quantity_available += event.quantity
    elif event.event_type == EventType.SOLD:
        _require_inventory(isinstance(event, InventorySold), "Sale event required")
        if event.reservation_id is not None:
            reservation = reservations.get(key)
            _require_inventory(
                reservation
                and reservation["order_id"] == event.order_id
                and reservation["remaining"] >= event.quantity,
                "Reservation does not cover this order's sale",
            )
            reservation["remaining"] -= event.quantity
            current.quantity_reserved -= event.quantity
        else:
            _require_inventory(current.quantity_available >= event.quantity, "Insufficient available stock")
            current.quantity_available -= event.quantity
    elif event.event_type == EventType.ADJUSTED:
        _require_inventory(
            current.quantity_available + event.quantity >= 0, "Adjustment exceeds available stock"
        )
        current.quantity_available += event.quantity
    elif event.event_type in {EventType.DISPATCHED, EventType.TRANSFERRED}:
        _apply_transfer(event, current)
    else:
        _require_inventory(False, "Unsupported inventory event")


def _apply_transfer(event: InventoryTransferred, current: InventoryCurrentState) -> None:
    _require_inventory(isinstance(event, InventoryTransferred), "Transfer event required")
    _require_inventory(
        event.source_location_id != event.destination_location_id
        and event.location_id == event.source_location_id,
        "Invalid transfer locations",
    )
    transfer = dict(
        product_id=event.product_id,
        source=event.source_location_id,
        destination=event.destination_location_id,
        quantity=event.quantity,
    )
    if event.event_type == EventType.DISPATCHED:
        _require_inventory(
            event.transfer_id not in in_transit and event.transfer_id not in received_transfers,
            "Transfer already dispatched",
        )
        _require_inventory(current.quantity_available >= event.quantity, "Insufficient available stock")
        current.quantity_available -= event.quantity
        in_transit[event.transfer_id] = transfer
    else:
        _require_inventory(
            in_transit.get(event.transfer_id) == transfer, "Receipt must match an in-transit transfer"
        )
        destination = inventory_state[event.product_id].setdefault(
            event.destination_location_id,
            InventoryCurrentState(
                product_id=event.product_id,
                location_id=event.destination_location_id,
                quantity_available=0,
                quantity_reserved=0,
                last_updated=event.timestamp,
            ),
        )
        destination.quantity_available += event.quantity
        destination.last_updated = event.timestamp
        del in_transit[event.transfer_id]
        received_transfers.add(event.transfer_id)


# region book:inventory-update-state-init-joined
async def update_inventory_state(event: InventoryEvent) -> None:
    # Full helpers and reservation/transfer/replay stores: demos/inventory_event_api_demo.py.
    """Validate and apply once, without awaiting between the checks and mutation."""
    prior = applied_events.get(event.event_id)
    if prior is not None:
        if prior != event.model_dump_json():
            raise HTTPException(409, "Event identity reused with changed content")
        return
    current = inventory_state.get(event.product_id, {}).get(event.location_id) or InventoryCurrentState(
        product_id=event.product_id,
        location_id=event.location_id,
        quantity_available=0,
        quantity_reserved=0,
        last_updated=event.timestamp,
    )

    _apply_inventory_delta(event, current)
    current.last_updated = event.timestamp
    inventory_state.setdefault(event.product_id, {})[event.location_id] = current
    applied_events[event.event_id] = event.model_dump_json()
    logger.info(
        "inventory.event_applied",
        extra={
            "event_id": event.event_id,
            "product_id": event.product_id,
            "available": current.quantity_available,
            "reserved": current.quantity_reserved,
        },
    )


# endregion book:inventory-update-state-init-joined


# region book:inventory-endpoint-receive-joined
@app.post("/events/receive", response_model=InventoryReceived)
async def receive_inventory(event: InventoryReceived, background_tasks: BackgroundTasks):
    """API endpoint for receiving inventory"""
    if event.quantity <= 0:
        raise HTTPException(400, "Received quantity must be positive")

    background_tasks.add_task(publish_event, event)
    await update_inventory_state(event)
    return event


@app.post("/events/sell", response_model=InventorySold)
async def sell_inventory(event: InventorySold, background_tasks: BackgroundTasks):
    await update_inventory_state(event)
    background_tasks.add_task(publish_event, event)
    return event


@app.post("/events/adjust", response_model=InventoryAdjusted)
async def adjust_inventory(event: InventoryAdjusted, background_tasks: BackgroundTasks):
    """API endpoint for inventory adjustments"""
    await update_inventory_state(event)
    background_tasks.add_task(publish_event, event)
    return event


@app.post("/events/transfer", response_model=InventoryTransferred)
async def transfer_inventory(event: InventoryTransferred, background_tasks: BackgroundTasks):
    await update_inventory_state(event)
    background_tasks.add_task(publish_event, event)
    return event


# endregion book:inventory-endpoint-receive-joined


# region book:inventory-endpoint-read
@app.get("/inventory/{product_id}/{location_id}", response_model=InventoryCurrentState)
async def get_inventory(product_id: str, location_id: str):
    """Get current inventory state for a product at a location"""
    if product_id not in inventory_state or location_id not in inventory_state[product_id]:
        raise HTTPException(404, "Inventory not found")
    return inventory_state[product_id][location_id]


@app.get("/inventory/{product_id}", response_model=InventoryStateResponse)
async def get_product_inventory(product_id: str):
    """Get inventory for a product across all locations"""
    if product_id not in inventory_state:
        raise HTTPException(404, "Product not found")
    return InventoryStateResponse(
        product_id=product_id,
        locations=inventory_state[product_id],
    )


# endregion book:inventory-endpoint-read


@app.post("/events/reserve", response_model=InventoryReserved)
async def reserve_inventory(event: InventoryReserved, background_tasks: BackgroundTasks):
    await update_inventory_state(event)
    background_tasks.add_task(publish_event, event)
    return event


@app.post("/events/release", response_model=InventoryReleased)
async def release_inventory(event: InventoryReleased, background_tasks: BackgroundTasks):
    await update_inventory_state(event)
    background_tasks.add_task(publish_event, event)
    return event


@app.post("/events/dispatch", response_model=InventoryDispatched)
async def dispatch_inventory(event: InventoryDispatched, background_tasks: BackgroundTasks):
    await update_inventory_state(event)
    background_tasks.add_task(publish_event, event)
    return event


# region book:inventory-event-main
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# endregion book:inventory-event-main
