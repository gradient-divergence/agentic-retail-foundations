from __future__ import annotations

import json
import logging
import uuid

# region book:inventory-event-imports
from datetime import datetime
from enum import Enum

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


# region book:inventory-event-type-enum
class EventType(str, Enum):
    """Types of inventory events"""

    RECEIVED = "inventory.received"
    SOLD = "inventory.sold"
    ADJUSTED = "inventory.adjusted"
    TRANSFERRED = "inventory.transferred"
    RESERVED = "inventory.reserved"
    RELEASED = "inventory.released"


# endregion book:inventory-event-type-enum


# region book:inventory-event-base-model
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


# endregion book:inventory-event-base-model


# region book:inventory-event-received
class InventoryReceived(InventoryEvent):
    """Event for receiving inventory"""

    event_type: EventType = EventType.RECEIVED
    supplier_id: str
    purchase_order_id: str | None = None


# endregion book:inventory-event-received


# region book:inventory-event-sold
class InventorySold(InventoryEvent):
    """Event for selling inventory"""

    event_type: EventType = EventType.SOLD
    order_id: str
    customer_id: str | None = None


# endregion book:inventory-event-sold


# region book:inventory-event-adjusted
class InventoryAdjusted(InventoryEvent):
    """Event for manual inventory adjustments"""

    event_type: EventType = EventType.ADJUSTED
    reason_code: str
    notes: str | None = None


# endregion book:inventory-event-adjusted


# region book:inventory-event-transferred
class InventoryTransferred(InventoryEvent):
    """Event for inventory transfers between locations"""

    event_type: EventType = EventType.TRANSFERRED
    source_location_id: str
    destination_location_id: str
    transfer_id: str | None = None


# endregion book:inventory-event-transferred


# region book:inventory-current-state
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


# endregion book:inventory-current-state


# region book:inventory-state-cache
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


# endregion book:inventory-state-cache


# region book:inventory-update-state-init
async def update_inventory_state(event: InventoryEvent) -> None:  # noqa: C901
    """Update the current inventory state based on event"""
    product_id = event.product_id
    location_id = event.location_id
    if product_id not in inventory_state:
        inventory_state[product_id] = {}

    if location_id not in inventory_state[product_id]:
        inventory_state[product_id][location_id] = InventoryCurrentState(
            product_id=product_id,
            location_id=location_id,
            quantity_available=0,
            quantity_reserved=0,
            last_updated=datetime.now(),
        )

    # endregion book:inventory-update-state-init

    # region book:inventory-update-state-apply
    current = inventory_state[product_id][location_id]

    if event.event_type == EventType.RECEIVED:
        current.quantity_available += event.quantity
    elif event.event_type == EventType.SOLD:
        current.quantity_available -= event.quantity
    elif event.event_type == EventType.ADJUSTED:
        current.quantity_available += event.quantity
    elif event.event_type == EventType.RESERVED:
        current.quantity_available -= event.quantity
        current.quantity_reserved += event.quantity
    elif event.event_type == EventType.RELEASED:
        current.quantity_reserved -= event.quantity
        current.quantity_available += event.quantity
    elif event.event_type == EventType.TRANSFERRED:
        if isinstance(event, InventoryTransferred):
            if product_id in inventory_state and event.source_location_id in inventory_state[product_id]:
                inventory_state[product_id][event.source_location_id].quantity_available -= event.quantity
                inventory_state[product_id][event.source_location_id].last_updated = datetime.now()

            if product_id not in inventory_state:
                inventory_state[product_id] = {}

            if event.destination_location_id not in inventory_state[product_id]:
                inventory_state[product_id][event.destination_location_id] = InventoryCurrentState(
                    product_id=product_id,
                    location_id=event.destination_location_id,
                    quantity_available=0,
                    quantity_reserved=0,
                    last_updated=datetime.now(),
                )

            inventory_state[product_id][event.destination_location_id].quantity_available += event.quantity
            inventory_state[product_id][event.destination_location_id].last_updated = datetime.now()

    current.last_updated = datetime.now()

    logger.info(
        "Updated inventory state for %s at %s: %s available, %s reserved",
        product_id,
        location_id,
        current.quantity_available,
        current.quantity_reserved,
    )


# endregion book:inventory-update-state-apply


# region book:inventory-endpoint-receive
@app.post("/events/receive", response_model=InventoryReceived)
async def receive_inventory(event: InventoryReceived, background_tasks: BackgroundTasks):
    """API endpoint for receiving inventory"""
    if event.quantity <= 0:
        raise HTTPException(400, "Received quantity must be positive")

    background_tasks.add_task(publish_event, event)
    background_tasks.add_task(update_inventory_state, event)
    return event


# endregion book:inventory-endpoint-receive


# region book:inventory-endpoint-sell
@app.post("/events/sell", response_model=InventorySold)
async def sell_inventory(event: InventorySold, background_tasks: BackgroundTasks):
    """API endpoint for selling inventory"""
    if event.quantity <= 0:
        raise HTTPException(400, "Sold quantity must be positive")

    product_id = event.product_id
    location_id = event.location_id

    if (
        product_id not in inventory_state
        or location_id not in inventory_state[product_id]
        or inventory_state[product_id][location_id].quantity_available < event.quantity
    ):
        raise HTTPException(400, "Insufficient inventory available")

    background_tasks.add_task(publish_event, event)
    background_tasks.add_task(update_inventory_state, event)
    return event


# endregion book:inventory-endpoint-sell


# region book:inventory-endpoint-adjust
@app.post("/events/adjust", response_model=InventoryAdjusted)
async def adjust_inventory(event: InventoryAdjusted, background_tasks: BackgroundTasks):
    """API endpoint for inventory adjustments"""
    product_id = event.product_id
    location_id = event.location_id

    if event.quantity < 0:
        if (
            product_id not in inventory_state
            or location_id not in inventory_state[product_id]
            or inventory_state[product_id][location_id].quantity_available < abs(event.quantity)
        ):
            raise HTTPException(400, "Insufficient inventory for adjustment")

    background_tasks.add_task(publish_event, event)
    background_tasks.add_task(update_inventory_state, event)
    return event


# endregion book:inventory-endpoint-adjust


# region book:inventory-endpoint-transfer
@app.post("/events/transfer", response_model=InventoryTransferred)
async def transfer_inventory(event: InventoryTransferred, background_tasks: BackgroundTasks):
    """API endpoint for inventory transfers"""
    if event.quantity <= 0:
        raise HTTPException(400, "Transfer quantity must be positive")

    if event.source_location_id == event.destination_location_id:
        raise HTTPException(400, "Source and destination locations must be different")

    product_id = event.product_id
    source_location_id = event.source_location_id

    if (
        product_id not in inventory_state
        or source_location_id not in inventory_state[product_id]
        or inventory_state[product_id][source_location_id].quantity_available < event.quantity
    ):
        raise HTTPException(400, "Insufficient inventory at source location")

    background_tasks.add_task(publish_event, event)
    background_tasks.add_task(update_inventory_state, event)
    return event


# endregion book:inventory-endpoint-transfer


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


# region book:inventory-event-main
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# endregion book:inventory-event-main
