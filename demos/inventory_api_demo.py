"""
FastAPI application demonstrating event-driven inventory management.

Requires Redis to be running.
Run with: uvicorn demos.inventory_api_demo:app --reload
"""

import asyncio
import json
import logging

import redis.asyncio as redis  # Use async redis client

# Use standard library for background tasks
from fastapi import BackgroundTasks, FastAPI, HTTPException

from models.enums import InventoryEventType

# Import refactored models and enums
from models.events import (  # Add Reserved/Released if needed
    InventoryAdjusted,
    InventoryEvent,
    InventoryReceived,
    InventorySold,
    InventoryTransferred,
)
from models.state import ProductInventoryState

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("inventory-api")

# Initialize FastAPI app
app = FastAPI(title="Retail Inventory Event Service")

# --- State Management (In-Memory for Demo) ---
# In production, use a persistent database (e.g., PostgreSQL, MongoDB)
# or potentially Redis itself if configured for persistence.
# The key is product_id:location_id
inventory_state_cache: dict[str, ProductInventoryState] = {}
state_lock = asyncio.Lock()  # Lock for concurrent state updates

# --- Redis Connection ---
redis_client: redis.Redis | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup."""
    global redis_client
    try:
        redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        await redis_client.ping()  # Verify connection
        logger.info("Connected to Redis successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}. Event publishing disabled.")
        redis_client = None


@app.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection on shutdown."""
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed.")


# --- Event Publishing & State Update Logic ---


async def publish_event_to_stream(event: InventoryEvent) -> None:
    """Publish inventory event to Redis stream."""
    if not redis_client:
        logger.warning("Redis client not available, cannot publish event.")
        return

    try:
        event_data = event.model_dump(mode="json")  # Use Pydantic v2 method
        stream_key = f"inventory-events:{event.location_id}"  # Stream per location
        all_events_stream = "inventory-events:all"

        # Use pipeline for atomic adds if needed, basic XADD for demo
        await redis_client.xadd(stream_key, {"data": json.dumps(event_data)})
        await redis_client.xadd(all_events_stream, {"data": json.dumps(event_data)})
        logger.info(f"Published {event.event_type.value} event: {event.event_id} to streams.")
    except Exception as e:
        logger.error(f"Failed to publish event {event.event_id} to Redis: {str(e)}")
        # Optionally re-raise or handle differently


async def update_inventory_state(event: InventoryEvent) -> None:
    """Update the in-memory inventory state based on an event. Thread-safe."""
    global inventory_state_cache
    product_id = event.product_id
    location_id = event.location_id
    key = f"{product_id}:{location_id}"

    async with state_lock:
        # Get current state or initialize
        current = inventory_state_cache.get(key)
        if not current:
            current = ProductInventoryState(product_id=product_id, location_id=location_id)
            inventory_state_cache[key] = current

        # Store previous quantities for logging/validation
        prev_on_hand = current.quantity_on_hand
        prev_reserved = current.quantity_reserved

        # Apply event logic
        qty = event.quantity
        if event.event_type == InventoryEventType.RECEIVED:
            current.quantity_on_hand += qty
        elif event.event_type == InventoryEventType.SOLD:
            # Assume sale reduces on_hand (fulfillment handles reservation release)
            current.quantity_on_hand -= qty
        elif event.event_type == InventoryEventType.ADJUSTED:
            current.quantity_on_hand += qty  # Adjustment can be +/-
        elif event.event_type == InventoryEventType.RESERVED:
            current.quantity_on_hand -= qty
            current.quantity_reserved += qty
        elif event.event_type == InventoryEventType.RELEASED:
            current.quantity_reserved -= qty
            current.quantity_on_hand += qty
        elif event.event_type == InventoryEventType.TRANSFERRED:
            # This event implies quantity LEAVING this location_id
            # Another event should be published for the destination location
            current.quantity_on_hand -= qty
        else:
            logger.warning(f"Unhandled inventory event type: {event.event_type} for {key}")
            return  # Don't update state for unhandled types

        # Recalculate available quantity
        current.calculate_available()
        current.last_updated = event.timestamp
        current.last_event_id = event.event_id
        current.version += 1  # Increment version for optimistic locking if needed

        logger.info(
            f"State Updated for {key} by Event {event.event_id} ({event.event_type.value}): "
            f"OnHand {prev_on_hand}->{current.quantity_on_hand}, "
            f"Reserved {prev_reserved}->{current.quantity_reserved}, "
            f"Available {current.quantity_available}"
        )


# --- API Endpoints ---


@app.post("/event/receive", status_code=202)
async def receive_inventory_event(event: InventoryReceived, background_tasks: BackgroundTasks):
    """API endpoint for receiving inventory."""
    if event.quantity <= 0:
        raise HTTPException(400, "Received quantity must be positive")
    background_tasks.add_task(update_inventory_state, event)
    background_tasks.add_task(publish_event_to_stream, event)
    return {"message": "Inventory Received event accepted", "event_id": event.event_id}


@app.post("/event/sell", status_code=202)
async def sell_inventory_event(event: InventorySold, background_tasks: BackgroundTasks):
    """API endpoint for selling inventory."""
    if event.quantity <= 0:
        raise HTTPException(400, "Sold quantity must be positive")
    # Check availability (simple check on cached state - may have race conditions)
    key = f"{event.product_id}:{event.location_id}"
    current_state = inventory_state_cache.get(key)
    if not current_state or current_state.calculate_available() < event.quantity:
        # Check needs recalculation in case state changed
        if current_state:
            current_state.calculate_available()
        if not current_state or current_state.quantity_available < event.quantity:
            raise HTTPException(
                400,
                f"Insufficient available inventory for {event.product_id} at {event.location_id}",
            )

    background_tasks.add_task(update_inventory_state, event)
    background_tasks.add_task(publish_event_to_stream, event)
    return {"message": "Inventory Sold event accepted", "event_id": event.event_id}


@app.post("/event/adjust", status_code=202)
async def adjust_inventory_event(event: InventoryAdjusted, background_tasks: BackgroundTasks):
    """API endpoint for inventory adjustments."""
    key = f"{event.product_id}:{event.location_id}"
    if event.quantity < 0:  # If reducing stock
        current_state = inventory_state_cache.get(key)
        if not current_state or current_state.quantity_on_hand < abs(event.quantity):
            raise HTTPException(
                400,
                f"Insufficient on-hand inventory for adjustment: {event.product_id} at {event.location_id}",
            )

    background_tasks.add_task(update_inventory_state, event)
    background_tasks.add_task(publish_event_to_stream, event)
    return {"message": "Inventory Adjusted event accepted", "event_id": event.event_id}


@app.post("/event/transfer", status_code=202)
async def transfer_inventory_event(event: InventoryTransferred, background_tasks: BackgroundTasks):
    """API endpoint for inventory transfers (records the outgoing part)."""
    if event.quantity <= 0:
        raise HTTPException(400, "Transfer quantity must be positive")
    if event.source_location_id == event.destination_location_id:
        raise HTTPException(400, "Source and destination locations must be different")

    key = f"{event.product_id}:{event.source_location_id}"
    current_state = inventory_state_cache.get(key)
    if not current_state or current_state.quantity_on_hand < event.quantity:
        raise HTTPException(
            400,
            f"Insufficient on-hand inventory for transfer at source: {event.product_id} at {event.source_location_id}",
        )

    # Publish event for the source location decrement
    # Note: A separate RECEIVED event should be published by the destination system/agent
    background_tasks.add_task(update_inventory_state, event)
    background_tasks.add_task(publish_event_to_stream, event)
    return {
        "message": "Inventory Transfer (Outgoing) event accepted",
        "event_id": event.event_id,
    }


# Add endpoints for Reserved/Released if needed

# --- Read Endpoint ---


@app.get("/inventory/{location_id}/{product_id}", response_model=ProductInventoryState)
async def get_inventory(location_id: str, product_id: str):
    """Get current inventory state for a product at a location."""
    key = f"{product_id}:{location_id}"
    current_state = inventory_state_cache.get(key)
    if not current_state:
        # Return a default state or 404
        # raise HTTPException(404, f"Inventory not found for {product_id} at {location_id}")
        return ProductInventoryState(product_id=product_id, location_id=location_id)

    # Recalculate available just in case state is read before background task completes
    current_state.calculate_available()
    return current_state


if __name__ == "__main__":
    import uvicorn

    # Note: Ensure Redis is running on localhost:6379 for this demo
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Use different port than gateway
