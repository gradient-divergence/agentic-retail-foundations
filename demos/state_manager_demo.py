"""
Demonstration of distributed state management using CRDTs (PN-Counter)
and event sourcing concepts with Redis.

Requires Redis to be running.
Run with: uvicorn demos.state_manager_demo:app --reload
"""

import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
import redis.asyncio as redis
from pydantic import BaseModel

# Import models and utilities
from utils.crdt import PNCounter  # Use the extracted CRDT

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("state-manager")

# Initialize FastAPI app
app = FastAPI(title="Inventory State Manager Demo")

# --- Redis Connections ---
# Separate connections for events and state snapshots
event_redis: redis.Redis | None = None
state_redis: redis.Redis | None = None
STATE_KEY_PREFIX = "inventory_state:"
EVENT_STREAM_KEY = "inventory_events_crdt"


# --- Pydantic Models for API ---
class InventoryUpdate(BaseModel):
    node_id: str  # ID of the node reporting the change
    product_id: str
    location_id: str
    increment: int | None = None
    decrement: int | None = None


class InventoryStateResponse(BaseModel):
    product_id: str
    location_id: str
    current_value: int
    state: dict[str, Any]  # Show the raw CRDT state


# --- Redis Connection Management ---


@app.on_event("startup")
async def startup_event():
    global event_redis, state_redis
    try:
        event_redis = redis.Redis(
            host="localhost", port=6379, db=2, decode_responses=True
        )
        state_redis = redis.Redis(
            host="localhost", port=6379, db=3, decode_responses=True
        )
        await event_redis.ping()
        await state_redis.ping()
        logger.info("Connected to Redis databases (DB2 for events, DB3 for state).")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}. State management disabled.")
        event_redis = None
        state_redis = None


@app.on_event("shutdown")
async def shutdown_event():
    if event_redis:
        await event_redis.close()
    if state_redis:
        await state_redis.close()
    logger.info("Redis connections closed.")


# --- Core CRDT Logic ---


async def get_crdt_state(product_id: str, location_id: str) -> PNCounter:
    """Retrieve the current CRDT state from Redis, or create a new one."""
    if not state_redis:
        raise HTTPException(503, "State DB not available")

    key = f"{STATE_KEY_PREFIX}{product_id}:{location_id}"
    raw_state = await state_redis.get(key)

    if raw_state:
        try:
            state_dict = json.loads(raw_state)
            # Need to reconstruct the PNCounter state from the stored dict
            counter = PNCounter(product_id, location_id)
            counter.increments = state_dict.get("p", {})  # Use state property keys
            counter.decrements = state_dict.get("n", {})
            logger.debug(f"Retrieved CRDT state for {key}")
            return counter
        except json.JSONDecodeError:
            logger.error(f"Failed to decode stored state for {key}. Starting fresh.")
            return PNCounter(product_id, location_id)
        except Exception as e:
            logger.error(f"Error reconstructing CRDT state for {key}: {e}")
            # Fallback to new counter on error
            return PNCounter(product_id, location_id)
    else:
        logger.debug(f"No existing CRDT state found for {key}, creating new.")
        return PNCounter(product_id, location_id)


async def save_crdt_state(counter: PNCounter):
    """Save the CRDT state back to Redis."""
    if not state_redis:
        raise HTTPException(503, "State DB not available")

    key = f"{STATE_KEY_PREFIX}{counter.product_id}:{counter.location_id}"
    try:
        # Use the state property for serialization
        await state_redis.set(key, json.dumps(counter.state))
        logger.debug(f"Saved CRDT state for {key}")
    except Exception as e:
        logger.error(f"Failed to save CRDT state for {key}: {e}")
        # Decide how to handle save errors (e.g., raise, retry)


async def publish_crdt_update_event(update_data: InventoryUpdate):
    """Publish the raw increment/decrement operation to an event stream."""
    if not event_redis:
        logger.warning("Event DB not available, cannot publish CRDT update event.")
        return

    try:
        event_payload = update_data.model_dump_json()
        await event_redis.xadd(EVENT_STREAM_KEY, {"data": event_payload})
        logger.info(f"Published CRDT update event to stream '{EVENT_STREAM_KEY}'")
    except Exception as e:
        logger.error(f"Failed to publish CRDT update event: {e}")


# --- API Endpoints ---


@app.post("/inventory/update", status_code=202, response_model=InventoryStateResponse)
async def update_inventory(update: InventoryUpdate):
    """
    Receive an inventory update (increment or decrement) from a node.
    Updates the local CRDT state and publishes the raw operation.
    """
    if not state_redis:
        raise HTTPException(503, "State DB not available")

    # Retrieve current CRDT state
    counter = await get_crdt_state(update.product_id, update.location_id)

    # Apply the update
    try:
        if update.increment is not None and update.increment > 0:
            counter.increment(update.node_id, update.increment)
            logger.info(
                f"Applied INCREMENT from {update.node_id} ({update.increment}) to {counter.product_id}:{counter.location_id}"
            )
        elif update.decrement is not None and update.decrement > 0:
            counter.decrement(update.node_id, update.decrement)
            logger.info(
                f"Applied DECREMENT from {update.node_id} ({update.decrement}) to {counter.product_id}:{counter.location_id}"
            )
        else:
            raise HTTPException(
                400, "Update must contain a positive increment or decrement."
            )

    except ValueError as e:
        raise HTTPException(400, str(e))

    # Save the updated state
    await save_crdt_state(counter)

    # Publish the raw update event (could be done in background)
    await publish_crdt_update_event(update)

    return InventoryStateResponse(
        product_id=counter.product_id,
        location_id=counter.location_id,
        current_value=counter.value(),
        state=counter.state,  # Return the raw p/n counters
    )


@app.get(
    "/inventory/state/{location_id}/{product_id}", response_model=InventoryStateResponse
)
async def get_inventory_state(location_id: str, product_id: str):
    """Get the current merged value and raw state of the CRDT counter."""
    if not state_redis:
        raise HTTPException(503, "State DB not available")

    counter = await get_crdt_state(product_id, location_id)
    return InventoryStateResponse(
        product_id=counter.product_id,
        location_id=counter.location_id,
        current_value=counter.value(),
        state=counter.state,
    )


# --- Background Synchronization (Placeholder) ---
# In a real system, a background worker would listen to the event stream
# and potentially merge updates from other nodes into the local state Redis.
# async def sync_worker():
#     if not event_redis or not state_redis:
#         logger.error("Cannot start sync worker: Redis not available.")
#         return
#     last_id = '$' # Start from the end for new events
#     while True:
#         try:
#             response = await event_redis.xread({EVENT_STREAM_KEY: last_id}, block=0) # Block indefinitely
#             if response:
#                 stream, messages = response[0]
#                 for message_id, message_data in messages:
#                     last_id = message_id
#                     try:
#                         update_event_data = json.loads(message_data['data'])
#                         update_event = InventoryUpdate(**update_event_data)
#                         # Get local CRDT state
#                         local_counter = await get_crdt_state(update_event.product_id, update_event.location_id)
#                         # Apply the received update (conceptually merging)
#                         if update_event.increment:
#                             local_counter.increment(update_event.node_id, update_event.increment)
#                         elif update_event.decrement:
#                             local_counter.decrement(update_event.node_id, update_event.decrement)
#                         # Save the potentially merged state
#                         await save_crdt_state(local_counter)
#                         logger.info(f"Synced event {message_id} from node {update_event.node_id}")
#                     except Exception as proc_e:
#                         logger.error(f"Error processing synced event {message_id}: {proc_e}")
#         except Exception as read_e:
#             logger.error(f"Error reading from event stream: {read_e}")
#             await asyncio.sleep(5) # Wait before retrying


if __name__ == "__main__":
    import uvicorn

    # Note: Ensure Redis is running on localhost:6379 for this demo
    logger.info("Starting Inventory State Manager API...")
    # Optionally start the sync worker in the background
    # asyncio.create_task(sync_worker())
    uvicorn.run(app, host="0.0.0.0", port=8004)  # Use different port
