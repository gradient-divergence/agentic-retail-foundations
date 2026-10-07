from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

try:
    import redis.asyncio as redis
except ModuleNotFoundError:
    raise SystemExit("Install the streaming extra: uv sync --extra streaming") from None
from fastapi import FastAPI
from pydantic import BaseModel, Field, JsonValue

# region book:inventory-crdt-imports
# Initialize FastAPI app
app = FastAPI(title="Omnichannel Inventory State Management")
event_redis = redis.Redis(host="redis", port=6379, db=0)
state_redis = redis.Redis(host="redis", port=6379, db=1)
# endregion book:inventory-crdt-imports


# region book:inventory-crdt-enum-event-type
class InventoryEventType(str, Enum):
    """Types of inventory events"""

    RECEIVED = "RECEIVED"
    SOLD = "SOLD"
    RESERVED = "RESERVED"
    RELEASED = "RELEASED"
    ADJUSTED = "ADJUSTED"
    TRANSFERRED_OUT = "TRANSFERRED_OUT"
    TRANSFERRED_IN = "TRANSFERRED_IN"
    SNAPSHOT = "SNAPSHOT"


# endregion book:inventory-crdt-enum-event-type


# region book:inventory-crdt-enum-channel
class InventoryChannel(str, Enum):
    """Available inventory channels"""

    STORE = "STORE"
    ONLINE = "ONLINE"
    MARKETPLACE = "MARKETPLACE"
    WAREHOUSE = "WAREHOUSE"
    POS = "POS"
    MOBILE_APP = "MOBILE_APP"


# endregion book:inventory-crdt-enum-channel


# region book:inventory-crdt-enum-reservation-status
class ReservationStatus(str, Enum):
    """Possible reservation statuses"""

    ACTIVE = "ACTIVE"
    FULFILLED = "FULFILLED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


# endregion book:inventory-crdt-enum-reservation-status


# region book:inventory-crdt-event-model
class InventoryEvent(BaseModel):
    """Base event model"""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: InventoryEventType
    product_id: str
    location_id: str
    channel: InventoryChannel
    quantity: int
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: str | None = None
    reference_id: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


# endregion book:inventory-crdt-event-model


# region book:inventory-crdt-reservation-model
class InventoryReservation(BaseModel):
    """Model for inventory reservations"""

    reservation_id: str
    product_id: str
    location_id: str
    quantity: int
    channel: InventoryChannel
    order_id: str | None = None
    created_at: datetime
    expires_at: datetime | None = None
    status: ReservationStatus = ReservationStatus.ACTIVE


# endregion book:inventory-crdt-reservation-model


# region book:inventory-crdt-product-state
class ProductInventoryState(BaseModel):
    """Current inventory state for a product at a location"""

    product_id: str
    location_id: str
    quantity_on_hand: int = 0
    quantity_reserved: int = 0
    quantity_available: int = 0
    last_updated: datetime = Field(default_factory=datetime.now)
    reservations: dict[str, InventoryReservation] = Field(default_factory=dict)
    version: int = 0
    last_event_id: str | None = None


class PNCounterSnapshot(BaseModel):
    product_id: str
    location_id: str
    increments: dict[str, int]
    decrements: dict[str, int]


# endregion book:inventory-crdt-product-state


# region book:inventory-crdt-pn-counter
class PNCounter:
    """
    Positive-Negative Counter CRDT for inventory tracking
    Guarantees eventual consistency across distributed nodes
    """

    def __init__(self, product_id: str, location_id: str, initial_value: int = 0):
        self.product_id = product_id
        self.location_id = location_id
        self.increments: dict[str, int] = {}
        self.decrements: dict[str, int] = {}
        if initial_value > 0:
            self.increments["initial"] = initial_value
        elif initial_value < 0:
            self.decrements["initial"] = abs(initial_value)

    # endregion book:inventory-crdt-pn-counter

    # region book:inventory-crdt-pn-increment
    def increment(self, node_id: str, value: int) -> None:
        """Increment counter by value"""
        if value < 0:
            raise ValueError("Cannot increment by negative value")
        if node_id not in self.increments:
            self.increments[node_id] = 0
        self.increments[node_id] += value

    # endregion book:inventory-crdt-pn-increment

    # region book:inventory-crdt-pn-decrement
    def decrement(self, node_id: str, value: int) -> None:
        """Decrement counter by value"""
        if value < 0:
            raise ValueError("Cannot decrement by negative value")
        if node_id not in self.decrements:
            self.decrements[node_id] = 0
        self.decrements[node_id] += value

    # endregion book:inventory-crdt-pn-decrement

    # region book:inventory-crdt-pn-value
    def value(self) -> int:
        """Get current counter value"""
        return sum(self.increments.values()) - sum(self.decrements.values())

    # endregion book:inventory-crdt-pn-value

    # region book:inventory-crdt-pn-merge
    def merge(self, other: PNCounter) -> PNCounter:
        """Merge with another counter - commutative and associative"""
        if (self.product_id, self.location_id) != (other.product_id, other.location_id):
            raise ValueError("Can only merge counters for the same product and location")
        result = PNCounter(self.product_id, self.location_id)
        all_inc_keys = set(self.increments.keys()) | set(other.increments.keys())
        for key in all_inc_keys:
            result.increments[key] = max(self.increments.get(key, 0), other.increments.get(key, 0))
        all_dec_keys = set(self.decrements.keys()) | set(other.decrements.keys())
        for key in all_dec_keys:
            result.decrements[key] = max(self.decrements.get(key, 0), other.decrements.get(key, 0))
        return result

    # endregion book:inventory-crdt-pn-merge

    # region book:inventory-crdt-pn-dict
    def to_dict(self) -> PNCounterSnapshot:
        """Convert to snapshot for storage"""
        return PNCounterSnapshot(
            product_id=self.product_id,
            location_id=self.location_id,
            increments=self.increments,
            decrements=self.decrements,
        )

    @classmethod
    def from_dict(cls, data: PNCounterSnapshot) -> PNCounter:
        """Create from snapshot"""
        counter = cls(data.product_id, data.location_id)
        counter.increments = data.increments.copy()
        counter.decrements = data.decrements.copy()
        return counter


# endregion book:inventory-crdt-pn-dict
