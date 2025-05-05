"""
Data models for events within the agent system.
"""

import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field  # Use Pydantic as in notebook

# Import relevant Enums
from .enums import AgentType, InventoryEventType


# Base Event Model (from notebook)
class RetailEvent(BaseModel):  # Using Pydantic BaseModel
    """Base event for retail system interactions."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str  # Generic event type name
    payload: dict[str, Any]
    source: AgentType  # Use the Enum
    timestamp: datetime = Field(default_factory=datetime.now)

    # # Removed to_json method - Pydantic handles serialization
    # def to_json(self) -> str:
    #     ...


# Inventory Event Models (from notebook)
class InventoryEvent(BaseModel):  # Using Pydantic BaseModel
    """Base model for all inventory events"""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: InventoryEventType  # Use the Enum
    timestamp: datetime = Field(default_factory=datetime.now)
    product_id: str
    location_id: str
    quantity: int
    user_id: str | None = None
    reference_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Specialized Event Types
class InventoryReceived(InventoryEvent):
    """Event for receiving inventory"""

    event_type: InventoryEventType = InventoryEventType.RECEIVED
    supplier_id: str
    purchase_order_id: str | None = None


class InventorySold(InventoryEvent):
    """Event for selling inventory"""

    event_type: InventoryEventType = InventoryEventType.SOLD
    order_id: str
    customer_id: str | None = None


class InventoryAdjusted(InventoryEvent):
    """Event for manual inventory adjustments"""

    event_type: InventoryEventType = InventoryEventType.ADJUSTED
    reason_code: str
    notes: str | None = None


class InventoryTransferred(InventoryEvent):
    """Event for inventory transfers between locations"""

    event_type: InventoryEventType = InventoryEventType.TRANSFERRED
    source_location_id: str
    destination_location_id: str
    transfer_id: str | None = None


# Reserved/Released could also be specialized if needed
class InventoryReserved(InventoryEvent):
    """Event for reserving inventory"""

    event_type: InventoryEventType = InventoryEventType.RESERVED
    order_id: str


class InventoryReleased(InventoryEvent):
    """Event for releasing reserved inventory"""

    event_type: InventoryEventType = InventoryEventType.RELEASED
    order_id: str
