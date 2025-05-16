"""
Data models for representing system state, e.g., inventory.
"""

from datetime import datetime

from pydantic import BaseModel, Field  # Use Pydantic as in notebook

# Import relevant Enums
from .enums import InventoryChannel, ReservationStatus


# State models (from notebook)
# Inherit from BaseModel, don't use as decorator
class InventoryReservation(BaseModel):
    """Model for inventory reservations"""

    reservation_id: str
    product_id: str
    location_id: str
    quantity: int
    channel: InventoryChannel  # Use Enum
    order_id: str | None = None
    created_at: datetime
    expires_at: datetime | None = None
    status: ReservationStatus = ReservationStatus.ACTIVE  # Use Enum


# Inherit from BaseModel
class ProductInventoryState(BaseModel):
    """Current inventory state for a product at a location"""

    product_id: str
    location_id: str
    quantity_on_hand: int = 0  # Physical count of inventory
    quantity_reserved: int = 0  # Inventory reserved for orders
    quantity_available: int = 0  # Calculated: on_hand - reserved
    last_updated: datetime = Field(default_factory=datetime.now)
    reservations: dict[str, InventoryReservation] = Field(default_factory=dict)
    version: int = 0  # Optimistic concurrency control
    last_event_id: str | None = None  # Last event that modified this state

    # Add method to calculate available quantity (as done implicitly before)
    def calculate_available(self):
        self.quantity_available = self.quantity_on_hand - self.quantity_reserved
        return self.quantity_available

    # Add method to update quantities based on event type (complex logic, maybe belongs in agent/handler)
    # def update_from_event(self, event): ...


# InventoryCurrentState from notebook seems redundant given ProductInventoryState
# class InventoryCurrentState(BaseModel):
#    ...
