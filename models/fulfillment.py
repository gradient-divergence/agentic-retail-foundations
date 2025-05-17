"""
Data models for store fulfillment optimization.
Includes Item and Order classes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .enums import AgentType, FulfillmentMethod, OrderStatus


@dataclass
class Item:
    """Represents an item to be picked in an order."""

    item_id: str
    name: str
    section: str  # e.g., 'grocery', 'produce', 'frozen'
    location: tuple[int, int]  # (x, y) coordinates in the store layout
    weight: float = 0.5  # Example weight in kg
    volume: float = 0.001  # Example volume in cubic meters
    temperature_zone: str = "ambient"  # ambient, refrigerated, frozen
    handling_time: float = 1.0  # Base time units to handle/pick this item
    fragility: float = 0.0  # 0.0 (not fragile) to 1.0 (very fragile)


@dataclass
class OrderLineItem:
    """Individual item in an order (Extracted from notebook)"""

    product_id: str
    quantity: int
    price: float
    fulfillment_method: FulfillmentMethod | None = None
    fulfillment_location_id: str | None = None
    status: OrderStatus = OrderStatus.CREATED
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """Represents a customer order to be fulfilled."""

    order_id: str
    items: list[OrderLineItem]
    customer_id: str
    created_at: datetime = field(default_factory=datetime.now)
    store_id: str | None = None
    status: OrderStatus = OrderStatus.CREATED
    preferred_fulfillment_method: FulfillmentMethod | None = None
    delivery_address: dict[str, str] | None = None
    pickup_store_id: str | None = None
    payment_details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def add_event(self, agent_type: AgentType, action: str, details: dict[str, Any]) -> None:
        """Add an event to the order history"""
        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "agent": agent_type.value,
                "action": action,
                "details": details,
            }
        )

    def update_status(
        self,
        new_status: OrderStatus,
        agent_type: AgentType,
        details: dict[str, Any],
    ) -> None:
        """Update order status with tracking"""
        old_status = self.status
        self.status = new_status
        self.add_event(
            agent_type,
            f"status_change_{old_status.value}_to_{new_status.value}",
            details,
        )


@dataclass
class Associate:
    """Represents a store associate who can fulfill orders."""

    associate_id: str
    name: str
    efficiency: float = 1.0  # Multiplier for picking speed ( >1 faster, <1 slower)
    authorized_zones: list[str] = field(default_factory=lambda: ["ambient", "refrigerated", "frozen"])
    current_location: tuple[int, int] = (
        0,
        0,
    )  # Starting location (e.g., packing station)
    max_capacity_weight: float = 15.0  # Max weight they can carry
    max_capacity_volume: float = 0.1  # Max volume they can carry
    shift_end_time: float | None = None  # Time remaining in shift (minutes)
    current_task_completion_time: float = 0.0  # When current task is expected to end
    current_order_ids: list[str] = field(default_factory=list)
