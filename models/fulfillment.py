"""
Data models for store fulfillment optimization.
Includes Item and Order classes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class Item:
    """Represents an item to be picked in an order."""
    item_id: str
    name: str
    section: str # e.g., 'grocery', 'produce', 'frozen'
    location: Tuple[int, int] # (x, y) coordinates in the store layout
    weight: float = 0.5 # Example weight in kg
    volume: float = 0.001 # Example volume in cubic meters
    temperature_zone: str = "ambient" # ambient, refrigerated, frozen
    handling_time: float = 1.0 # Base time units to handle/pick this item
    fragility: float = 0.0 # 0.0 (not fragile) to 1.0 (very fragile)

@dataclass
class Order:
    """Represents a customer order to be fulfilled."""
    order_id: str
    items: List[Item]
    priority: int = 1 # Lower number means higher priority
    due_time: Optional[float] = None # Time limit for fulfillment in minutes from now
    status: str = "pending" # e.g., pending, assigned, picking, completed

    def estimate_picking_time(self, associate_efficiency: float = 1.0) -> float:
        """Estimate the time to pick all items in the order."""
        # Placeholder: more sophisticated estimation needed
        # Consider handling_time per item, travel time (needs layout)
        base_time = sum(item.handling_time for item in self.items)
        return base_time / associate_efficiency

@dataclass
class Associate:
    """Represents a store associate who can fulfill orders."""
    associate_id: str
    name: str
    efficiency: float = 1.0 # Multiplier for picking speed ( >1 faster, <1 slower)
    authorized_zones: List[str] = field(default_factory=lambda: ["ambient", "refrigerated", "frozen"])
    current_location: Tuple[int, int] = (0, 0) # Starting location (e.g., packing station)
    max_capacity_weight: float = 15.0 # Max weight they can carry
    max_capacity_volume: float = 0.1 # Max volume they can carry
    shift_end_time: Optional[float] = None # Time remaining in shift (minutes)
    current_task_completion_time: float = 0.0 # When current task is expected to end
    current_order_ids: List[str] = field(default_factory=list)

    def __repr__(self):
        return (
            f"Order({self.order_id}: {len(self.items)} items, priority {self.priority})"
        )
