"""
Data models for store fulfillment optimization.
Includes Item and Order classes.
"""



class Item:
    """Represents a product in the store inventory."""

    def __init__(
        self,
        item_id: str,
        name: str,
        category: str,
        location: tuple[int, int],
        temperature_zone: str = "ambient",
        handling_time: float = 1.0,
        fragility: float = 0.0,
    ):
        self.item_id = item_id
        self.name = name
        self.category = category
        self.location = location  # (x, y) coordinates in store
        self.temperature_zone = temperature_zone  # "ambient", "refrigerated", "frozen"
        self.handling_time = handling_time  # base time to pick in minutes
        self.fragility = fragility  # 0.0 to 1.0, affects stacking and handling

    def __repr__(self):
        return f"Item({self.item_id}: {self.name} at {self.location})"


class Order:
    """Represents a customer order with multiple items."""

    def __init__(
        self,
        order_id: str,
        items: list[Item],
        priority: int = 1,
        due_time: float | None = None,
    ):
        self.order_id = order_id
        self.items = items
        self.priority = priority  # 1 (standard) to 5 (highest)
        self.due_time = due_time  # minutes from now
        self.assigned_to = None
        self.status = "pending"  # pending, in_progress, completed

    def get_temperature_zones(self) -> set[str]:
        """Return the set of temperature zones required for this order."""
        return {item.temperature_zone for item in self.items}

    def get_item_locations(self) -> list[tuple[int, int]]:
        """Return the locations of all items in the order."""
        return [item.location for item in self.items]

    def estimate_picking_time(self, associate_efficiency: float = 1.0) -> float:
        """Estimate the time to pick all items in the order."""
        # Base handling time for all items
        base_time = sum(item.handling_time for item in self.items)
        # Adjust for associate efficiency
        return base_time / associate_efficiency

    def __repr__(self):
        return (
            f"Order({self.order_id}: {len(self.items)} items, priority {self.priority})"
        )
