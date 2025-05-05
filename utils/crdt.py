"""
Conflict-free Replicated Data Types (CRDTs).
Contains implementations like PN-Counter.
"""

from typing import Any


# PN-Counter CRDT (from notebook)
class PNCounter:
    """
    Positive-Negative Counter CRDT for inventory tracking.
    Guarantees eventual consistency across distributed nodes.
    Based on the concept described in CRDT literature.
    """

    def __init__(self, product_id: str, location_id: str, initial_value: int = 0):
        self.product_id = product_id
        self.location_id = location_id
        # Dictionary of node_id -> increment count
        self.increments: dict[str, int] = {}
        # Dictionary of node_id -> decrement count
        self.decrements: dict[str, int] = {}

        # If initial value is provided, represent it as increments/decrements from a conceptual 'initial' node
        if initial_value > 0:
            self.increments["initial"] = initial_value
        elif initial_value < 0:
            self.decrements["initial"] = abs(initial_value)

    def increment(self, node_id: str, value: int = 1) -> None:
        """Increment counter by value (default 1) for a given node."""
        if value <= 0:
            raise ValueError("Increment value must be positive")
        self.increments[node_id] = self.increments.get(node_id, 0) + value

    def decrement(self, node_id: str, value: int = 1) -> None:
        """Decrement counter by value (default 1) for a given node."""
        if value <= 0:
            raise ValueError("Decrement value must be positive")
        self.decrements[node_id] = self.decrements.get(node_id, 0) + value

    def value(self) -> int:
        """Get current counter value by summing increments and subtracting decrements."""
        return sum(self.increments.values()) - sum(self.decrements.values())

    def merge(self, other: "PNCounter") -> None:
        """
        Merge another PN Counter state into this one.
        The merge operation takes the maximum count for each node's increments/decrements.
        This operation is commutative, associative, and idempotent.
        Modifies the current counter in-place.
        """
        if not isinstance(other, PNCounter):
            raise TypeError("Can only merge with another PNCounter")

        # Merge increments (take max value for each node)
        all_inc_keys: set[str] = set(self.increments.keys()) | set(
            other.increments.keys()
        )
        for key in all_inc_keys:
            self.increments[key] = max(
                self.increments.get(key, 0), other.increments.get(key, 0)
            )

        # Merge decrements (take max value for each node)
        all_dec_keys: set[str] = set(self.decrements.keys()) | set(
            other.decrements.keys()
        )
        for key in all_dec_keys:
            self.decrements[key] = max(
                self.decrements.get(key, 0), other.decrements.get(key, 0)
            )

    @property
    def state(self) -> dict[str, Any]:
        """Return the full state (increments and decrements) for serialization."""
        return {"p": self.increments, "n": self.decrements}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "product_id": self.product_id,
            "location_id": self.location_id,
            "increments": self.increments,
            "decrements": self.decrements,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PNCounter":
        """Create from dictionary representation."""
        if not all(
            k in data for k in ["product_id", "location_id", "increments", "decrements"]
        ):
            raise ValueError("Invalid data format for PNCounter.from_dict")

        counter = cls(data["product_id"], data["location_id"])
        # Ensure increments/decrements are dicts, provide default empty dict
        counter.increments = (
            data.get("increments", {})
            if isinstance(data.get("increments", {}), dict)
            else {}
        )
        counter.decrements = (
            data.get("decrements", {})
            if isinstance(data.get("decrements", {}), dict)
            else {}
        )
        return counter
