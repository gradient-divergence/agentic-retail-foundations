"""
Data model for retail store entities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from models.inventory import InventoryPosition, InventoryStatus


@dataclass
class StoreLocation:
    """Geographic and address information for a store."""

    city: str
    state: Optional[str] = None
    postal_code: Optional[str] = None
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    country: str = "USA"
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def get_formatted_address(self) -> str:
        """Return a formatted address string."""
        parts = []
        if self.address_line1:
            parts.append(self.address_line1)
        if self.address_line2:
            parts.append(self.address_line2)

        city_state = []
        if self.city:
            city_state.append(self.city)
        if self.state:
            city_state.append(self.state)
        if city_state:
            parts.append(", ".join(city_state))

        if self.postal_code:
            parts.append(self.postal_code)

        if self.country:
            parts.append(self.country)

        return "\n".join(parts)


class StoreType(str, Enum):
    """The type or format of a retail store."""

    FLAGSHIP = "FLAGSHIP"
    STANDARD = "STANDARD"
    EXPRESS = "EXPRESS"
    OUTLET = "OUTLET"
    POPUP = "POPUP"
    WAREHOUSE = "WAREHOUSE"
    ONLINE = "ONLINE"


@dataclass
class Store:
    """
    Represents a retail store with inventory, location, and transfer capabilities.
    Used primarily for inventory collaboration between stores.
    """

    store_id: str
    name: str
    location: str  # Simple location string for now, could use StoreLocation
    initial_cooperation_score: float = 1.0
    store_type: Optional[StoreType] = None
    transfer_cost_factor: float = 1.0
    capacity: Optional[int] = None
    opening_date: Optional[datetime] = None
    inventory: Dict[str, InventoryPosition] = field(default_factory=dict)
    transfer_history: List[Dict[str, Any]] = field(default_factory=list)
    cooperation_score: float = 1.0

    def __post_init__(self):
        # Set cooperation_score to initial value
        self.cooperation_score = self.initial_cooperation_score

    def add_product(
        self,
        product_id: str,
        current_stock: int,
        target_stock: int,
        sales_rate_per_day: float,
    ):
        """
        Add a product to the store's inventory or update if exists.

        Args:
            product_id: Unique identifier for the product
            current_stock: Current inventory level
            target_stock: Desired inventory level
            sales_rate_per_day: Average daily sales
        """
        self.inventory[product_id] = InventoryPosition(
            product_id=product_id,
            current_stock=current_stock,
            target_stock=target_stock,
            daily_sales_rate=sales_rate_per_day,
        )

    def update_sales_rate(self, product_id: str, new_rate: float):
        """
        Update the sales rate for a product.

        Args:
            product_id: ID of the product to update
            new_rate: New daily sales rate
        """
        if product_id in self.inventory:
            self.inventory[product_id].daily_sales_rate = new_rate
            self.inventory[product_id].last_updated = datetime.now()

    def get_inventory_status(self, product_id: str) -> Optional[InventoryStatus]:
        """Get the inventory status enum for a product."""
        if product_id not in self.inventory:
            return None
        return self.inventory[product_id].get_status()

    def get_sharable_inventory(self) -> Dict[str, int]:
        """Return a dict of product_id to excess units for products with excess inventory."""
        sharable = {}
        for pid, pos in self.inventory.items():
            excess = pos.excess_units()
            if excess > 0:
                sharable[pid] = excess
        return sharable

    def get_needed_inventory(self) -> Dict[str, int]:
        """Return a dict of product_id to needed units for products with low or critical inventory."""
        needed = {}
        for pid, pos in self.inventory.items():
            if pos.get_status() in [InventoryStatus.LOW, InventoryStatus.CRITICAL]:
                needed[pid] = pos.needed_units()
        return needed

    def can_transfer(self, product_id: str, quantity: int) -> bool:
        """Check if the store can transfer out the given quantity of a product."""
        if product_id not in self.inventory:
            return False
        return self.inventory[product_id].excess_units() >= quantity

    def execute_transfer(
        self, product_id: str, quantity: int, partner_id: str, is_sending: bool = True
    ) -> bool:
        """
        Execute a transfer of inventory to or from another store.

        Args:
            product_id: ID of the product being transferred
            quantity: Number of units to transfer
            partner_id: ID of the partner store
            is_sending: True if sending to partner, False if receiving

        Returns:
            bool: Success or failure of the transfer
        """
        if product_id not in self.inventory:
            return False

        position = self.inventory[product_id]

        # Validate the transfer is valid
        if is_sending:
            if not self.can_transfer(product_id, quantity):
                return False
            position.current_stock -= quantity
            direction = "out"
            # Cooperation score increases when helping others
            self.cooperation_score = min(1.5, self.cooperation_score + 0.05)
        else:
            position.current_stock += quantity
            direction = "in"

        # Record the transfer in history
        self.transfer_history.append(
            {
                "timestamp": datetime.now(),
                "product_id": product_id,
                "quantity": quantity,
                "direction": direction,
                "partner_store": partner_id,
            }
        )
        return True

    def calculate_transfer_value(
        self, product_id: str, quantity: int, is_sending: bool
    ) -> float:
        """
        Calculate the value/benefit of transferring a product.
        Positive values indicate beneficial transfers, negative values indicate harmful ones.
        Used to decide whether transfers should be approved.

        Args:
            product_id: ID of product to transfer
            quantity: Number of units
            is_sending: True if sending, False if receiving

        Returns:
            float: Value score (higher is better)
        """
        if product_id not in self.inventory:
            return 0.0

        pos = self.inventory[product_id]

        if is_sending:
            # Sending logic - considers days of supply
            days_supply = pos.days_of_supply()
            if days_supply < 7:
                return -10.0 * quantity  # Strongly negative if low supply
            elif days_supply < 14:
                return -1.0 * quantity  # Slightly negative if moderate supply
            else:
                return 2.0 * quantity  # Positive if excess supply
        else:
            # Receiving logic - considers inventory status
            status = pos.get_status()
            if status == InventoryStatus.CRITICAL:
                return 20.0 * quantity  # Very valuable if critical
            elif status == InventoryStatus.LOW:
                return 10.0 * quantity  # Valuable if low
            else:
                return 0.0  # Neutral otherwise

    def __str__(self) -> str:
        return f"Store(id={self.store_id}, name={self.name}, location={self.location})"
