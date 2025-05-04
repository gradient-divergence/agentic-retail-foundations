"""
Data models for suppliers.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


class SupplierRating(Enum):
    """
    Enum for supplier rating levels.
    Higher value can indicate better rating.
    """

    PREFERRED = 3
    STANDARD = 2
    PROVISIONAL = 1
    # Add more granular levels if needed


class SupplierStatus(Enum):
    """
    Operational status of a supplier.
    """

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    UNDER_REVIEW = "UNDER_REVIEW"
    DISQUALIFIED = "DISQUALIFIED"


# Forward declaration hint if SupplierBid is defined elsewhere or used only as type hint
# class SupplierBid: pass


@dataclass
class Supplier:
    """
    Represents a supplier entity with capabilities and performance factors.
    """

    supplier_id: str
    name: str
    rating: SupplierRating
    product_capabilities: List[str] = field(default_factory=list)
    # Performance factors (example: lower value is better/faster/cheaper)
    cost_factor: float = 1.0
    speed_factor: float = 1.0
    quality_factor: float = 1.0  # Could represent defect rate, lower is better
    status: SupplierStatus = SupplierStatus.ACTIVE
    # current_bids: Dict[str, 'SupplierBid'] = field(default_factory=dict) # Might belong elsewhere
    contact_email: Optional[str] = None
    address: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.rating, SupplierRating):
            raise TypeError("Supplier rating must be a SupplierRating Enum member.")
        if not isinstance(self.status, SupplierStatus):
            raise TypeError("Supplier status must be a SupplierStatus Enum member.")
        # Add validation for factors if needed (e.g., must be positive)
        if self.cost_factor <= 0 or self.speed_factor <= 0 or self.quality_factor <= 0:
            print(
                f"Warning: Supplier factors for {self.name} should ideally be positive."
            )

    def can_supply(self, product_id: str) -> bool:
        """
        Check if the supplier lists the given product_id in their capabilities.
        """
        return product_id in self.product_capabilities
