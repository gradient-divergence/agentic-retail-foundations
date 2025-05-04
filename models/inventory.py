"""
Inventory-related data models for agentic-retail-foundations.
Includes ProductInfo, InventoryItem, and SalesData dataclasses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


@dataclass
class ProductInfo:
    """
    Data model for product information, including pricing, cost, supplier, and inventory details.
    """

    product_id: str
    name: str
    category: str
    price: float
    cost: float
    lead_time_days: int
    shelf_life_days: int | None = None
    supplier_id: str = ""
    alternative_suppliers: list[str] = field(default_factory=list)
    min_order_quantity: int = 1
    current_price: float = field(init=False)
    competitor_prices: dict[str, float] = field(default_factory=dict)
    sales_last_7_days: list[int] = field(default_factory=list)
    inventory: int = 0

    def __post_init__(self):
        self.current_price = self.price


@dataclass
class InventoryItem:
    """
    Data model for inventory item state.
    """

    product_id: str
    current_stock: int
    reorder_point: int
    optimal_stock: int
    last_reorder_date: datetime | None = None
    expected_delivery_date: datetime | None = None
    pending_order_quantity: int = 0


@dataclass
class SalesData:
    """
    Data model for sales data, including daily sales and trend calculation.
    """

    product_id: str
    daily_sales: list[int]

    def average_daily_sales(self) -> float:
        if not self.daily_sales:
            return 0.0
        return float(sum(self.daily_sales)) / max(1, len(self.daily_sales))

    def trend(self) -> float:
        """Calculate a simplistic sales trend (positive = increasing, negative = decreasing)."""
        if len(self.daily_sales) < 14:
            return 0.0
        recent_week_sales = sum(self.daily_sales[-7:])
        previous_week_sales = sum(self.daily_sales[-14:-7])
        if previous_week_sales == 0:
            return 1.0 if recent_week_sales > 0 else 0.0
        return (recent_week_sales - previous_week_sales) / previous_week_sales


class InventoryStatus(Enum):
    """Enumeration of inventory status levels for a product in a store."""

    CRITICAL = "CRITICAL"
    LOW = "LOW"
    ADEQUATE = "ADEQUATE"
    EXCESS = "EXCESS"


@dataclass
class InventoryPosition:
    """Represents the inventory position for a product in a store."""

    product_id: str
    current_stock: int
    target_stock: int
    daily_sales_rate: float
    last_updated: datetime = datetime.now()

    def get_status(self) -> InventoryStatus:
        """Return the inventory status based on current stock and target stock."""
        ratio = self.current_stock / self.target_stock
        if ratio < 0.3:
            return InventoryStatus.CRITICAL
        elif ratio < 0.8:
            return InventoryStatus.LOW
        elif ratio > 1.2:
            return InventoryStatus.EXCESS
        else:
            return InventoryStatus.ADEQUATE

    def excess_units(self) -> int:
        """Return the number of excess units if status is EXCESS, else 0."""
        if self.get_status() == InventoryStatus.EXCESS:
            return self.current_stock - self.target_stock
        return 0

    def needed_units(self) -> int:
        """Return the number of units needed if status is LOW or CRITICAL, else 0."""
        if self.get_status() in [InventoryStatus.LOW, InventoryStatus.CRITICAL]:
            return self.target_stock - self.current_stock
        return 0

    def days_of_supply(self) -> float:
        """Return the number of days of supply based on daily sales rate."""
        if self.daily_sales_rate <= 0:
            return float("inf")
        return self.current_stock / self.daily_sales_rate
