"""Compact orient-phase excerpt for the BDI inventory agent."""

# region book:bdi-orient-excerpt
from pydantic import BaseModel

from models.inventory import InventoryItem, ProductInfo, SalesData


class OrientSummary(BaseModel):
    inventory_status: str
    projected_daily_sales: float
    days_of_supply: float


def orient_product(product: ProductInfo, inventory: InventoryItem, sales: SalesData) -> OrientSummary:
    avg_sales = sales.average_daily_sales()
    projected_daily_sales = max(avg_sales * (1.0 + sales.trend()), 0.1)
    days_of_supply = inventory.current_stock / projected_daily_sales

    if days_of_supply <= product.lead_time_days + 3:
        inventory_status = "low_stock_risk"
    elif days_of_supply >= product.lead_time_days + 14:
        inventory_status = "overstock_risk"
    else:
        inventory_status = "balanced"

    return OrientSummary(
        inventory_status=inventory_status,
        projected_daily_sales=projected_daily_sales,
        days_of_supply=days_of_supply,
    )


# endregion book:bdi-orient-excerpt
