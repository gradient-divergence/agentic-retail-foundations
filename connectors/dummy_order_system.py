"""
Module: connectors.dummy_order_system

Provides a dummy in-memory order management system for testing agents.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class DummyOrderSystem:
    """
    Dummy order management system connector for demonstration purposes.
    """

    _orders: dict[str, dict[str, Any]] = {
        "ORD987": {
            "order_id": "ORD987",
            "customer_id": "C123",
            "order_date": "2023-10-20",
            "status": "Delivered",
            "items": [{"name": "Yoga Mat Deluxe"}],
            "delivery_date": "2023-10-21",
        },
        "ORD988": {
            "order_id": "ORD988",
            "customer_id": "C123",
            "order_date": "2023-10-25",
            "status": "Shipped",
            "items": [{"name": "Running Shoes"}, {"name": "Water Bottle"}],
            "est_delivery": "2023-11-01",
            "tracking": "TRK123",
        },
        "ORD999": {
            "order_id": "ORD999",
            "customer_id": "C456",
            "order_date": "2023-10-28",
            "status": "Delayed",
            "items": [{"name": "Water Bottle"}],
            "est_delivery": "2023-11-03",
        },
    }
    _return_eligibility = {
        "O1001": {"eligible": True},
        "O1002": {"eligible": False},
        "O2001": {"eligible": True},
    }
    _return_policy = {
        "return_window_days": 30,
        "reason_required": True,
        "restocking_fee": 5.0,
        "return_methods": ["in_store", "mail"],
    }

    async def get_recent_orders(self, cid: str, limit: int = 3) -> list[dict[str, Any]]:
        """Get recent orders for a customer."""
        await asyncio.sleep(0.02)
        cust_orders = [o for o in self._orders.values() if o.get("customer_id") == cid]
        cust_orders.sort(
            key=lambda x: date.fromisoformat(x.get("order_date", "1900-01-01")),
            reverse=True,
        )
        return cust_orders[:limit]

    async def get_order_details(self, oid: str) -> dict[str, Any] | None:
        """Get details for a specific order."""
        await asyncio.sleep(0.02)
        return self._orders.get(oid)

    async def check_return_eligibility(self, oid: str) -> dict[str, Any]:
        """Check if an order is eligible for return."""
        await asyncio.sleep(0.01)
        order = self._orders.get(oid)
        if not order:
            return {"eligible": False, "reason": "Order not found."}

        delivery_date_str = order.get("delivery_date")
        if order["status"] == "Delivered" and isinstance(delivery_date_str, str):
            try:
                delivery_dt = datetime.fromisoformat(delivery_date_str)
                if datetime.now() - delivery_dt <= timedelta(days=30):
                    return {"eligible": True}
                else:
                    return {"eligible": False, "reason": "Past 30-day return window."}
            except Exception:
                return {"eligible": False, "reason": "Invalid delivery date."}
        elif order["status"] == "Shipped":
            return {"eligible": False, "reason": "Not yet delivered."}
        else:
            return {"eligible": False, "reason": f"Status is '{order['status']}'."}

    def get_return_policy(self) -> dict[str, Any]:
        """Get the return policy."""
        return self._return_policy

    async def report_visual_audit(self, issue_summary: dict[str, Any]):
        """Report a visual audit issue (dummy implementation)."""
        logger.info(f"DUMMY: Received visual audit: {issue_summary.get('section_id')}")
        await asyncio.sleep(0.01)
