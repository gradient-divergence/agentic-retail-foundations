"""
Module: connectors.dummy_db

Provides a dummy in-memory database for customers and products for testing agents.
"""

from typing import Any
import asyncio


class DummyDB:
    """
    Dummy database connector for customers and products.
    """

    _customers = {
        "C123": {
            "name": "Alice",
            "loyalty_tier": "Gold",
            "customer_since": "2022-01-15",
        },
        "C456": {"name": "Bob", "loyalty_tier": "Standard"},
    }
    _products = {
        "P3": {
            "product_id": "P3",
            "name": "Yoga Mat Deluxe",
            "price": 39.99,
            "description": "Premium mat.",
            "features": ["Eco-friendly"],
        },
        "P2": {
            "product_id": "P2",
            "name": "Running Shoes",
            "price": 89.99,
            "description": "Lightweight shoes.",
            "features": ["Mesh"],
        },
        "P4": {
            "product_id": "P4",
            "name": "Water Bottle",
            "price": 12.99,
            "description": "Insulated bottle.",
            "features": ["BPA-free"],
        },
    }
    _inventory = {
        "P3": {"status": "In Stock"},
        "P2": {"status": "Low Stock"},
        "P4": {"status": "In Stock"},
    }

    async def get_customer(self, cid: str) -> dict[str, Any]:
        """Get customer info by ID."""
        await asyncio.sleep(0.01)
        return self._customers.get(
            cid, {"name": f"Cust {cid}", "loyalty_tier": "Standard"}
        )

    async def search_products(self, query: str) -> list[dict[str, Any]]:
        """Search for products by name or ID (simple substring match)."""
        await asyncio.sleep(0.01)
        results = []
        for p in self._products.values():
            if (
                query.lower() in p["name"].lower()
                or query.lower() in p["product_id"].lower()
            ):
                results.append(p)
        return results

    async def get_product(self, pid: str) -> dict[str, Any] | None:
        """Get product details by ID."""
        await asyncio.sleep(0.01)
        return self._products.get(pid)

    async def get_inventory(self, pid: str) -> dict[str, Any]:
        """Get inventory status for a product."""
        await asyncio.sleep(0.01)
        return self._inventory.get(pid, {"status": "Unknown"})

    async def resolve_product_id(self, identifier: str) -> str | None:
        """Resolve a product ID or name to a product ID."""
        await asyncio.sleep(0.01)
        if identifier in self._products:
            return identifier
        for pid, data in self._products.items():
            if identifier.lower() in data["name"].lower():
                return pid
        return None
