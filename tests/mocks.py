"""Mock objects for testing agent dependencies."""

from unittest.mock import AsyncMock

# Basic mock for the OpenAI client
class MockAsyncOpenAI(AsyncMock):
    pass

# Mock for Product Database
class MockProductDB:
    async def get_product(self, product_id):
        if product_id == "RESOLVED_ID_123":
            # Use normal quotes inside the dictionary
            return {"product_id": product_id, "name": "Test Product", "price": 99.99}
        return None

    async def get_inventory(self, product_id):
        if product_id == "RESOLVED_ID_123":
            return {"product_id": product_id, "stock_level": 50}
        return {"product_id": product_id, "stock_level": 0}

    async def resolve_product_id(self, identifier):
        if identifier == "Test Product" or identifier == "Product ABC":
            return "RESOLVED_ID_123"
        return None

# Mock for Order Management System
class MockOrderSystem:
    async def get_order_details(self, order_id):
        if order_id == "ORD123":
            return {"order_id": order_id, "status": "Shipped", "items": ["item1"]}
        elif order_id == "ORD456":
            return {"order_id": order_id, "status": "Processing", "items": ["item2"]}
        elif order_id == "REGEX999":
             return {"order_id": order_id, "status": "Delivered", "items": ["item3"]}
        return None

    async def get_recent_orders(self, customer_id, limit=3):
        return [
            {"order_id": "ORD123", "status": "Shipped"},
            {"order_id": "ORD456", "status": "Processing"},
        ]

    async def check_return_eligibility(self, order_id):
        # Use normal quotes
        return {"eligible": True, "reason": None} if order_id == "ORD123" else {"eligible": False, "reason": "Too old"}

# Mock for Customer Database
class MockCustomerDB:
    async def get_customer(self, customer_id):
        # Use normal quotes
        return {"customer_id": customer_id, "name": "Test Customer", "loyalty_tier": "Gold"}

    async def log_interaction(self, *args, **kwargs):
        pass # No operation needed for current tests
