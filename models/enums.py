"""
Centralized Enum definitions for the project.
"""

from enum import Enum


class AgentType(str, Enum):
    """Types of agents in the retail ecosystem"""

    INVENTORY = "inventory"
    PRICING = "pricing"
    FULFILLMENT = "fulfillment"
    CUSTOMER = "customer"
    PAYMENT = "payment"
    DELIVERY = "delivery"
    STORE_OPS = "store_operations"
    WAREHOUSE = "warehouse"
    FINANCIAL = "financial"
    MASTER = "master_orchestrator"
    ORDER_INGESTION = "order_ingestion"
    ORDER_MANAGER = "order_manager"
    VALIDATION = "validation"
    SYSTEM = "system"
    TEST_AGENT = "test_agent"
    # Add other agent types as needed


class FulfillmentMethod(str, Enum):
    """Available order fulfillment methods"""

    SHIP_FROM_STORE = "ship_from_store"
    SHIP_FROM_WAREHOUSE = "ship_from_warehouse"
    PICKUP_IN_STORE = "pickup_in_store"
    DELIVERY_FROM_STORE = "delivery_from_store"
    DROPSHIP_FROM_VENDOR = "dropship_from_vendor"
    CREATED = "created"
    VALIDATED = "validated"
    ALLOCATED = "allocated"
    PAYMENT_PROCESSED = "payment_processed"
    PROCESSING = "processing"
    PICKING = "picking"
    PACKING = "packing"


class OrderStatus(str, Enum):
    """Possible states of a retail order"""

    CREATED = "created"
    VALIDATED = "validated"
    ALLOCATED = "allocated"
    PAYMENT_PROCESSED = "payment_processed"
    PROCESSING = "processing"
    PICKING = "picking"
    PACKING = "packing"
    READY_FOR_PICKUP = "ready_for_pickup"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXCEPTION = "exception"


class InventoryEventType(str, Enum):
    """Types of inventory events"""

    RECEIVED = "RECEIVED"  # New inventory arrived
    SOLD = "SOLD"  # Inventory sold to customer
    ADJUSTED = "ADJUSTED"  # Manual adjustment (e.g., for shrinkage)
    TRANSFERRED = "TRANSFERRED"  # Unified transfer event
    # TRANSFERRED_OUT = "TRANSFERRED_OUT" # Deprecated?
    # TRANSFERRED_IN = "TRANSFERRED_IN" # Deprecated?
    RESERVED = "RESERVED"  # Inventory reserved (e.g., for online order)
    RELEASED = "RELEASED"  # Reserved inventory released back to available


class InventoryChannel(str, Enum):
    """Available inventory channels"""

    STORE = "STORE"  # Physical store
    ONLINE = "ONLINE"  # E-commerce website
    MARKETPLACE = "MARKETPLACE"  # Third-party marketplace
    WAREHOUSE = "WAREHOUSE"  # Distribution center
    POS = "POS"  # Point of sale system
    MOBILE_APP = "MOBILE_APP"  # Mobile application


class ReservationStatus(str, Enum):
    """Possible reservation statuses"""

    ACTIVE = "ACTIVE"
    FULFILLED = "FULFILLED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
