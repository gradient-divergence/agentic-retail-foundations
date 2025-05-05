import pytest
import asyncio

# Module to test
from connectors.dummy_db import DummyDB

@pytest.fixture
def db() -> DummyDB:
    """Provides a DummyDB instance for testing."""
    return DummyDB()

# --- Test get_customer --- #

@pytest.mark.asyncio
async def test_get_customer_found(db):
    """Test getting a known customer."""
    cid = "C123"
    customer = await db.get_customer(cid)
    assert customer is not None
    assert customer["name"] == "Alice"
    assert customer["loyalty_tier"] == "Gold"

@pytest.mark.asyncio
async def test_get_customer_not_found(db):
    """Test getting an unknown customer returns default."""
    cid = "C_UNKNOWN"
    customer = await db.get_customer(cid)
    assert customer is not None
    assert customer["name"] == f"Cust {cid}"
    assert customer["loyalty_tier"] == "Standard"

# --- Test search_products --- #

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, expected_pids",
    [
        ("P3", ["P3"]), # Exact ID
        ("Yoga Mat Deluxe", ["P3"]), # Exact Name
        ("Shoe", ["P2"]), # Partial Name
        ("running shoes", ["P2"]), # Case-insensitive Name
        ("bottle", ["P4"]), # Partial Name
        ("p", ["P2", "P3", "P4"]), # Partial ID/Name (matches all)
        ("deluxe", ["P3"]), # Partial Name
        ("XYZ", []), # No match
        ("", ["P2", "P3", "P4"]), # Empty query matches all?
    ]
)
async def test_search_products(db, query, expected_pids):
    """Test product search with various queries."""
    results = await db.search_products(query)
    assert isinstance(results, list)
    found_pids = sorted([p["product_id"] for p in results])
    assert found_pids == sorted(expected_pids)

# --- Test get_product --- #

@pytest.mark.asyncio
async def test_get_product_found(db):
    """Test getting a known product."""
    pid = "P2"
    product = await db.get_product(pid)
    assert product is not None
    assert product["product_id"] == pid
    assert product["name"] == "Running Shoes"

@pytest.mark.asyncio
async def test_get_product_not_found(db):
    """Test getting an unknown product."""
    pid = "P_UNKNOWN"
    product = await db.get_product(pid)
    assert product is None

# --- Test get_inventory --- #

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pid, expected_status",
    [
        ("P3", "In Stock"),
        ("P2", "Low Stock"),
        ("P_UNKNOWN", "Unknown"),
    ]
)
async def test_get_inventory(db, pid, expected_status):
    """Test getting inventory status for known and unknown products."""
    inventory = await db.get_inventory(pid)
    assert inventory is not None
    assert inventory["status"] == expected_status

# --- Test resolve_product_id --- #

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identifier, expected_pid",
    [
        ("P4", "P4"), # Exact ID match
        ("P_UNKNOWN", None), # Unknown ID
        ("Water Bottle", "P4"), # Exact name match
        ("yoga mat deluxe", "P3"), # Case-insensitive name match
        ("shoes", "P2"), # Partial name match
        ("mat", "P3"), # Partial name match
        ("XYZ", None), # No match
    ]
)
async def test_resolve_product_id(db, identifier, expected_pid):
    """Test resolving product identifiers (ID or name substring)."""
    resolved_pid = await db.resolve_product_id(identifier)
    assert resolved_pid == expected_pid 