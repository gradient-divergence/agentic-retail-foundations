import pytest

# Module to test
from connectors.dummy_planogram_db import DummyPlanogramDB


@pytest.fixture
def planogram_db() -> DummyPlanogramDB:
    """Provides a DummyPlanogramDB instance for testing."""
    return DummyPlanogramDB()


# --- Test get_section_camera --- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location_id, section_id, expected_camera_id",
    [
        ("LOC1", "SEC001", "CAM01"),
        ("LOC1", "SEC002", "CAM02"),
        ("LOC1", "SEC_UNKNOWN", None),
        ("LOC_UNKNOWN", "SEC001", None),
    ],
)
async def test_get_section_camera(planogram_db, location_id, section_id, expected_camera_id):
    """Test getting camera ID for known and unknown sections/locations."""
    camera_id = await planogram_db.get_section_camera(location_id, section_id)
    assert camera_id == expected_camera_id


# --- Test get_section_planogram --- #


@pytest.mark.asyncio
async def test_get_section_planogram_found(planogram_db):
    """Test getting planogram for a known section."""
    location_id = "LOC1"
    section_id = "SEC001"
    planogram = await planogram_db.get_section_planogram(location_id, section_id)

    assert planogram is not None
    assert isinstance(planogram, dict)
    assert "products" in planogram
    assert isinstance(planogram["products"], list)
    assert len(planogram["products"]) == 3  # Based on dummy data
    assert planogram["products"][0]["product_id"] == "S1"


@pytest.mark.asyncio
async def test_get_section_planogram_found_other(planogram_db):
    """Test getting planogram for another known section."""
    location_id = "LOC1"
    section_id = "SEC002"
    planogram = await planogram_db.get_section_planogram(location_id, section_id)

    assert planogram is not None
    assert len(planogram["products"]) == 2  # Based on dummy data
    assert planogram["products"][0]["product_id"] == "S4"


@pytest.mark.asyncio
async def test_get_section_planogram_not_found(planogram_db):
    """Test getting planogram for an unknown section returns None."""
    location_id = "LOC1"
    section_id = "SEC_UNKNOWN"
    planogram = await planogram_db.get_section_planogram(location_id, section_id)
    assert planogram is None
