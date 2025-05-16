import pytest

# Module to test
from connectors.dummy_inventory_system import DummyInventorySystem


@pytest.fixture(autouse=True)
def clear_class_state():
    """Fixture to clear the class-level state before each test."""
    # Clear the list before each test run
    DummyInventorySystem._audit_reports.clear()
    yield
    # Clear again after test run (optional, but good practice)
    DummyInventorySystem._audit_reports.clear()


@pytest.fixture
def inventory_system() -> DummyInventorySystem:
    """Provides a DummyInventorySystem instance for testing."""
    return DummyInventorySystem()


# --- Test Initialization (mostly about class state) --- #


def test_initial_state(inventory_system):
    """Verify initial class state is empty."""
    # The fixture already clears this, but double-check
    assert DummyInventorySystem._audit_reports == []


# --- Test report_visual_audit --- #


@pytest.mark.asyncio
async def test_report_visual_audit_appends_report(inventory_system):
    """Test that report_visual_audit appends the report to the class list."""
    report1 = {"shelf_id": "A1", "items": [{"sku": "P123", "count": 5}]}
    await inventory_system.report_visual_audit(report1)

    assert len(DummyInventorySystem._audit_reports) == 1
    assert DummyInventorySystem._audit_reports[0] == report1

    report2 = {"shelf_id": "B2", "items": [{"sku": "P456", "count": 0}]}
    await inventory_system.report_visual_audit(report2)

    assert len(DummyInventorySystem._audit_reports) == 2
    assert DummyInventorySystem._audit_reports[1] == report2


@pytest.mark.asyncio
async def test_report_visual_audit_print_output(inventory_system, capsys):
    """Test that report_visual_audit prints the report."""
    report = {"shelf_id": "C3", "items": []}
    await inventory_system.report_visual_audit(report)

    captured = capsys.readouterr()
    assert "[DummyInventorySystem] Visual audit report received:" in captured.out
    assert str(report) in captured.out


@pytest.mark.asyncio
async def test_report_visual_audit_class_state(inventory_system):
    """Test that reports accumulate across different instances (due to class state)."""
    inventory_system_2 = DummyInventorySystem()  # Create a second instance

    report1 = {"instance": 1}
    report2 = {"instance": 2}

    await inventory_system.report_visual_audit(report1)
    await inventory_system_2.report_visual_audit(report2)

    # Check the class variable contains both reports
    assert len(DummyInventorySystem._audit_reports) == 2
    assert report1 in DummyInventorySystem._audit_reports
    assert report2 in DummyInventorySystem._audit_reports
