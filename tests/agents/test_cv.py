import pytest
import numpy as np
import logging
from unittest.mock import patch, MagicMock, ANY, AsyncMock
from datetime import datetime

# Module to test
from agents.cv import ShelfMonitoringAgent
from agents.sensor import SensorDataProcessor

# It's often cleaner to apply patches within the tests that need them
# rather than using global fixtures, especially for complex libraries.

# --- Fixtures --- #

@pytest.fixture
def mock_planogram_db() -> MagicMock:
    db = MagicMock()
    db.get_section_camera = AsyncMock()
    db.get_section_planogram = AsyncMock()
    return db

@pytest.fixture
def mock_inventory_system() -> MagicMock:
    inv = MagicMock()
    inv.report_visual_audit = AsyncMock()
    inv.get_current_price = AsyncMock()
    # Add other methods as needed by tests
    return inv

@pytest.fixture
def agent_params(mock_planogram_db, mock_inventory_system) -> dict:
    return {
        "model_path": "fake/model/path",
        "planogram_database": mock_planogram_db,
        "inventory_system": mock_inventory_system,
        "camera_stream_urls": {"CAM01": "rtsp://fake"},
        "confidence_threshold": 0.7,
    }

@pytest.fixture
@patch('agents.cv.tf.saved_model.load') # Patch model loading for the agent fixture
def shelf_monitoring_agent(mock_tf_load, agent_params) -> ShelfMonitoringAgent:
    """Fixture to create a ShelfMonitoringAgent instance with mocked dependencies."""
    # Set up a mock model instance to be returned by the patched load function
    mock_model_instance = MagicMock()
    # Configure the mock model to be callable and return a mock detection result dictionary
    # The actual structure might need adjustment based on how the model is used
    mock_detection_result = {
        "detection_boxes": [MagicMock(numpy=MagicMock(return_value=np.array([[[]]])))],
        "detection_classes": [MagicMock(numpy=MagicMock(return_value=np.array([[]])))],
        "detection_scores": [MagicMock(numpy=MagicMock(return_value=np.array([[]])))],
    }
    mock_model_instance.return_value = mock_detection_result # Mock the call `model(tensor)`
    mock_tf_load.return_value = mock_model_instance

    # Create the agent instance
    agent = ShelfMonitoringAgent(**agent_params)
    # Manually assign the callable mock if needed, depending on the exact model usage pattern
    agent.detection_model = mock_model_instance # Ensure the instance has the callable mock
    return agent

# --- Test Initialization --- #

# Patch the specific tf function used in __init__
@patch('agents.cv.tf.saved_model.load')
def test_agent_initialization_model_load_success(mock_tf_load, agent_params):
    """Test initialization when TF model loads successfully."""
    mock_model_instance = MagicMock()
    mock_tf_load.return_value = mock_model_instance

    agent = ShelfMonitoringAgent(**agent_params)

    mock_tf_load.assert_called_once_with(agent_params["model_path"])
    assert agent.detection_model is mock_model_instance
    assert agent.planogram_db is agent_params["planogram_database"]
    assert agent.inventory_system is agent_params["inventory_system"]
    assert agent.confidence_threshold == agent_params["confidence_threshold"]

# Patch the specific tf function used in __init__
@patch('agents.cv.tf.saved_model.load')
def test_agent_initialization_model_load_fail(mock_tf_load, agent_params, caplog):
    """Test initialization uses dummy model when TF model load fails."""
    mock_tf_load.side_effect = OSError("Load failed")

    with caplog.at_level(logging.WARNING):
        agent = ShelfMonitoringAgent(**agent_params)

    mock_tf_load.assert_called_once_with(agent_params["model_path"])
    # Check if the dummy model class is instantiated (check type)
    assert agent.detection_model is not None
    assert type(agent.detection_model).__name__ == '_DummyModel'
    assert "Could not load model" in caplog.text
    assert "Falling back to a dummy detection model" in caplog.text

# --- Test Helper Methods --- #

# Patch cv2 and tf functions directly within the test
@patch('agents.cv.cv2')
@patch('agents.cv.tf')
def test_preprocess_image(mock_tf, mock_cv2, agent_params):
    """Test image preprocessing steps."""
    # Configure mocks
    mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
    mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
    mock_tf.expand_dims.return_value = np.expand_dims(np.zeros((640, 640, 3)), 0)

    agent = ShelfMonitoringAgent(**agent_params)
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor = agent._preprocess_image(dummy_image)

    mock_cv2.resize.assert_called_once_with(dummy_image, (640, 640))
    mock_cv2.cvtColor.assert_called_once_with(ANY, mock_cv2.COLOR_BGR2RGB)
    mock_tf.expand_dims.assert_called_once()
    assert tensor.shape == (1, 640, 640, 3)

@patch.object(ShelfMonitoringAgent, '_get_class_mapping')
def test_process_detections(mock_get_mapping, agent_params):
    """Test processing of raw model detections."""
    # Create agent instance *inside* the test to ensure params are set correctly
    agent = ShelfMonitoringAgent(**agent_params)
    agent.confidence_threshold = 0.6 # Override default if needed for test case

    mock_get_mapping.return_value = {1: "SKU_A", 2: "SKU_B"}

    # Mock tensor-like objects with .numpy()
    boxes_tensor_mock = MagicMock()
    boxes_tensor_mock.numpy.return_value = np.array([[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]], dtype=np.float32)
    classes_tensor_mock = MagicMock()
    classes_tensor_mock.numpy.return_value = np.array([1, 2], dtype=np.int32)
    scores_tensor_mock = MagicMock()
    scores_tensor_mock.numpy.return_value = np.array([0.8, 0.5], dtype=np.float32)

    # The dictionary values should be LISTS containing ONE tensor-like mock each
    mock_detections = {
        "detection_boxes": [boxes_tensor_mock],
        "detection_classes": [classes_tensor_mock],
        "detection_scores": [scores_tensor_mock],
    }

    img_h, img_w = 480, 640
    processed = agent._process_detections(mock_detections, img_w, img_h)

    # Assertions
    assert len(processed) == 1 # Only detection 1 (score 0.8) should pass threshold 0.6
    p1 = processed[0]
    assert p1["product_id"] == "SKU_A"
    assert p1["confidence"] == pytest.approx(0.8)
    expected_box = [int(0.1*img_h), int(0.1*img_w), int(0.5*img_h), int(0.5*img_w)]
    assert p1["bounding_box"] == expected_box
    assert p1["shelf_position"]["x"] == pytest.approx((0.1 + 0.5) / 2)
    assert p1["shelf_position"]["y"] == pytest.approx((0.1 + 0.5) / 2)

# --- Test Planogram Comparison --- #

@pytest.fixture
def sample_planogram() -> dict:
    return {
        "products": [
            {"product_id": "SKU_A", "expected_count": 5, "position": {"x": 0.2, "y": 0.3}},
            {"product_id": "SKU_B", "expected_count": 3, "position": {"x": 0.6, "y": 0.3}},
            {"product_id": "SKU_C", "expected_count": 4, "position": {"x": 0.4, "y": 0.7}},
        ]
    }

@pytest.mark.parametrize(
    "test_id, detected_products, expected_issues_types_pids",
    [
        # Case 1: Perfect match
        (
            "perfect_match",
            [
                # SKU_A (Count=5, Pos=~0.2,0.3)
                {"product_id": "SKU_A", "shelf_position": {"x": 0.21, "y": 0.31}}, # Pos OK
                {"product_id": "SKU_A", "shelf_position": {"x": 0.19, "y": 0.29}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.20, "y": 0.30}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.22, "y": 0.32}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.18, "y": 0.28}},
                # SKU_B (Count=3, Pos=~0.6,0.3)
                {"product_id": "SKU_B", "shelf_position": {"x": 0.61, "y": 0.31}},
                {"product_id": "SKU_B", "shelf_position": {"x": 0.59, "y": 0.29}},
                {"product_id": "SKU_B", "shelf_position": {"x": 0.60, "y": 0.30}},
                # SKU_C (Count=4, Pos=~0.4,0.7)
                {"product_id": "SKU_C", "shelf_position": {"x": 0.41, "y": 0.71}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.39, "y": 0.69}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.40, "y": 0.70}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.42, "y": 0.72}},
            ],
            {} # No issues expected
        ),
        # Case 2: Low stock SKU_A
        (
            "low_stock",
            [
                # SKU_A (Count=3, Expected=5)
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                # Other products OK
                {"product_id": "SKU_B", "shelf_position": {"x": 0.6, "y": 0.3}},
                {"product_id": "SKU_B", "shelf_position": {"x": 0.6, "y": 0.3}},
                {"product_id": "SKU_B", "shelf_position": {"x": 0.6, "y": 0.3}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
            ],
            {("LOW_STOCK", "SKU_A")} # Expect low stock for SKU_A
        ),
        # Case 3: Out of stock SKU_B
        (
            "out_of_stock",
            [
                # SKU_A OK
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                # SKU_B Missing (Expected=3)
                # SKU_C OK
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
            ],
            {("OUT_OF_STOCK", "SKU_B")} # Expect OOS for SKU_B
        ),
        # Case 4: Unexpected product
        (
            "unexpected_product",
            [
                # SKU_A, B, C OK
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_B", "shelf_position": {"x": 0.6, "y": 0.3}},
                {"product_id": "SKU_B", "shelf_position": {"x": 0.6, "y": 0.3}},
                {"product_id": "SKU_B", "shelf_position": {"x": 0.6, "y": 0.3}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                # Unexpected SKU_X
                {"product_id": "SKU_X", "shelf_position": {"x": 0.5, "y": 0.5}},
            ],
            {("UNEXPECTED_PRODUCT", "SKU_X")} # Expect unexpected SKU_X
        ),
        # Case 5: Misplaced product
        (
            "misplaced_product",
            [
                 # SKU_A OK
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                # SKU_B OK
                {"product_id": "SKU_B", "shelf_position": {"x": 0.6, "y": 0.3}},
                {"product_id": "SKU_B", "shelf_position": {"x": 0.6, "y": 0.3}},
                {"product_id": "SKU_B", "shelf_position": {"x": 0.6, "y": 0.3}},
                # SKU_C OK count, but one is misplaced (pos=0.8,0.8 instead of 0.4,0.7)
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.8, "y": 0.8}}, # Misplaced!
            ],
            {("MISPLACED_PRODUCT", "SKU_C")} # Expect misplaced SKU_C
        ),
        # Case 6: Multiple issues
        (
            "multiple_issues",
            [
                # SKU_A Low Stock (2 detected, 5 expected)
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                {"product_id": "SKU_A", "shelf_position": {"x": 0.2, "y": 0.3}},
                # SKU_B Misplaced (1 detected at wrong pos, 3 expected)
                {"product_id": "SKU_B", "shelf_position": {"x": 0.1, "y": 0.1}},
                # SKU_C OK
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                {"product_id": "SKU_C", "shelf_position": {"x": 0.4, "y": 0.7}},
                # SKU_X Unexpected
                {"product_id": "SKU_X", "shelf_position": {"x": 0.9, "y": 0.9}},
            ],
            {
                ("LOW_STOCK", "SKU_A"), # Low A
                ("LOW_STOCK", "SKU_B"), # B is too few detected (1 instead of 3)
                ("MISPLACED_PRODUCT", "SKU_B"), # Detected B is misplaced
                ("UNEXPECTED_PRODUCT", "SKU_X"), # Unexpected X
            }
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "" # Use test_id
)
def test_compare_with_planogram(
    shelf_monitoring_agent, sample_planogram,
    test_id: str, detected_products: list, expected_issues_types_pids: set
):
    """Test planogram comparison logic for various scenarios."""
    # Use the agent's class method for the comparison
    issues = shelf_monitoring_agent._compare_with_planogram(detected_products, sample_planogram)

    # Check the types and product IDs of reported issues
    found_issues_set = {(issue["type"], issue["product_id"]) for issue in issues}

    if test_id == "perfect_match":
        # For the perfect match case, assert that the list of issues is empty
        assert not issues, f"Expected no issues for perfect_match, but got: {issues}"
    else:
        # For all other cases, compare the sets directly
        assert found_issues_set == expected_issues_types_pids, f"Issues mismatch in {test_id}:\nExpected: {sorted(list(expected_issues_types_pids))}\nActual: {sorted(list(found_issues_set))}"

# --- Test Report Issues --- #

@pytest.mark.asyncio
async def test_report_issues(shelf_monitoring_agent: ShelfMonitoringAgent, mock_inventory_system: AsyncMock, capsys):
    """Test that _report_issues calls inventory system and prints."""
    location_id = "LOC1"
    section_id = "SEC1"
    timestamp = datetime.now().isoformat()
    issues = [
        {"type": "LOW_STOCK", "product_id": "P1"},
        {"type": "MISPLACED_PRODUCT", "product_id": "P2"},
    ]

    # The agent instance now has the mocked inventory system from agent_params
    await shelf_monitoring_agent._report_issues(location_id, section_id, issues, timestamp)

    # Verify call to the mocked inventory system associated with the agent
    shelf_monitoring_agent.inventory_system.report_visual_audit.assert_awaited_once()
    call_args = shelf_monitoring_agent.inventory_system.report_visual_audit.call_args.args
    expected_summary = {
        "location_id": location_id,
        "section_id": section_id,
        "timestamp": timestamp,
        "issues": issues,
    }
    assert call_args[0] == expected_summary

    # Verify print output
    captured = capsys.readouterr()
    assert f"Detected {len(issues)} issues in section {section_id}" in captured.out
    assert f"- {issues[0]['type']}: {issues[0]['product_id']}" in captured.out
    assert f"- {issues[1]['type']}: {issues[1]['product_id']}" in captured.out

# --- Test _check_section Orchestration --- #

@pytest.mark.asyncio
@patch('agents.cv.cv2.VideoCapture')
@patch.object(ShelfMonitoringAgent, '_preprocess_image')
@patch.object(ShelfMonitoringAgent, '_process_detections')
@patch.object(ShelfMonitoringAgent, '_compare_with_planogram')
@patch.object(ShelfMonitoringAgent, '_report_issues', new_callable=AsyncMock) # Mock as AsyncMock
async def test_check_section_flow_with_issues(
    mock_report, mock_compare, mock_process, mock_preprocess, mock_videocapture,
    shelf_monitoring_agent: ShelfMonitoringAgent, mock_planogram_db: MagicMock
):
    """Test the orchestration flow of _check_section when issues are found."""
    location_id = "L1"
    section_id = "S1"
    camera_id = "CAM01"

    # --- Mock Setup ---
    # Planogram DB
    mock_planogram_db.get_section_camera.return_value = camera_id
    mock_planogram = {"products": []}
    mock_planogram_db.get_section_planogram.return_value = mock_planogram
    # Video Capture
    mock_stream = MagicMock()
    mock_stream.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8)) # Return a valid frame
    mock_videocapture.return_value = mock_stream
    # Preprocess
    mock_tensor = MagicMock()
    mock_preprocess.return_value = mock_tensor
    # Detection Model (The fixture `shelf_monitoring_agent` already has a mocked model)
    mock_detection_result = {"detection_boxes": [], "detection_classes": [], "detection_scores": []}
    # The agent's model is already mocked by the fixture, no need to patch here if fixture does it.
    # Ensure the fixture's mock model returns the desired result for this test.
    shelf_monitoring_agent.detection_model.return_value = mock_detection_result

    # Process Detections
    mock_detected_products = [{"product_id": "P1"}]
    mock_process.return_value = mock_detected_products
    # Compare Planogram
    mock_issues = [{"type": "UNEXPECTED_PRODUCT", "product_id": "P1"}]
    mock_compare.return_value = mock_issues
    # Report Issues (already mocked via @patch)
    mock_report.return_value = None

    # --- Execute --- #
    # Need to manually add the stream as start_monitoring isn't called directly
    shelf_monitoring_agent.active_streams[camera_id] = mock_stream
    await shelf_monitoring_agent._check_section(location_id, section_id, camera_id, mock_stream)

    # --- Assertions --- #
    mock_planogram_db.get_section_planogram.assert_awaited_once_with(location_id, section_id)
    mock_stream.read.assert_called_once()
    mock_preprocess.assert_called_once() # Check frame was passed
    # Assert the agent's mocked model was called
    shelf_monitoring_agent.detection_model.assert_called_once_with(mock_tensor)
    mock_process.assert_called_once_with(mock_detection_result, 100, 100) # Check detections and frame shape
    mock_compare.assert_called_once_with(mock_detected_products, mock_planogram)
    mock_report.assert_awaited_once() # Check issues and timestamp passed
    assert mock_report.await_args.args[0] == location_id
    assert mock_report.await_args.args[1] == section_id
    assert mock_report.await_args.args[2] == mock_issues
    assert isinstance(mock_report.await_args.args[3], str) # Timestamp

@pytest.mark.asyncio
@patch('agents.cv.cv2.VideoCapture')
@patch.object(ShelfMonitoringAgent, '_preprocess_image')
@patch.object(ShelfMonitoringAgent, '_process_detections')
@patch.object(ShelfMonitoringAgent, '_compare_with_planogram')
@patch.object(ShelfMonitoringAgent, '_report_issues', new_callable=AsyncMock) # Mock as AsyncMock
async def test_check_section_flow_no_issues(
    mock_report, mock_compare, mock_process, mock_preprocess, mock_videocapture,
    shelf_monitoring_agent: ShelfMonitoringAgent, mock_planogram_db: MagicMock
):
    """Test the orchestration flow of _check_section when no issues are found."""
    location_id = "L1"
    section_id = "S1"
    camera_id = "CAM01"

    # Mock Setup (similar to above, but compare returns no issues)
    mock_planogram_db.get_section_camera.return_value = camera_id
    mock_planogram = {"products": []}
    mock_planogram_db.get_section_planogram.return_value = mock_planogram
    mock_stream = MagicMock()
    mock_stream.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
    mock_videocapture.return_value = mock_stream
    mock_preprocess.return_value = MagicMock()
    # Ensure the agent's mocked model returns the desired result
    shelf_monitoring_agent.detection_model.return_value = {}
    mock_process.return_value = []
    mock_compare.return_value = [] # <--- No issues found

    shelf_monitoring_agent.active_streams[camera_id] = mock_stream
    await shelf_monitoring_agent._check_section(location_id, section_id, camera_id, mock_stream)

    # Assertions
    mock_planogram_db.get_section_planogram.assert_awaited_once()
    mock_stream.read.assert_called_once()
    mock_preprocess.assert_called_once()
    shelf_monitoring_agent.detection_model.assert_called_once() # Assert model was called
    mock_process.assert_called_once()
    mock_compare.assert_called_once()
    mock_report.assert_not_awaited() # Report should NOT be called

# Placeholder tests
# def test_report_issues(): ...
# def test_check_section(): ... 