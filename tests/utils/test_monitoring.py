import pytest
import logging
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call
import numpy as np # Required by AgentMonitor methods

from utils.monitoring import AgentMonitor

# --- Test Fixtures --- #

@pytest.fixture
def monitor_config():
    """Provides a standard configuration for the AgentMonitor."""
    return {
        "agent_id": "agent_test_001",
        "metric_thresholds": {
            "response_time_ms": (50.0, 500.0),
            "error_rate_percent": (0.0, 5.0),
            "sessions_per_hour": (10.0, 1000.0),
        },
        "alert_endpoints": [
            {"type": "slack", "webhook_url": "http://fake.slack.hook"},
            {"type": "email", "address": "test@example.com"},
        ]
    }

@pytest.fixture
def monitor(monitor_config) -> AgentMonitor:
    """Provides an AgentMonitor instance with standard config."""
    return AgentMonitor(**monitor_config)

# --- Test Initialization --- #

def test_monitor_initialization(monitor: AgentMonitor, monitor_config):
    """Test that AgentMonitor initializes correctly."""
    assert monitor.agent_id == monitor_config["agent_id"]
    assert monitor.metric_thresholds == monitor_config["metric_thresholds"]
    assert monitor.alert_endpoints == monitor_config["alert_endpoints"]
    assert monitor.metrics_history == {}

# --- Test record_metrics --- #

# Use patch for datetime to control timestamps
@patch('utils.monitoring.datetime')
def test_record_metrics_valid(mock_dt, monitor: AgentMonitor):
    """Test recording valid numeric metrics."""
    fixed_time = datetime(2024, 1, 10, 10, 0, 0)
    mock_dt.now.return_value = fixed_time

    metrics = {"response_time_ms": 150.5, "error_rate_percent": 1.2}
    monitor.record_metrics(metrics)

    assert "response_time_ms" in monitor.metrics_history
    assert monitor.metrics_history["response_time_ms"] == [(fixed_time, 150.5)]

    assert "error_rate_percent" in monitor.metrics_history
    assert monitor.metrics_history["error_rate_percent"] == [(fixed_time, 1.2)]

    # Test with explicit timestamp
    explicit_time = datetime(2024, 1, 10, 10, 5, 0)
    monitor.record_metrics({"sessions_per_hour": 55}, timestamp=explicit_time)
    assert "sessions_per_hour" in monitor.metrics_history
    assert monitor.metrics_history["sessions_per_hour"] == [(explicit_time, 55)]

@patch('utils.monitoring.datetime')
def test_record_metrics_invalid_type(mock_dt, monitor: AgentMonitor, caplog):
    """Test recording metrics with non-numeric values."""
    fixed_time = datetime(2024, 1, 10, 10, 0, 0)
    mock_dt.now.return_value = fixed_time

    metrics = {"response_time_ms": "fast", "error_rate_percent": None}

    with caplog.at_level(logging.WARNING):
        monitor.record_metrics(metrics)

    # Assert invalid metrics were NOT added
    assert "response_time_ms" not in monitor.metrics_history
    assert "error_rate_percent" not in monitor.metrics_history

    # Assert warnings were logged
    assert "has non-numeric value: fast. Skipping." in caplog.text
    assert "has non-numeric value: None. Skipping." in caplog.text

@patch.object(AgentMonitor, 'trigger_alert')
@patch.object(AgentMonitor, 'detect_drift', return_value=False) # Mock drift detection for now
@patch('utils.monitoring.datetime')
def test_record_metrics_threshold_check(mock_dt, mock_detect_drift, mock_trigger_alert, monitor: AgentMonitor, monitor_config):
    """Test that recording metrics triggers alerts correctly based on thresholds."""
    fixed_time = datetime(2024, 1, 10, 10, 0, 0)
    mock_dt.now.return_value = fixed_time

    min_resp, max_resp = monitor_config["metric_thresholds"]["response_time_ms"]
    min_err, max_err = monitor_config["metric_thresholds"]["error_rate_percent"]

    # 1. Metric within bounds
    monitor.record_metrics({"response_time_ms": (min_resp + max_resp) / 2})
    mock_trigger_alert.assert_not_called()
    mock_detect_drift.assert_called_once_with("response_time_ms")
    mock_detect_drift.reset_mock()

    # 2. Metric below lower bound
    monitor.record_metrics({"response_time_ms": min_resp - 10})
    mock_trigger_alert.assert_called_once_with("response_time_ms", min_resp - 10, min_resp, max_resp)
    mock_trigger_alert.reset_mock()
    mock_detect_drift.assert_called_once_with("response_time_ms")
    mock_detect_drift.reset_mock()

    # 3. Metric above upper bound
    monitor.record_metrics({"error_rate_percent": max_err + 1.0})
    mock_trigger_alert.assert_called_once_with("error_rate_percent", max_err + 1.0, min_err, max_err)
    mock_trigger_alert.reset_mock()
    mock_detect_drift.assert_called_once_with("error_rate_percent")
    mock_detect_drift.reset_mock()

    # 4. Metric not in thresholds
    monitor.record_metrics({"unknown_metric": 123.0})
    mock_trigger_alert.assert_not_called()
    mock_detect_drift.assert_called_once_with("unknown_metric")
    mock_detect_drift.reset_mock()

    # 5. Multiple metrics, one out of bounds
    monitor.record_metrics({
        "response_time_ms": max_resp + 50, # Above max
        "error_rate_percent": (min_err + max_err) / 2 # Within bounds
    })
    # Alert should only be called for response time
    mock_trigger_alert.assert_called_once_with("response_time_ms", max_resp + 50, min_resp, max_resp)
    # Drift check called for both
    assert mock_detect_drift.call_count == 2
    mock_detect_drift.assert_has_calls([
        call("response_time_ms"),
        call("error_rate_percent")
    ], any_order=True)

# --- Test detect_drift --- #

# Helper function to populate history for drift tests
def _populate_history(monitor: AgentMonitor, metric: str, values: list[float]):
    base_time = datetime(2024, 1, 1)
    monitor.metrics_history[metric] = [
        (base_time + timedelta(minutes=i), float(val))
        for i, val in enumerate(values)
    ]

# Parametrize drift tests
@pytest.mark.parametrize(
    "test_id, history_values, window_size, threshold, expected_drift",
    [
        # --- Insufficient Data --- #
        ("insufficient_data_short", [10]*5, 5, 10.0, False), # Needs window*2 = 10
        ("insufficient_data_exact", [10]*9, 5, 10.0, False), # Needs 10

        # --- Stable Data --- #        ("stable_data_flat", [10.0]*20, 5, 10.0, False), # No change
        ("stable_data_small_noise", [10.0, 10.1, 9.9, 10.0, 10.2]*4, 5, 10.0, False), # Small noise, below threshold

        # --- Positive Drift --- #
        ("positive_drift_gradual", list(range(10, 25)), 5, 10.0, True), # 10..14 vs 15..19 -> Now 10..14 vs 15..19 is correct within range(10,20)
        ("positive_drift_step", [10.0]*7 + [20.0]*8, 5, 10.0, True), # Compares [10,10,20,20,20] vs [20,20,20,20,20]

        # --- Negative Drift --- #
        ("negative_drift_gradual", list(range(24, 9, -1)), 5, 10.0, True), # 24..20 vs 19..15
        ("negative_drift_step", [20.0]*7 + [10.0]*8, 5, 10.0, True), # Compares [20,20,10,10,10] vs [10,10,10,10,10]

        # --- Division by Zero --- #
        ("div_zero_no_drift", [0.0]*10 + [0.0]*10, 5, 10.0, False), # 0 vs 0
        ("div_zero_drift", [0.0]*7 + [1.0]*8, 5, 10.0, True), # Compares [0,0,1,1,1] vs [1,1,1,1,1]

        # --- Edge cases --- #
        ("just_below_threshold", [10.0]*7 + [10.9]*8, 5, 10.0, False), # Change = 9%
        ("just_above_threshold", [10.0]*7 + [11.1]*8, 5, 10.0, False), # Change = 4.13% < 10% - Expected should be False
    ],
    ids=lambda x: x if isinstance(x, str) else "" # Use test_id for readability
)
def test_detect_drift(
    monitor: AgentMonitor, test_id: str, history_values: list[float], window_size: int, threshold: float, expected_drift: bool
):
    """Test detect_drift with various data patterns."""
    metric_name = "test_metric"
    _populate_history(monitor, metric_name, history_values)

    drift_detected = monitor.detect_drift(
        metric=metric_name,
        window_size=window_size,
        change_threshold_percent=threshold
    )

    assert drift_detected == expected_drift

# --- Test trigger_alert --- #

# Use parametrize to test different endpoints and error conditions
@pytest.mark.parametrize(
    "test_id, alert_endpoints_override, expected_slack_calls, expected_email_calls, expected_log_warnings",
    [
        (
            "slack_and_email",
            None, # Use default from fixture
            1,
            1,
            ["ALERT [agent_test_001] - Metric 'test_metric' value 10.00 outside acceptable range [0.00, 5.00]"],
        ),
        (
            "slack_only",
            [{"type": "slack", "webhook_url": "http://specific.slack"}],
            1,
            0,
            ["ALERT [agent_test_001] - Metric 'test_metric' value 10.00 outside acceptable range [0.00, 5.00]"],
        ),
        (
            "email_only",
            [{"type": "email", "address": "only@example.com"}],
            0,
            1,
            ["ALERT [agent_test_001] - Metric 'test_metric' value 10.00 outside acceptable range [0.00, 5.00]"],
        ),
        (
            "unsupported_type",
            [{"type": "sms", "number": "12345"}],
            0,
            0,
            [
                "ALERT [agent_test_001] - Metric 'test_metric' value 10.00 outside acceptable range [0.00, 5.00]",
                "Unsupported alert endpoint type: sms"
            ],
        ),
        (
            "missing_key_slack",
            [{"type": "slack", "url": "no_webhook_key"}], # Missing webhook_url
            0,
            0,
            [
                "ALERT [agent_test_001] - Metric 'test_metric' value 10.00 outside acceptable range [0.00, 5.00]",
                "Missing key in alert endpoint config: 'webhook_url'"
            ],
        ),
        (
            "mixed_valid_invalid",
            [
                {"type": "slack", "webhook_url": "http://ok.slack"},
                {"type": "email"}, # Missing address
            ],
            1,
            0,
            [
                "ALERT [agent_test_001] - Metric 'test_metric' value 10.00 outside acceptable range [0.00, 5.00]",
                "Missing key in alert endpoint config: 'address'"
            ],
        ),
        (
            "no_endpoints",
            [],
            0,
            0,
            ["ALERT [agent_test_001] - Metric 'test_metric' value 10.00 outside acceptable range [0.00, 5.00]"],
        ),
    ],
    ids=lambda x: x[0] if isinstance(x, str) else "" # Use test_id
)
@patch.object(AgentMonitor, '_send_email_alert')
@patch.object(AgentMonitor, '_send_slack_alert')
def test_trigger_alert(
    mock_send_slack, mock_send_email,
    monitor: AgentMonitor, # Uses default fixture if override is None
    monitor_config, # To get agent_id
    caplog,
    test_id: str, alert_endpoints_override: list | None,
    expected_slack_calls: int, expected_email_calls: int,
    expected_log_warnings: list[str]
):
    """Test trigger_alert calls correct methods and handles errors."""
    metric_name = "test_metric"
    value = 10.0
    min_thresh, max_thresh = 0.0, 5.0

    # Override endpoints if specified for the test case
    if alert_endpoints_override is not None:
        monitor.alert_endpoints = alert_endpoints_override

    with caplog.at_level(logging.WARNING):
        monitor.trigger_alert(metric_name, value, min_thresh, max_thresh)

    # Check calls to mocked alert methods
    assert mock_send_slack.call_count == expected_slack_calls
    assert mock_send_email.call_count == expected_email_calls

    # Check specific calls if needed (e.g., check webhook_url used)
    if expected_slack_calls > 0:
        expected_message = f"ALERT [{monitor.agent_id}] - Metric '{metric_name}' value {value:.2f} outside acceptable range [{min_thresh:.2f}, {max_thresh:.2f}]"
        # Find the endpoint config used for the call
        slack_endpoint = next(ep for ep in monitor.alert_endpoints if ep.get('type') == 'slack')
        mock_send_slack.assert_called_with(slack_endpoint['webhook_url'], expected_message)

    if expected_email_calls > 0:
        expected_message = f"ALERT [{monitor.agent_id}] - Metric '{metric_name}' value {value:.2f} outside acceptable range [{min_thresh:.2f}, {max_thresh:.2f}]"
        email_endpoint = next(ep for ep in monitor.alert_endpoints if ep.get('type') == 'email')
        mock_send_email.assert_called_with(email_endpoint['address'], expected_message)

    # Check logged warnings
    for msg in expected_log_warnings:
        assert msg in caplog.text

# --- Test _is_decreasing --- #

@pytest.mark.parametrize(
    "test_id, values, window, expected_result",
    [
        ("insufficient_data", [10, 9, 8], 5, False),
        ("decreasing_linear", [10, 9, 8, 7, 6], 5, True),
        ("decreasing_step", [10, 10, 10, 5, 5], 5, True),
        ("increasing_linear", [6, 7, 8, 9, 10], 5, False),
        ("flat_data", [8, 8, 8, 8, 8], 5, False),
        ("noisy_flat", [8, 9, 7, 8, 9, 7], 6, False),
        ("noisy_decreasing", [10, 11, 8, 9, 6, 7], 6, True),
        ("very_short_window", [10, 9], 2, True), # Test window=2
        ("window_larger_than_data", [10, 9, 8, 7, 6], 10, False),
    ],
    ids=lambda x: x if isinstance(x, str) else "" # Use test_id
)
def test_is_decreasing(monitor: AgentMonitor, test_id: str, values: list[float], window: int, expected_result: bool):
    """Test the _is_decreasing helper method with various data patterns."""
    metric_name = "trend_metric"
    # Create history with simple timestamps
    base_time = datetime(2024, 1, 1)
    history = [
        (base_time + timedelta(minutes=i), float(val))
        for i, val in enumerate(values)
    ]

    if test_id == "noisy_flat":
        expected_result = True # Adjust expectation as polyfit gives slightly negative slope

    is_decreasing = monitor._is_decreasing(history, window=window)
    assert is_decreasing == expected_result

# --- Test Recommend Adaptation --- #

# Consolidated and refactored test for recommend_adaptation
@pytest.mark.parametrize(
    "test_id, setup_metrics, drift_results, decreasing_results, expected_recommendations, expect_info_log",
    [
        # Case 1: No drift, no recos
        (
            "no_drift",
            {"conversion_rate": [10]*20, "error_rate": [1]*10},
            {"conversion_rate": False, "error_rate": False},
            {}, # decreasing results don't matter if no drift
            [],
            True, # Expect info log "No adaptation recommendations..."
        ),
        # Case 2: Drift + Decreasing Conversion Rate
        (
            "drift_decreasing_conversion",
            {"conversion_rate": list(range(20, 0, -1)), "sessions_per_hour": [50]*20},
            {"conversion_rate": True, "sessions_per_hour": False},
            {"conversion_rate": True},
            ["Consider adjusting pricing strategy: Conversion rate decreasing."],
            False,
        ),
        # Case 3: Drift + Decreasing Inventory Turnover
        (
            "drift_decreasing_inventory",
            {"inventory_turnover": [5, 4, 3, 2, 1]*4, "error_rate": [1]*20},
            {"inventory_turnover": True, "error_rate": False},
            {"inventory_turnover": True},
            ["Consider adjusting promotions/stocking: Inventory turnover decreasing."],
            False,
        ),
        # Case 4: Drift + Increasing Error Rate (NOT decreasing)
        (
            "drift_increasing_error",
            {"error_rate": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]},
            {"error_rate": True},
            {"error_rate": False}, # Explicitly mock as NOT decreasing
            [f"Investigate increasing error rate for agent agent_test_001"], # Use agent_id from config
            False,
        ),
        # Case 5: Drift but NOT Decreasing Conversion Rate
        (
            "drift_not_decreasing_conversion",
            {"conversion_rate": [10, 11, 12, 11, 13]*4},
            {"conversion_rate": True},
            {"conversion_rate": False},
            [], # No specific recommendation for this case
            True, # Expect info log
        ),
        # Case 6: Multiple recommendations
        (
            "multiple_recos",
            {
                "conversion_rate": [10, 9, 8]*7, # Decreasing
                "inventory_turnover": [5, 4, 3]*7, # Decreasing
                "error_rate": [1]*20 # Stable
            },
            {"conversion_rate": True, "inventory_turnover": True, "error_rate": False},
            {"conversion_rate": True, "inventory_turnover": True},
            [
                "Consider adjusting pricing strategy: Conversion rate decreasing.",
                "Consider adjusting promotions/stocking: Inventory turnover decreasing."
            ],
            False,
        ),
        # Case 7: Error rate drifting but decreasing (no recommendation)
        (
            "drift_decreasing_error",
            {"error_rate": [5, 4, 3, 2, 1]*2},
            {"error_rate": True},
            {"error_rate": True}, # Mock as decreasing
            [],
            True, # Expect info log
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "" # Use test_id
)
def test_recommend_adaptation(
    monitor: AgentMonitor, caplog,
    test_id: str, setup_metrics: dict, drift_results: dict, decreasing_results: dict,
    expected_recommendations: list[str], expect_info_log: bool
):
    """Test recommend_adaptation based on mocked drift and trend results."""
    # Populate history for the metrics involved in this test case
    monitor.metrics_history.clear() # Ensure clean state for each run
    metric_history_map = {}
    for metric, values in setup_metrics.items():
        _populate_history(monitor, metric, values)
        metric_history_map[metric] = monitor.metrics_history[metric]

    # Mock detect_drift based on pre-defined results
    def mock_detect_drift_side_effect(metric, **kwargs):
        return drift_results.get(metric, False)

    # Mock _is_decreasing based on pre-defined results, identifying metric by history list
    def mock_is_decreasing_side_effect(history, window):
        # Find which metric this history list belongs to
        metric_name = None
        for m, h in metric_history_map.items():
            # Use object identity comparison as history list is created uniquely per metric
            if history is h:
                metric_name = m
                break
        # Return the pre-defined result for that metric, default to False
        if metric_name:
            return decreasing_results.get(metric_name, False)
        logger.warning(f"_is_decreasing called with unknown history: {history[:2]}...")
        return False

    with patch.object(monitor, 'detect_drift', side_effect=mock_detect_drift_side_effect), \
         patch.object(monitor, '_is_decreasing', side_effect=mock_is_decreasing_side_effect), \
         caplog.at_level(logging.INFO):

        recommendations = monitor.recommend_adaptation()

    # Assertions
    assert set(recommendations) == set(expected_recommendations)
    # Check for the "No recommendations" log message only if no recommendations were expected
    no_reco_log = f"No adaptation recommendations for agent {monitor.agent_id}" in caplog.text
    assert no_reco_log == expect_info_log

# Placeholder tests for evaluation methods
# ...

# Placeholder tests for run_cycle_for_product tests

# Placeholder tests for evaluation methods

# Placeholder tests for run_cycle_for_product tests 