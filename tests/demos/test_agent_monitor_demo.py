from unittest.mock import patch

from demos.agent_monitor_demo import AgentMonitor


def test_drift_with_zero_baseline_is_defined():
    monitor = AgentMonitor("agent", {}, [])
    for index in range(60):
        monitor.record_metrics(index, {"metric": 0 if index < 30 else 1})
    assert monitor.detect_drift("metric")
    for index in range(30):
        monitor.record_metrics(index + 60, {"metric": 0})
    assert monitor.detect_drift("metric")
    for index in range(30):
        monitor.record_metrics(index + 90, {"metric": 0})
    assert not monitor.detect_drift("metric")


def test_configured_alert_endpoints_work_offline():
    monitor = AgentMonitor(
        "agent",
        {"metric": (0, 1)},
        [
            {"type": "slack", "webhook_url": "offline"},
            {"type": "email", "address": "offline@example.invalid"},
        ],
    )
    with patch("utils.monitoring.logger.info") as log:
        monitor.record_metrics(0, {"metric": 2})
    assert log.call_count == 2
