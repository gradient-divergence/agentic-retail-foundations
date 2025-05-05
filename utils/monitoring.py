"""
Utilities for monitoring agent performance and triggering alerts.
"""

import logging
from datetime import datetime
import numpy as np  # For drift detection
from collections import defaultdict
import pytest # Import locally if needed, better at top

# Define a logger for this module
logger = logging.getLogger(__name__)


class AgentMonitor:
    """Monitors agent performance metrics and detects issues."""

    def __init__(
        self,
        agent_id: str,
        metric_thresholds: dict[str, tuple[float, float]],
        alert_endpoints: list[dict[str, str]],
    ):
        """
        Args:
            agent_id: ID of the agent being monitored.
            metric_thresholds: Dict mapping metric names to (min_value, max_value) tuples.
            alert_endpoints: List of dicts specifying alert channels (e.g., {"type": "slack", "webhook_url": "..."}).
        """
        self.agent_id = agent_id
        self.metric_thresholds = metric_thresholds
        self.alert_endpoints = alert_endpoints
        # Stores metric_name -> List[(timestamp, value)]
        self.metrics_history: dict[str, list[tuple[datetime, float]]] = defaultdict(
            list
        )
        logger.info(f"Initialized monitor for agent {agent_id}")

    def record_metrics(
        self, metrics_dict: dict[str, float], timestamp: datetime | None = None
    ):
        """Record a set of performance metrics at a specific time."""
        ts = timestamp or datetime.now()
        for metric, value in metrics_dict.items():
            if not isinstance(value, int | float):
                logger.warning(
                    f"Metric '{metric}' for agent {self.agent_id} has non-numeric value: {value}. Skipping."
                )
                continue

            self.metrics_history[metric].append((ts, value))
            logger.debug(f"Recorded metric for {self.agent_id}: {metric}={value}")

            # Check if metric exceeds thresholds
            if metric in self.metric_thresholds:
                min_val, max_val = self.metric_thresholds[metric]
                if not (min_val <= value <= max_val):
                    self.trigger_alert(metric, value, min_val, max_val)

            # Check for drift after adding new data
            if self.detect_drift(metric):
                logger.warning(
                    f"Drift detected for metric '{metric}' in agent {self.agent_id}"
                )
                # Optionally trigger a different type of alert for drift

    def detect_drift(
        self, metric: str, window_size: int = 30, change_threshold_percent: float = 15.0
    ) -> bool:
        """Detect if a metric is drifting significantly from historical patterns."""
        history = self.metrics_history.get(metric, [])
        if len(history) < window_size * 2:
            return False  # Not enough history

        try:
            recent_values = [v for _, v in history[-window_size:]]
            previous_values = [v for _, v in history[-window_size * 2 : -window_size]]

            if not recent_values or not previous_values:
                return False  # Should not happen if len check passed, but safety first

            recent_avg = np.mean(recent_values)
            previous_avg = np.mean(previous_values)

            if previous_avg == 0:  # Avoid division by zero
                drift = bool(recent_avg != 0)
                return drift

            percent_change = abs((recent_avg - previous_avg) / previous_avg) * 100
            drift = bool(percent_change > change_threshold_percent)
            return drift
        except Exception as e:
            # Ensure the error log includes the exception type and message
            logger.error(f"Error during drift detection for {metric}: {type(e).__name__}: {e}")
            return False

    def trigger_alert(
        self, metric: str, value: float, min_threshold: float, max_threshold: float
    ):
        """Send alerts when metrics exceed thresholds."""
        message = f"ALERT [{self.agent_id}] - Metric '{metric}' value {value:.2f} outside acceptable range [{min_threshold:.2f}, {max_threshold:.2f}]"
        logger.warning(message)  # Log the alert

        for endpoint in self.alert_endpoints:
            try:
                if endpoint.get("type") == "slack":
                    self._send_slack_alert(endpoint["webhook_url"], message)
                elif endpoint.get("type") == "email":
                    self._send_email_alert(endpoint["address"], message)
                # Add other notification types here
                else:
                    logger.warning(
                        f"Unsupported alert endpoint type: {endpoint.get('type')}"
                    )
            except KeyError as e:
                logger.error(
                    f"Missing key in alert endpoint config: {e}. Endpoint: {endpoint}"
                )
            except Exception as e:
                logger.error(f"Failed to send alert to {endpoint}: {e}")

    def recommend_adaptation(self) -> list[str]:
        """Based on metrics, recommend agent adaptation strategies (Placeholder)."""
        recommendations = []
        for metric, history in self.metrics_history.items():
            if self.detect_drift(metric):
                if metric == "conversion_rate" and self._is_decreasing(history, 10):
                    recommendations.append(
                        "Consider adjusting pricing strategy: Conversion rate decreasing."
                    )
                elif metric == "inventory_turnover" and self._is_decreasing(
                    history, 10
                ):
                    recommendations.append(
                        "Consider adjusting promotions/stocking: Inventory turnover decreasing."
                    )
                elif metric == "error_rate" and not self._is_decreasing(history, 5):
                    recommendations.append(
                        f"Investigate increasing error rate for agent {self.agent_id}"
                    )

        if not recommendations:
            logger.info(
                f"No adaptation recommendations for agent {self.agent_id} based on current drift detection."
            )

        return recommendations

    def _is_decreasing(
        self, history: list[tuple[datetime, float]], window: int = 10
    ) -> bool:
        """Check if metric shows a decreasing trend using linear regression slope."""
        if len(history) < window:
            return False
        try:
            recent_values = [v for _, v in history[-window:]]
            # Use indices as x-values for trend calculation
            indices = np.arange(len(recent_values))
            slope = np.polyfit(indices, recent_values, 1)[0]
            return bool(slope < 0)  # Negative slope indicates decreasing trend
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return False

    # --- Placeholder Alert Methods ---
    def _send_slack_alert(self, webhook_url: str, message: str):
        """Placeholder for sending a Slack alert."""
        # In a real implementation: use requests library to post to webhook_url
        logger.info(f"[SIMULATED SLACK ALERT] to {webhook_url}: {message}")
        pass

    def _send_email_alert(self, address: str, message: str):
        """Placeholder for sending an email alert."""
        # In a real implementation: use smtplib or email library
        logger.info(f"[SIMULATED EMAIL ALERT] to {address}: {message}")
        pass
