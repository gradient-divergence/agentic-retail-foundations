from __future__ import annotations

import logging
from math import isfinite

import numpy as np

from utils.monitoring import AgentMonitor as MonitorUtilities


# region book:agent-monitor-class
class AgentMonitor:
    def __init__(self, agent_id, metric_thresholds, alert_endpoints):
        self.agent_id = agent_id
        self.metric_thresholds = metric_thresholds
        self.alert_endpoints = alert_endpoints
        self.metrics_history = {}
        self.history_limit = 120

    # endregion book:agent-monitor-class

    # region book:agent-monitor-record-metrics
    def record_metrics(self, timestamp, metrics_dict):
        """Record a set of performance metrics at a specific time"""
        if any(not isfinite(value) for value in metrics_dict.values()):
            logging.getLogger(__name__).warning(
                "monitor.measurement_rejected", extra={"agent_id": self.agent_id, "reason": "nonfinite"}
            )
            raise ValueError("Metric measurements must be finite")
        for metric, value in metrics_dict.items():
            if metric not in self.metrics_history:
                self.metrics_history[metric] = []
            self.metrics_history[metric].append((timestamp, value))
            self.metrics_history[metric] = self.metrics_history[metric][-self.history_limit :]

            if metric in self.metric_thresholds:
                min_val, max_val = self.metric_thresholds[metric]
                if value < min_val or value > max_val:
                    self.trigger_alert(metric, value, min_val, max_val)

    # endregion book:agent-monitor-record-metrics

    # region book:agent-monitor-detect-drift
    def detect_drift(self, metric, window_size=30):
        """Illustrative 15% mean-shift heuristic; assumes equally spaced observations"""
        if len(self.metrics_history.get(metric, [])) < window_size * 2:
            return False

        recent = [v for _, v in self.metrics_history[metric][-window_size:]]
        previous = [v for _, v in self.metrics_history[metric][-window_size * 2 : -window_size]]
        recent_avg = sum(recent) / len(recent)
        previous_avg = sum(previous) / len(previous)
        if previous_avg == 0:
            return recent_avg != 0
        percent_change = abs((recent_avg - previous_avg) / previous_avg) * 100
        return percent_change > 15

    # endregion book:agent-monitor-detect-drift

    # region book:agent-monitor-trigger-alert
    def trigger_alert(self, metric, value, min_threshold, max_threshold):
        """Send alerts when metrics exceed thresholds"""
        message = (
            f"ALERT: Agent {self.agent_id} - {metric} value {value} "
            f"outside acceptable range [{min_threshold}, {max_threshold}]"
        )

        for endpoint in self.alert_endpoints:
            if endpoint["type"] == "slack":
                self._send_slack_alert(endpoint["webhook_url"], message)
            elif endpoint["type"] == "email":
                self._send_email_alert(endpoint["address"], message)

    # endregion book:agent-monitor-trigger-alert

    _send_slack_alert = MonitorUtilities._send_slack_alert
    _send_email_alert = MonitorUtilities._send_email_alert

    # region book:agent-monitor-recommend-adaptation
    def recommend_adaptation(self):
        """Based on metrics, recommend agent adaptation strategies"""
        recommendations = []

        for metric, history in self.metrics_history.items():
            if self.detect_drift(metric):
                if metric == "conversion_rate" and self._is_decreasing(history, 10):
                    recommendations.append(
                        "Investigate conversion against plan, availability, and seasonality"
                    )
                elif metric == "inventory_turnover" and self._is_decreasing(history, 10):
                    recommendations.append(
                        "Investigate turnover receipts, demand, and clearance contribution"
                    )

        return recommendations

    # endregion book:agent-monitor-recommend-adaptation

    # region book:agent-monitor-is-decreasing
    def _is_decreasing(self, history, window=10):
        """Check if metric shows a decreasing trend"""
        if len(history) < window:
            return False

        recent = [v for _, v in history[-window:]]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        return slope < 0


# endregion book:agent-monitor-is-decreasing
