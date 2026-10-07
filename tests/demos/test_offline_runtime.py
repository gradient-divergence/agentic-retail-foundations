import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "name,dependency,expected",
    [
        ("agentops_latency_middleware", "prometheus_client", "monitoring"),
        ("api_gateway_demo", "jose", "auth"),
        ("api_gateway_demo", "redis", "streaming"),
        ("api_gateway_book_demo", "redis", "streaming"),
        ("inventory_api_demo", "redis", "streaming"),
        ("inventory_event_api_demo", "redis", "streaming"),
        ("inventory_crdt_demo", "redis", "streaming"),
        ("state_manager_demo", "redis", "streaming"),
        ("dynamic_pricing_feedback_book_demo", "redis", "streaming"),
        ("pricing_explainability_shap_demo", "shap", "explainability"),
        ("supply_chain_gnn_delay_demo", "torch_geometric", "gnn"),
        ("dynamic_pricing_feedback_book_demo", None, "Redis"),
        ("dynamic_pricing_feedback_demo", None, "Redis"),
        ("spark_streaming_demo", None, "Kafka"),
        ("stream_processing_sales_velocity_demo", None, "Kafka"),
    ],
)
def test_missing_extra_or_service_has_one_line_exit(name, dependency, expected):
    code = f"""
import runpy, socket, sys
def unavailable(*args, **kwargs):
    raise OSError("service unavailable")
socket.create_connection = unavailable
if {dependency!r}:
    sys.modules[{dependency!r}] = None
runpy.run_module('demos.{name}', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=8)
    assert result.returncode != 0
    assert result.stdout == ""
    assert expected in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1
    assert "Traceback" not in result.stderr
