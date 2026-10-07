import asyncio
import sys
from types import SimpleNamespace

import pytest

from tests.demos.api_fakes import load_demo


@pytest.fixture
def demo(monkeypatch):
    observations = []
    histogram = SimpleNamespace(labels=lambda endpoint: SimpleNamespace(observe=observations.append))
    monkeypatch.setitem(
        sys.modules,
        "prometheus_client",
        SimpleNamespace(
            Histogram=lambda *args: histogram,
            generate_latest=lambda: b"agent_api_latency_seconds 1\n",
            CONTENT_TYPE_LATEST="text/plain; version=0.0.4; charset=utf-8",
        ),
    )
    module = load_demo(monkeypatch, "agentops_latency_middleware")
    return module, observations


def test_metrics_are_raw_prometheus_text(demo):
    module, _ = demo
    response = asyncio.run(module.metrics())
    assert response.body == b"agent_api_latency_seconds 1\n"
    assert response.media_type.startswith("text/plain")


def test_failed_requests_are_timed_with_monotonic_clock(demo, monkeypatch):
    module, observations = demo
    clock = iter([10.0, 10.25])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock))

    async def failing(request):
        raise ValueError("handler failed")

    with pytest.raises(ValueError, match="handler failed"):
        asyncio.run(module.monitor(SimpleNamespace(url=SimpleNamespace(path="/demo")), failing))
    assert observations == [0.25]
