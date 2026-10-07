import sys
from types import SimpleNamespace

import pytest

from tests.demos.api_fakes import HTTPException, load_demo


def test_import_without_supabase_config_is_offline(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules, "supabase", SimpleNamespace(create_client=lambda *args: calls.append(args))
    )
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    demo = load_demo(monkeypatch, "monitoring_dashboard_supabase_demo")
    assert not calls
    with pytest.raises(HTTPException) as exc:
        demo.get_agent_metrics()
    assert exc.value.status_code == 503


def test_successful_response_needs_only_data(monkeypatch):
    table = SimpleNamespace(
        select=lambda query: SimpleNamespace(execute=lambda: SimpleNamespace(data=[{"agent_id": "demo"}]))
    )
    monkeypatch.setenv("SUPABASE_URL", "https://example.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "supabase",
        SimpleNamespace(create_client=lambda *args: SimpleNamespace(table=lambda name: table)),
    )
    demo = load_demo(monkeypatch, "monitoring_dashboard_supabase_demo")
    assert demo.get_agent_metrics() == {"agents": [{"agent_id": "demo"}]}


def test_provider_error_does_not_expose_credentials(monkeypatch):
    def fail(*args):
        raise RuntimeError("test-key https://example.invalid")

    monkeypatch.setenv("SUPABASE_URL", "https://example.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "supabase", SimpleNamespace(create_client=fail))
    demo = load_demo(monkeypatch, "monitoring_dashboard_supabase_demo")
    with pytest.raises(HTTPException) as exc:
        demo.get_agent_metrics()
    assert "test-key" not in exc.value.detail
