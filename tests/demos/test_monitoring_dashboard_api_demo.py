import sys
from types import SimpleNamespace

import pytest

from tests.demos.api_fakes import HTTPException, load_demo


def test_import_does_not_connect_and_request_without_config_is_clear(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "psycopg2", SimpleNamespace(connect=lambda *args: calls.append(args)))
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)
    demo = load_demo(monkeypatch, "monitoring_dashboard_api_demo")
    assert not calls
    with pytest.raises(HTTPException) as exc:
        demo.get_agent_metrics()
    assert exc.value.status_code == 503
    assert "SUPABASE_DB_URL" in exc.value.detail
    assert not calls
