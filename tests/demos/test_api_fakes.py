import sys

import demos
from tests.demos.api_fakes import load_demo


def test_fake_import_restores_module_cache_and_package_attribute(monkeypatch):
    monkeypatch.delitem(sys.modules, "demos.hitl_approval_api_demo", raising=False)
    monkeypatch.delattr(demos, "hitl_approval_api_demo", raising=False)
    with monkeypatch.context() as isolated:
        module = load_demo(isolated, "hitl_approval_api_demo")
        assert demos.hitl_approval_api_demo is module
    assert "demos.hitl_approval_api_demo" not in sys.modules
    assert not hasattr(demos, "hitl_approval_api_demo")
