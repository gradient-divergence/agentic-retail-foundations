import importlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

PROVIDER_DEMOS = [
    "inventory_agents_sdk_demo",
    "dynamic_pricing_agents_sdk_demo",
    "virtual_shopping_responses_demo",
    "openai_agents_handoff_guardrails_trace_eval_demo",
    "google_adk_quickstart_demo",
    "a2a_agent_card_demo",
    "ap2_payment_request_demo",
]


@pytest.mark.parametrize("name", PROVIDER_DEMOS)
def test_import_is_offline_without_optional_sdk(name):
    result = subprocess.run(
        [sys.executable, "-c", f"import demos.{name}"], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("name", PROVIDER_DEMOS[:4])
def test_openai_missing_key_is_clear(monkeypatch, name):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    module = importlib.import_module(f"demos.{name}")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        module.run_demo()


def test_sdk_loader_rejects_cached_local_agents(monkeypatch):
    import agents
    from demos.openai_agents_sdk_import import import_openai_agents_sdk

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    original = sys.modules["agents"]
    with pytest.raises(ImportError, match="local.*agents|fresh Python"):
        import_openai_agents_sdk()
    assert sys.modules["agents"] is original is agents


def test_missing_sdks_have_clear_errors(monkeypatch):
    import demos.openai_agents_sdk_import as loader

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delitem(sys.modules, "agents", raising=False)

    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(loader.importlib, "import_module", missing)
    original_path = list(sys.path)
    with pytest.raises(ModuleNotFoundError, match="OpenAI Agents SDK is not installed"):
        loader.import_openai_agents_sdk()
    assert sys.path == original_path


def test_provider_entrypoints_fail_offline_without_using_network(monkeypatch):
    import demos.a2a_agent_card_demo as a2a
    import demos.ap2_payment_request_demo as ap2
    import demos.google_adk_quickstart_demo as google

    monkeypatch.setitem(sys.modules, "a2a", None)
    monkeypatch.setitem(sys.modules, "ap2", None)
    with pytest.raises(ModuleNotFoundError, match="a2a-sdk"):
        a2a.build_agent_card()
    with pytest.raises(ModuleNotFoundError, match="ap2 types"):
        ap2.build_payment_request()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        google.main()


def test_payment_request_is_only_a_test_sketch():
    from pathlib import Path

    source = Path("demos/ap2_payment_request_demo.py").read_text()
    assert "test-mode" in source
    assert "submit" not in source and "requests." not in source


def test_responses_prints_reply_without_tool_calls(capsys):
    from demos.virtual_shopping_responses_demo import run_demo

    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output=[], output_text="Try a linen outfit.")
        )
    )
    run_demo(client=client)
    assert capsys.readouterr().out == "Try a linen outfit.\n"


def test_responses_roundtrip_keeps_tool_call_id(capsys):
    from demos.virtual_shopping_responses_demo import run_demo

    calls = []

    def create(**kwargs):
        calls.append(list(kwargs["input"]))
        if len(calls) == 1:
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="recommend_outfit",
                        call_id="call-1",
                        arguments='{"style":"summer"}',
                    )
                ],
                output_text="",
            )
        return SimpleNamespace(output=[], output_text="Summer outfit")

    run_demo(client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    assert calls[1][-1]["call_id"] == "call-1"
    assert "sundress" in calls[1][-1]["output"]
    assert capsys.readouterr().out == "Summer outfit\n"


@pytest.mark.parametrize("name", PROVIDER_DEMOS[:4])
def test_command_reports_missing_key(name):
    env = {key: value for key, value in os.environ.items() if key != "OPENAI_API_KEY"}
    result = subprocess.run(
        [sys.executable, "-m", f"demos.{name}"], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode != 0
    assert "OPENAI_API_KEY" in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("amount", [-1, 0, 1.5])
def test_inventory_tool_rejects_invalid_orders(amount):
    from demos.inventory_agents_sdk_demo import inventory_db, order_product

    before = dict(inventory_db)
    with pytest.raises(ValueError):
        order_product("product_123", amount)
    assert inventory_db == before


@pytest.mark.parametrize("price", [-1, float("nan"), float("inf")])
def test_pricing_tool_rejects_invalid_prices(price):
    from demos.dynamic_pricing_agents_sdk_demo import current_prices, update_price

    before = dict(current_prices)
    with pytest.raises(ValueError):
        update_price("product_456", price)
    assert current_prices == before
