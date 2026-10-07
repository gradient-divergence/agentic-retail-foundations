import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "inventory_agents_sdk_demo",
        "dynamic_pricing_agents_sdk_demo",
        "openai_agents_handoff_guardrails_trace_eval_demo",
    ],
)
def test_installed_sdk_builds_and_invokes_tools_offline(name):
    code = f"""
import asyncio
import json
import os
import socket
from types import SimpleNamespace

def offline(*args, **kwargs):
    raise AssertionError("No network is allowed")
socket.socket.connect = offline
os.environ["OPENAI_API_KEY"] = "offline-test-key"
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
from demos import {name} as demo
from demos.openai_agents_sdk_import import import_openai_agents_sdk
sdk = import_openai_agents_sdk()
seen = []
def run(agent, task):
    for tool in agent.tools:
        assert isinstance(tool, sdk.FunctionTool)
        schema = tool.params_json_schema
        assert schema["additionalProperties"] is False
        payloads = {{
            "check_inventory": {{"product_id": "product_123"}},
            "order_product": {{"product_id": "product_123", "amount": 2}},
            "get_competitor_price": {{"product_id": "product_456"}},
            "get_inventory": {{"product_id": "product_456"}},
            "update_price": {{"product_id": "product_456", "new_price": 110.0}},
            "propose_price": {{"payload": {{
                "product_id": "SKU123", "current_price": 49.99, "discount_pct": 15,
            }}}},
        }}
        context = sdk.RunContextWrapper(context=None)
        result = asyncio.run(tool.on_invoke_tool(context, json.dumps(payloads[tool.name])))
        assert result is not None
        assert not (isinstance(result, str) and result.startswith("An error occurred"))
        seen.append(tool.name)
    if agent.name == "PlannerAgent":
        output = '{{"product_id":"SKU123","current_price":49.99,"discount_pct":15,"new_price":42.49}}'
    else:
        output = '{{"status":"executed"}}'
    return SimpleNamespace(final_output=output)
sdk.Runner.run_sync = run
demo.run_demo()
assert seen
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
