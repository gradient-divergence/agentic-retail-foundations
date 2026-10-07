import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from demos import virtual_shopping_assistant_demo as demo


class FakeClient:
    def __init__(self, name="recommend_outfit", arguments='{"style": "summer"}'):
        self.inputs = []
        self.call = SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id="call-1")
        self.responses = self

    def create(self, **kwargs):
        self.inputs.append(list(kwargs["input"]))
        if len(self.inputs) == 1:
            return SimpleNamespace(output=[self.call], output_text="")
        return SimpleNamespace(output=[], output_text="A summer outfit")


def test_fake_client_runs_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = FakeClient()
    assert demo.run_assistant_demo(client=client) == "A summer outfit"
    assert json.loads(client.inputs[1][-1]["output"])[0] == "Red sundress with floral prints"


@pytest.mark.parametrize("arguments", ['{"style": 42}', '{"style": ""}', '["summer"]', "{bad"])
def test_invalid_arguments_do_not_execute_tool(monkeypatch, arguments):
    executions = []
    monkeypatch.setattr(demo, "recommend_outfit", lambda **kwargs: executions.append(kwargs))
    client = FakeClient(arguments=arguments)
    demo.run_assistant_demo(client=client)
    assert executions == []
    assert "error" in json.loads(client.inputs[1][-1]["output"])


def test_unknown_tool_returns_error():
    client = FakeClient(name="delete_order")
    demo.run_assistant_demo(client=client)
    assert json.loads(client.inputs[1][-1]["output"]) == {"error": "Unknown function"}


def test_no_key_script_exits_cleanly():
    env = {**os.environ, "OPENAI_API_KEY": ""}
    result = subprocess.run(
        [sys.executable, "-m", "demos.virtual_shopping_assistant_demo"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OPENAI_API_KEY" in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_provider_error_keeps_existing_message(monkeypatch):
    class ProviderError(Exception):
        pass

    def fail_response(**kwargs):
        raise ProviderError("service unavailable")

    client = SimpleNamespace(responses=SimpleNamespace(create=fail_response))
    sdk = SimpleNamespace(OpenAI=lambda **kwargs: client, OpenAIError=ProviderError)
    monkeypatch.setitem(sys.modules, "openai", sdk)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    assert demo.run_assistant_demo() == (
        "Sorry, there was an error communicating with the AI service: service unavailable"
    )
