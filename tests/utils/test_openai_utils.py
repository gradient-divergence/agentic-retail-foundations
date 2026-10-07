import ast
import asyncio
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_sdk(monkeypatch):
    """Keep SDK type checks while replacing only the unavailable provider boundary."""
    sdk = ModuleType("openai")
    sdk.AsyncOpenAI = type("AsyncOpenAI", (), {})
    sdk.OpenAI = type("OpenAI", (), {})
    chat_types = ModuleType("openai.types.chat")
    chat_types.ChatCompletion = type("ChatCompletion", (SimpleNamespace,), {})
    chat_types.ChatCompletionMessageParam = dict
    for name, module in (
        ("openai", sdk),
        ("openai.types", ModuleType("openai.types")),
        ("openai.types.chat", chat_types),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    response = chat_types.ChatCompletion(
        id="chatcmpl-mock",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                index=0,
                message=SimpleNamespace(content="Test response", role="assistant"),
                logprobs=None,
            )
        ],
        created=1677652288,
        model="gpt-mock",
        object="chat.completion",
    )
    create = AsyncMock(return_value=response)
    client = MagicMock(spec=sdk.AsyncOpenAI)
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    from utils.openai_utils import safe_chat_completion

    return SimpleNamespace(
        client=client,
        sync_client=MagicMock(spec=sdk.OpenAI),
        response=response,
        create=create,
        call=safe_chat_completion,
    )


def test_safe_chat_completion_success(fake_sdk, caplog):
    messages = [{"role": "user", "content": "Test message"}]
    with caplog.at_level(logging.DEBUG):
        completion = asyncio.run(
            fake_sdk.call(fake_sdk.client, model="gpt-test", messages=messages, temperature=0.7)
        )
    assert completion.choices[0].message.content == "Test response"
    fake_sdk.create.assert_awaited_once_with(model="gpt-test", messages=messages, temperature=0.7)
    assert "OpenAI completions.create call succeeded" in caplog.text


def test_safe_chat_completion_retry_on_failure(fake_sdk, caplog):
    fake_sdk.create.side_effect = [TimeoutError("API timed out"), fake_sdk.response]
    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep, caplog.at_level(logging.WARNING):
        completion = asyncio.run(
            fake_sdk.call(fake_sdk.client, model="gpt-test", messages=[], retry_backoff=0.1)
        )
    assert completion.choices[0].message.content == "Test response"
    assert fake_sdk.create.await_count == 2
    sleep.assert_awaited_once_with(0.1)
    assert "OpenAI call failed (attempt 1/3): API timed out" in caplog.text


def test_safe_chat_completion_retries_generator_messages(fake_sdk):
    requests = []

    async def create(**kwargs):
        requests.append(list(kwargs["messages"]))
        if len(requests) == 1:
            raise TimeoutError("First request failed")
        return fake_sdk.response

    fake_sdk.create.side_effect = create
    messages = ({"role": "user", "content": "Keep this message"} for _ in range(1))
    asyncio.run(fake_sdk.call(fake_sdk.client, model="gpt-test", messages=messages, retry_backoff=0))
    assert requests == [
        [{"role": "user", "content": "Keep this message"}],
        [{"role": "user", "content": "Keep this message"}],
    ]


def test_safe_chat_completion_failure_after_retries(fake_sdk):
    error = TimeoutError("Persistent API timeout")
    fake_sdk.create.side_effect = error
    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep, pytest.raises(TimeoutError) as exc:
        asyncio.run(fake_sdk.call(fake_sdk.client, model="gpt-test", messages=[], retry_backoff=0.1))
    assert exc.value is error
    assert fake_sdk.create.await_count == 3
    assert [call.args[0] for call in sleep.await_args_list] == [0.1, 0.2]


@pytest.mark.parametrize(
    "client_name, error, message",
    [
        (None, RuntimeError, "OpenAI client is not initialised"),
        ("sync_client", TypeError, "Sync OpenAI client provided"),
    ],
)
def test_safe_chat_completion_invalid_client(fake_sdk, client_name, error, message):
    client = getattr(fake_sdk, client_name) if client_name else None
    with pytest.raises(error, match=message):
        asyncio.run(fake_sdk.call(client, model="gpt-test", messages=[]))


def test_safe_chat_completion_with_real_sdk(monkeypatch):
    import openai as sdk
    from openai.types.chat import ChatCompletion

    from utils.openai_utils import safe_chat_completion

    response = ChatCompletion(
        id="chatcmpl-test", choices=[], created=1677652288, model="gpt-test", object="chat.completion"
    )
    client = sdk.AsyncOpenAI(api_key="test-key")
    monkeypatch.setattr(client.chat.completions, "create", AsyncMock(return_value=response))
    try:
        assert asyncio.run(safe_chat_completion(client, model="gpt-test", messages=[])) is response
    finally:
        asyncio.run(client.close())


def test_packages_import_without_optional_providers():
    code = """
import importlib
import sys

class NoProviders:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'openai', 'fastapi', 'uvicorn', 'google', 'redis', 'kafka', 'cv2',
            'rdflib', 'SPARQLWrapper', 'marimo', 'econml', 'doubleml', 'langchain',
            'supabase', 'psycopg2', 'a2a', 'ap2',
        }:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, NoProviders())
for name in (
    'agents', 'agents.coordinators', 'agents.cross_functional', 'agents.protocols',
    'capstone', 'config', 'connectors', 'demos', 'environments', 'models', 'utils',
    'utils.openai_utils', 'utils.env', 'utils.logger', 'models.enums', 'tests',
):
    importlib.import_module(name)
from utils import safe_chat_completion
assert callable(safe_chat_completion)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("target", ["test", "check"])
def test_make_targets_do_not_install_dependencies(tmp_path, target):
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "if sys.argv[1:3] not in (['-m', 'pytest'], ['-m', 'ruff']):\n"
        "    sys.exit(99)\n"
        "print(json.dumps(sys.argv[1:]))\n"
    )
    python.chmod(0o755)
    result = subprocess.run(
        [
            "make",
            target,
            f"VENV_DIR={tmp_path / 'absent-venv'}",
            f"PYTHON={python}",
            f"PYTHON_FOR_VENV={sys.executable}",
            f"PYTEST={python} -m pytest",
            f"RUFF={python} -m ruff",
            f"UV={python}",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("[")]
    assert commands[-1] == ["-m", "pytest", "tests"]
    if target == "check":
        assert commands[0] == [
            "-m",
            "ruff",
            "check",
            "agents",
            "capstone",
            "config",
            "connectors",
            "demos",
            "environments",
            "models",
            "utils",
            "tests",
        ]


@pytest.mark.parametrize(
    "provider, test_file",
    [("cv2", "tests/agents/test_cv.py"), ("openai", "tests/utils/test_nlp.py")],
)
def test_missing_locked_provider_is_collection_error(provider, test_file):
    code = (
        f"import sys; sys.modules[{provider!r}] = None; import pytest; "
        f"raise SystemExit(pytest.main(['-o', 'addopts=', '--collect-only', '-ra', '-q', {test_file!r}]))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "ModuleNotFoundError" in result.stdout
    assert provider in result.stdout


def test_coroutine_tests_run_without_pytest_asyncio():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:asyncio",
            "-o",
            "addopts=",
            "-q",
            "tests/connectors/test_dummy_db.py::test_get_customer_found",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_pytest_collects_without_coverage_plugin():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:pytest_cov",
            "--collect-only",
            "tests/config/test_config.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_direct_import_dependencies_are_declared():
    tomllib = pytest.importorskip("tomllib", reason="Dependency audit needs Python 3.11 or newer")
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())["project"]
    groups = {"runtime": metadata["dependencies"], **metadata["optional-dependencies"]}
    aliases = {
        "a2a": "a2a-sdk",
        "cv2": "opencv-python-headless",
        "dotenv": "python-dotenv",
        "google": "google-adk",
        "jose": "python-jose",
        "kafka": "kafka-python",
        "psycopg2": "psycopg2-binary",
        "sklearn": "scikit-learn",
    }
    declared = {
        group: {re.sub(r"[-_.]+", "-", re.match(r"[\w.-]+", requirement)[0]).lower() for requirement in reqs}
        for group, reqs in groups.items()
    }
    local = {path.stem for path in PROJECT_ROOT.iterdir() if path.is_dir() or path.suffix == ".py"}
    paths = subprocess.check_output(["git", "ls-files", "*.py"], cwd=PROJECT_ROOT, text=True).splitlines()
    missing = set()
    for name in paths:
        available = set().union(
            *(
                names
                for group, names in declared.items()
                if name.startswith("tests/") or group not in {"dev", "docs"}
            )
        )
        for node in ast.walk(ast.parse((PROJECT_ROOT / name).read_text(), filename=name)):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                root = module.split(".")[0]
                if root in local or root in sys.stdlib_module_names:
                    continue
                package = re.sub(r"[-_.]+", "-", aliases.get(root, root)).lower()
                if package not in available:
                    missing.add((package, name, node.lineno))
    assert not missing, sorted(missing)


def test_heavy_stacks_are_named_extras():
    tomllib = pytest.importorskip("tomllib", reason="Dependency audit needs Python 3.11 or newer")

    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())["project"]
    runtime = {re.match(r"[\w.-]+", requirement)[0] for requirement in metadata["dependencies"]}
    extras = metadata["optional-dependencies"]
    packages = {
        "spark": "pyspark",
        "gnn": "torch-geometric",
        "monitoring": "prometheus-client",
        "streaming": "redis",
        "cloud": "supabase",
        "auth": "passlib",
        "explainability": "shap",
    }
    for extra, package in packages.items():
        assert package not in runtime
        assert any(re.match(r"[\w.-]+", requirement)[0] == package for requirement in extras[extra])
    assert {"jsonschema", "statsmodels"} <= runtime
