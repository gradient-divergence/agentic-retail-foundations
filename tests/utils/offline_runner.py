"""Run offline regressions while the environment job repairs the eager SDK import.

Usage: python tests/utils/offline_runner.py <pytest arguments>
The fake SDK supplies imports only; constructing a provider client fails.
"""

import sys
from pathlib import Path
from types import ModuleType

import pytest


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        import openai  # noqa: F401
    except ModuleNotFoundError:
        sdk = ModuleType("openai")
        types = ModuleType("openai.types")
        chat = ModuleType("openai.types.chat")

        class UnavailableClient:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Provider calls are unavailable in the offline runner")

        sdk.AsyncOpenAI = UnavailableClient
        sdk.OpenAI = UnavailableClient
        chat.ChatCompletion = object
        chat.ChatCompletionMessageParam = dict
        sys.modules.update({"openai": sdk, "openai.types": types, "openai.types.chat": chat})
    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
