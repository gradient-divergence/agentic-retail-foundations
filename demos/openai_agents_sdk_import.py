from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def import_openai_agents_sdk() -> ModuleType:
    """Import the OpenAI Agents SDK even when a local 'agents' package exists."""
    repo_root = Path(__file__).resolve().parents[1]
    original_path = list(sys.path)
    try:
        sys.path = [path for path in sys.path if path not in ("", str(repo_root))]
        return importlib.import_module("agents")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenAI Agents SDK is not installed. Install with: "
            "pip install --upgrade openai openai-agents python-dotenv"
        ) from exc
    finally:
        sys.path = original_path
