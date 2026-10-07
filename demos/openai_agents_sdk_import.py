from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def import_openai_agents_sdk() -> ModuleType:
    """Import the OpenAI Agents SDK even when a local 'agents' package exists."""
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("Set OPENAI_API_KEY to run this provider demo.")
    repo_root = Path(__file__).resolve().parents[1]
    cached = sys.modules.get("agents")
    if cached is not None and Path(getattr(cached, "__file__", "") or "").resolve() == (
        repo_root / "agents" / "__init__.py"
    ):
        raise ImportError(
            "The local agents package is already loaded; run this demo in a fresh Python process."
        )
    original_path = list(sys.path)
    try:
        sys.path = [path for path in sys.path if Path(path).resolve() != repo_root]
        return importlib.import_module("agents")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenAI Agents SDK is not installed. Install with: "
            "pip install --upgrade openai openai-agents python-dotenv"
        ) from exc
    finally:
        sys.path = original_path
