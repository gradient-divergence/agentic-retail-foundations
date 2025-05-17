from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

"""Environment helper utilities.

Automatically loads a `.env` file from the project root so that environment
variables (e.g., ``OPENAI_API_KEY``) defined there become available via
``os.getenv``.  Uses `python-dotenv`, which is already listed in
`pyproject.toml` dependencies.
"""

__all__ = ["load_project_dotenv"]


def _find_project_root(start: Path | None = None) -> Path:
    """Traverse upwards until we find a directory that contains `pyproject.toml`."""
    current = start or Path(__file__).resolve().parent
    for _ in range(10):  # safety break
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return Path(__file__).resolve().parent


def load_project_dotenv() -> None:
    """Load environment variables from the project-level `.env` if present."""
    project_root = _find_project_root()
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
