import os
from pathlib import Path
from unittest.mock import patch

# Functions/Module to test
from utils.env import _find_project_root, load_project_dotenv

# Import the real load_dotenv to test its effects

# --- Test _find_project_root --- #


def test_find_project_root_found_in_start(tmp_path: Path):
    """Test finding pyproject.toml in the starting directory."""
    start_dir = tmp_path / "subdir"
    start_dir.mkdir()
    project_file = start_dir / "pyproject.toml"
    project_file.touch()

    found_root = _find_project_root(start=start_dir)
    assert found_root == start_dir


def test_find_project_root_found_one_level_up(tmp_path: Path):
    """Test finding pyproject.toml one level up."""
    project_root = tmp_path
    project_file = project_root / "pyproject.toml"
    project_file.touch()

    start_dir = project_root / "subdir1" / "subdir2"
    start_dir.mkdir(parents=True)

    found_root = _find_project_root(start=start_dir)
    assert found_root == project_root


def test_find_project_root_found_multiple_levels_up(tmp_path: Path):
    """Test finding pyproject.toml multiple levels up."""
    project_root = tmp_path / "level1"
    project_root.mkdir()
    project_file = project_root / "pyproject.toml"
    project_file.touch()

    start_dir = project_root / "subdir1" / "subdir2" / "subdir3"
    start_dir.mkdir(parents=True)

    found_root = _find_project_root(start=start_dir)
    assert found_root == project_root


# Skipping explicit "not found" test for _find_project_root due to
# complexity in reliably testing the __file__ based fallback.

# --- Test load_project_dotenv --- #


# We patch load_dotenv where it's imported in the utils.env module
@patch("utils.env.load_dotenv")
@patch("utils.env._find_project_root")
def test_load_dotenv_calls_load_dotenv_when_exists(mock_find_root, mock_load_dotenv, tmp_path: Path):
    """Test load_project_dotenv calls dotenv.load_dotenv when .env exists."""
    # Setup temporary project structure
    project_root = tmp_path / "proj1"
    project_root.mkdir()
    # (pyproject.toml isn't strictly needed here as we mock _find_project_root,
    # but good practice for realism)
    (project_root / "pyproject.toml").touch()
    env_file = project_root / ".env"
    env_file.touch()  # Create the .env file

    # Mock _find_project_root to return our temp root
    mock_find_root.return_value = project_root

    # Call the function
    load_project_dotenv()

    # Assert _find_project_root was called
    mock_find_root.assert_called_once()

    # Assert dotenv.load_dotenv was called with correct args
    mock_load_dotenv.assert_called_once_with(dotenv_path=env_file, override=False)


@patch("utils.env.load_dotenv")
@patch("utils.env._find_project_root")
def test_load_dotenv_does_not_call_when_not_exists(mock_find_root, mock_load_dotenv, tmp_path: Path):
    """Test load_project_dotenv does NOT call dotenv.load_dotenv if .env is missing."""
    # Setup temporary project structure (no .env file)
    project_root = tmp_path / "proj2"
    project_root.mkdir()
    (project_root / "pyproject.toml").touch()
    # --- .env file is NOT created --- #

    # Mock _find_project_root
    mock_find_root.return_value = project_root

    # Call the function
    load_project_dotenv()

    # Assert _find_project_root was called
    mock_find_root.assert_called_once()

    # Assert dotenv.load_dotenv was *NOT* called
    mock_load_dotenv.assert_not_called()


@patch("utils.env._find_project_root")
def test_load_dotenv_override_behavior(mock_find_root, tmp_path: Path, monkeypatch):
    """Test that existing environment variables are not overridden (override=False)."""
    # Setup temporary project structure with .env file
    project_root = tmp_path / "proj_override"
    project_root.mkdir()
    (project_root / "pyproject.toml").touch()
    env_file = project_root / ".env"
    env_file.write_text("EXISTING_VAR=dotenv_value\nNEW_VAR=new_value")

    # Mock _find_project_root
    mock_find_root.return_value = project_root

    # Set an existing environment variable *before* calling load_project_dotenv
    original_value = "original_value"
    monkeypatch.setenv("EXISTING_VAR", original_value)
    # Ensure NEW_VAR doesn't exist initially
    monkeypatch.delenv("NEW_VAR", raising=False)

    # Call the function - this will use the *real* load_dotenv
    load_project_dotenv()

    # Assert _find_project_root was called
    mock_find_root.assert_called_once()

    # Check os.environ directly to see the effect
    assert os.environ.get("EXISTING_VAR") == original_value  # Not overridden
    assert os.environ.get("NEW_VAR") == "new_value"  # Loaded new variable
