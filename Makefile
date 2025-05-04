# Use bash for more predictable scripting
SHELL := /bin/bash

# Default target when 'make' is run without arguments
.DEFAULT_GOAL := help

# Automatically determine project name from the current directory
PROJECT_NAME := $(shell basename $(CURDIR))

# Directories containing project source code and tests
# Adjust SRC_DIRS if your structure differs (e.g., src/agents instead of agents/)
SRC_DIRS    := agents models utils connectors config environments demos notebooks
TEST_DIRS   := tests # Assuming tests are in a 'tests' directory


# Define VENV_DIR and VENV_BIN for clarity
VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin

# --- Tool Definitions ---
# UV is assumed to be available globally for initial venv creation.
# Other tools explicitly point to the versions inside the virtual environment.
# The '?=' allows overriding from the command line if needed (e.g., make PYTHON=python3.11 venv)
UV          ?= uv
PYTHON      ?= $(VENV_BIN)/python
RUFF        ?= $(VENV_BIN)/ruff
MYPY        ?= $(VENV_BIN)/mypy
PYTEST      ?= $(VENV_BIN)/pytest
COVERAGE    ?= $(VENV_BIN)/coverage
MKDOCS      ?= $(VENV_BIN)/mkdocs
PRECOMMIT   ?= $(VENV_BIN)/pre-commit

# Phony targets prevent conflicts with files of the same name
.PHONY: venv install clean help lint format format-check type-check test coverage docs-build docs-serve precommit ci check clean-venv default shell

# Default target
default: help

# Helper to list targets with descriptions
help: ## Show this help message and exit
	@echo "Usage: make [-f Makefile.mk] <target>\n"
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	   | sort \
	   | awk 'BEGIN {FS = ":.*?##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

###############################################################################
# Environment & Dependency Management
###############################################################################

# The main target to set up the environment and install dependencies.
# It relies on the .venv/pyvenv.cfg target to do the initial creation.
venv: $(VENV_DIR)/pyvenv.cfg ## Set up venv, install base tools, and project dependencies
	@echo "--> Installing project dependencies into $(VENV_DIR)..."
	$(VENV_BIN)/uv pip install --no-cache-dir .[dev]
	@echo "--> Virtual environment setup and dependencies installed in $(VENV_DIR)."
	@echo "--> Activate it using: source $(VENV_BIN)/activate"

# This target creates the virtual environment using the *global* uv command
# and installs essential tools (pip, uv) into it.
# It only runs if the marker file (.venv/pyvenv.cfg) doesn't exist or pyproject.toml is newer.
$(VENV_DIR)/pyvenv.cfg: pyproject.toml
	@echo "--> Creating virtual environment in $(VENV_DIR) using global uv..."
	@command -v $(UV) >/dev/null 2>&1 || { echo >&2 "Error: '$(UV)' command not found in PATH. Please install uv (see https://github.com/astral-sh/uv)."; exit 1; }
	# Create the venv using global uv
	$(UV) venv $(VENV_DIR) --seed --python $(shell which python3 || which python)
	@echo "--> Installing/upgrading pip and installing uv within $(VENV_DIR)..."
	# Use global uv again, but target the venv's python to install packages INTO the venv
	$(UV) pip install --python $(VENV_BIN)/python --no-cache-dir --upgrade pip uv
	@touch $(VENV_DIR)/pyvenv.cfg # Create marker file to indicate venv is set up


# Alias for venv, ensures dependencies are installed.
install: venv ## Alias for 'make venv' - ensures environment and dependencies are set up.
	@echo "--> Environment checked/updated. Dependencies installed via 'make venv'."

shell: venv ## Launch a new shell with the virtual environment activated and project name in prompt
	@echo "--> Launching a new shell for project '$(PROJECT_NAME)' with venv activated."
	@echo "--> Type 'exit' to return."
	$(SHELL) -c " \
		source $(VENV_BIN)/activate; \
		export PS1='($(PROJECT_NAME)) $$PS1'; \
		exec $(SHELL) \
	"

precommit: venv ## Install Git pre-commit hooks (ensures venv exists first)
	@echo "--> Installing pre-commit hooks..."
	$(PRECOMMIT) install

###############################################################################
# Code Quality (Targets assume venv is active or run via make)
###############################################################################

lint: venv ## Run Ruff linter (checks only)
	@echo "--> Running linter ($(RUFF))..."
	$(RUFF) check . $(TEST_DIRS)

format: venv ## Auto-format code with Ruff formatter & apply fixes
	@echo "--> Formatting code and fixing lint issues ($(RUFF))..."
	$(RUFF) format . $(TEST_DIRS)
	$(RUFF) check --fix . $(TEST_DIRS)

format-check: venv ## Check if code formatting is correct and lint rules pass
	@echo "--> Checking formatting and lint rules ($(RUFF))..."
	$(RUFF) format --check . $(TEST_DIRS)
	$(RUFF) check . $(TEST_DIRS) # Check lint rules without fixing

type-check: venv ## Run static type checks with MyPy
	@echo "--> Running static type checks ($(MYPY))..."
	$(MYPY) . # MyPy config often in pyproject.toml

###############################################################################
# Testing (Targets assume venv is active or run via make)
###############################################################################

test: venv ## Run unit & integration tests with Pytest
	@echo "--> Running tests ($(PYTEST))..."
	$(PYTEST) $(TEST_DIRS)

coverage: venv ## Generate test coverage report
	@echo "--> Generating test coverage report ($(COVERAGE))..."
	$(COVERAGE) run --source=$(subst $(eval) ,$(eval) ,,$(SRC_DIRS)) -m pytest $(TEST_DIRS)
	$(COVERAGE) report -m --fail-under=80 # Example: fail if coverage < 80%
	@echo "--> HTML report generated: htmlcov/index.html"
	$(COVERAGE) html

###############################################################################
# Documentation (Targets assume venv is active or run via make)
###############################################################################

# Add variable for configurable port
MKDOCS_PORT ?= 8000

docs-serve: venv ## Serve docs locally with live reload
	@echo "--> Serving documentation locally ($(MKDOCS))..."
	# Use the variable for the address
	$(MKDOCS) serve -a localhost:$(MKDOCS_PORT)

docs-build: venv ## Build static documentation site
	@echo "--> Building documentation ($(MKDOCS))..."
	$(MKDOCS) build --clean

###############################################################################
# Continuous Integration composite targets
###############################################################################

check: lint type-check test ## Run linters, type checks, and tests (no coverage)
	@echo "--> All checks passed."

ci: format-check type-check test coverage docs-build ## Full CI pipeline checks (lint implied by format-check)
	@echo "--> CI checks passed."

###############################################################################
# Maintenance utilities
###############################################################################

clean: ## Remove caches, build artefacts, coverage data, and __pycache__
	@echo "--> Cleaning up project artefacts..."
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info .coverage coverage.xml coverage.json
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "--> Cleanup complete."

# Optional: Target to remove the virtual environment entirely
clean-venv: ## Remove the virtual environment directory
	@echo "--> Removing virtual environment directory: $(VENV_DIR)..."
	rm -rf $(VENV_DIR)
	@echo "--> Virtual environment removed."
