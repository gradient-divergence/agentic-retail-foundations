# Use bash for more predictable scripting
SHELL := /bin/bash

# Default target when 'make' is run without arguments
.DEFAULT_GOAL := help

# Automatically determine project name from the current directory
PROJECT_NAME := $(shell basename $(CURDIR))

# Directories containing project source code and tests
# Adjust SRC_DIRS if your structure differs (e.g., src/agents instead of agents/)
SRC_DIRS    := agents capstone config connectors demos environments models utils
TEST_DIRS   := tests # Assuming tests are in a 'tests' directory


# Define VENV_DIR and VENV_BIN for clarity
VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin

# --- Tool Definitions ---
# UV synchronizes the locked environment using the pinned Python version.
# Other tools explicitly point to the versions inside the virtual environment.
# The '?=' allows overriding tools from the command line.
UV          ?= uv
PYTHON      ?= $(VENV_BIN)/python
RUFF        ?= $(VENV_BIN)/ruff
MYPY        ?= $(VENV_BIN)/mypy
PYTEST      ?= $(PYTHON) -m pytest
COVERAGE    ?= $(VENV_BIN)/coverage
MKDOCS      ?= $(VENV_BIN)/mkdocs
PRECOMMIT   ?= $(VENV_BIN)/pre-commit

# Phony targets prevent conflicts with files of the same name
.PHONY: venv install clean help lint format format-check type-check test coverage docs-build docs-serve precommit precommit-run ci check clean-venv default shell

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

venv: ## Sync locked dependencies and optional demo stacks
	@echo "--> Installing project dependencies into $(VENV_DIR)..."
	UV_PROJECT_ENVIRONMENT="$(VENV_DIR)" $(UV) sync --locked --all-extras
	@echo "--> Virtual environment setup and dependencies installed in $(VENV_DIR)."
	@echo "--> Activate it using: source $(VENV_BIN)/activate"

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

precommit-run: venv ## Run pre-commit checks on all files
	@echo "--> Running pre-commit checks on all files..."
	$(PRECOMMIT) run --all-files

###############################################################################
# Code Quality (Targets assume venv is active or run via make)
###############################################################################

lint: ## Run Ruff linter with installed tools (checks only; no installation)
	@echo "--> Running linter ($(RUFF))..."
	$(RUFF) check $(SRC_DIRS) $(TEST_DIRS)

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

test: ## Run offline tests with the selected interpreter (no installation)
	@echo "--> Running tests ($(PYTEST))..."
	$(PYTEST) $(TEST_DIRS)

coverage: ## Run tests with coverage and generate reports (requires pytest-cov)
	$(PYTEST) $(TEST_DIRS) --cov --cov-report=term-missing
	@echo "--> Generating test coverage report ($(COVERAGE))..."
	$(COVERAGE) report -m --fail-under=70 # Ensure fail-under matches pyproject.toml
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

check: lint test ## Run Ruff and offline tests with installed tools (no installation)
	@echo "--> All checks passed."

ci: format-check test coverage docs-build ## What CI runs (lint implied by format-check; mypy runs separately: make type-check)
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
