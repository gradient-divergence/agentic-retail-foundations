# Contributing to Agentic Retail Foundations

Thank you for your interest in contributing to the Agentic Retail Foundations project! We welcome contributions from the community to help improve the framework, add new features, fix bugs, and enhance the documentation.

This guide provides instructions on how to get started, the development workflow, coding standards, and how to submit your contributions.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Fork and Clone](#fork-and-clone)
  - [Environment Setup](#environment-setup)
  - [Install Pre-commit Hooks](#install-pre-commit-hooks)
- [Development Workflow](#development-workflow)
  - [Create a Feature Branch](#create-a-feature-branch)
  - [Making Changes](#making-changes)
  - [Running Checks Locally](#running-checks-locally)
- [Coding Standards](#coding-standards)
  - [Formatting and Linting](#formatting-and-linting)
  - [Type Hinting](#type-hinting)
  - [Docstrings and Comments](#docstrings-and-comments)
  - [Naming Conventions](#naming-conventions)
- [Testing](#testing)
  - [Writing Tests](#writing-tests)
  - [Running Tests](#running-tests)
  - [Test Coverage](#test-coverage)
- [Documentation](#documentation)
  - [Updating Docstrings](#updating-docstrings)
  - [Updating Project Documentation](#updating-project-documentation)
- [Submitting Changes](#submitting-changes)
  - [Commit Messages](#commit-messages)
  - [Creating a Pull Request](#creating-a-pull-request)
  - [Code Review](#code-review)
- [Reporting Bugs and Suggesting Features](#reporting-bugs-and-suggesting-features)
- [Code of Conduct](#code-of-conduct)
- [Questions?](#questions)

## Getting Started

### Prerequisites

Ensure you have the following installed:

*   Git
*   Python 3.10 or later
*   `uv` (Python package manager) - See installation instructions: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

### Fork and Clone

1.  **Fork** the repository on GitHub.
2.  **Clone** your fork locally:
    ```sh
    git clone https://github.com/YOUR_USERNAME/agentic-retail-foundations.git
    cd agentic-retail-foundations
    ```
3.  **Add the upstream repository** to keep your fork synchronized:
    ```sh
    git remote add upstream https://github.com/gradient-divergence/agentic-retail-foundations.git
    ```

### Environment Setup

We use `uv` for managing the virtual environment and dependencies.

1.  **Create the virtual environment and install dependencies:**
    ```sh
    make install
    # This uses uv to create a .venv directory and installs packages from pyproject.toml
    ```
2.  **Activate the virtual environment:**
    ```sh
    source .venv/bin/activate
    ```
    You should see `(.venv)` appear in your shell prompt.

### Install Pre-commit Hooks

We use `pre-commit` to run checks (like formatting and linting) automatically before each commit. This helps maintain code quality.

```sh
# Ensure your virtual environment is active
make precommit
# or: pre-commit install
```

## Development Workflow

### Create a Feature Branch

Before starting work, create a new branch from the latest `main` branch:

```sh
# Fetch the latest changes from upstream
git fetch upstream

# Ensure your local main branch is up-to-date
git checkout main
git pull upstream main

# Create your feature branch
git checkout -b your-feature-name # e.g., feature/add-new-agent or fix/bug-in-nlp
```

### Making Changes

*   Write clear, modular, and well-documented code.
*   Follow the project's directory structure (refer to `README.md`).
*   If adding new functionality, consider if it requires new tests or documentation updates.
*   Keep changes focused on a single feature or bug fix per branch/pull request.

### Running Checks Locally

Before committing and pushing, run the quality checks to ensure your code meets the project standards:

```sh
# Format code (uses ruff format)
make format

# Check for linting errors (uses ruff check)
make lint

# Check for type errors (uses mypy)
make type-check

# Run the test suite (uses pytest)
make test

# Run all checks performed by CI (recommended before submitting PR)
make ci
```

If you installed pre-commit hooks, `make format` and `make lint` (or parts thereof) will run automatically when you commit.

## Coding Standards

### Formatting and Linting

*   We use **Ruff** for both formatting and linting, configured via `pyproject.toml`.
*   Run `make format` to automatically format your code according to the project style.
*   Run `make lint` to identify any style violations or potential issues.
*   The primary goal is consistency and readability.

### Type Hinting

*   Use Python type hints for all function signatures (arguments and return values), class attributes, and variables where appropriate.
*   Run `make type-check` (`mypy`) to catch type errors statically.
*   Aim for complete type coverage where feasible.

### Docstrings and Comments

*   Write clear and concise docstrings for all public modules, classes, functions, and methods. We recommend following the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings.
*   Use comments (`#`) sparingly to explain *why* something is done, rather than *what* is being done (the code should ideally explain the *what*).

### Naming Conventions

*   Follow standard Python naming conventions (PEP 8):
    *   `snake_case` for functions, methods, variables.
    *   `PascalCase` for classes.
    *   `UPPER_SNAKE_CASE` for constants.
*   Use descriptive names.

## Testing

### Writing Tests

*   New features, agent logic, or utility functions should include corresponding unit tests.
*   Place tests in the `tests/` directory, mirroring the structure of the main `agents/`, `models/`, `utils/` directories where applicable.
*   Use `pytest` as the testing framework.
*   Use mocks (e.g., from `unittest.mock` or `pytest-mock`, place reusable mocks in `tests/mocks.py`) to isolate units under test from external dependencies (like databases, APIs, or complex agent interactions).

### Running Tests

*   Run the full test suite:
    ```sh
    make test
    ```
*   Run tests for a specific file or directory:
    ```sh
    # Ensure venv is active
    pytest tests/utils/test_nlp.py
    pytest tests/agents/
    ```

### Test Coverage

*   While we strive for high test coverage, the focus is on testing critical paths and complex logic.
*   You can check coverage by running:
    ```sh
    make coverage
    ```
    This will run tests and generate an HTML report in the `htmlcov/` directory.

## Documentation

Maintaining good documentation is crucial.

### Updating Docstrings

*   Ensure all new or modified public functions, classes, and methods have accurate and informative docstrings.

### Updating Project Documentation

*   If your changes affect the overall project structure, setup, usage, or add significant new features, update the relevant sections in:
    *   `README.md`
    *   Files within the `docs/` directory (used by MkDocs).
*   You can preview changes to the MkDocs documentation locally:
    ```sh
    make docs-serve
    ```
    Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Submitting Changes

### Commit Messages

*   Write clear and concise commit messages.
*   Use the present tense ("Add feature X", "Fix bug Y").
*   Reference relevant issue numbers if applicable (e.g., "Fix #123: Handle edge case in auction protocol").

### Creating a Pull Request

1.  Push your feature branch to your fork on GitHub:
    ```sh
    git push origin your-feature-name
    ```
2.  Go to the original `agentic-retail-foundations` repository on GitHub.
3.  GitHub should prompt you to create a Pull Request (PR) from your recently pushed branch. Click that button.
4.  Fill out the PR template:
    *   Provide a clear title summarizing the changes.
    *   Write a detailed description explaining the purpose of the changes, what was done, and how it was tested.
    *   Link any relevant GitHub issues.
5.  Ensure the "Allow edits from maintainers" checkbox is ticked.
6.  Submit the Pull Request.

### Code Review

*   A project maintainer will review your PR.
*   Be prepared to respond to comments and make necessary changes.
*   Once approved and CI checks pass, your PR will be merged.

## Reporting Bugs and Suggesting Features

*   Use the **GitHub Issues** tracker for the main repository:
    *   Check if a similar issue already exists.
    *   For bugs, provide detailed steps to reproduce, expected behavior, actual behavior, and your environment details.
    *   For feature requests, clearly describe the proposed feature and its potential benefits.

## Code of Conduct

All contributors are expected to adhere to the project's **Code of Conduct** (link to be added - `CODE_OF_CONDUCT.md`). Please ensure you read and understand it.

## Questions?

If you have questions about contributing, feel free to open an issue on GitHub. 