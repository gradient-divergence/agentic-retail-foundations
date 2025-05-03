# Agentic Retail Foundations

A modular, extensible framework for building, simulating, and analyzing agent-based AI architectures for retail. This project provides reusable agent models, data models, and interactive Marimo notebooks for rapid experimentation and research.

<table>
  <tr>
    <td width="80%">
      <strong>Featured Book:</strong><br>
      <a href="https://github.com/gradient-divergence/agentic-retail-foundations"><strong>Foundations of Agentic AI for Retail: Concepts, Technologies, and Architectures for Autonomous Retail Systems</strong></a> by Dr. Fatih Nayebi.
      <br><br>
      <em>Explore the future of retail powered by autonomous AI systems.</em>
      <br><br>
      <strong>Purchase on Amazon:</strong> <a href="https://www.amazon.com/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">US</a> | <a href="https://www.amazon.ca/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">CA</a> | <a href="https://www.amazon.co.jp/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">JP</a> | <a href="https://www.amazon.co.uk/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">UK</a> | <a href="https://www.amazon.de/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">DE</a> | <a href="https://www.amazon.fr/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">FR</a> | <a href="https://www.amazon.in/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">IN</a> | <a href="https://www.amazon.it/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">IT</a> | <a href="https://www.amazon.es/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">ES</a>
      <br>
      <strong>Associated Code Repository:</strong> <a href="https://github.com/gradient-divergence/agentic-retail-foundations">gradient-divergence/agentic-retail-foundations</a>
    </td>
    <td width="20%" align="center" valign="top">
      <a href="https://www.amazon.com/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">
        <img src="https://github.com/gradient-divergence/.github/blob/main/book-cover.png" alt="Book Cover: Foundations of Agentic AI for Retail" width="150"/>
      </a>
    </td>
  </tr>
</table>

## Directory Structure

```
agentic-retail-foundations/
    agents/           # Agent logic (BDI, OODA, etc.)
    models/           # Data models (product, inventory, sales, etc.)
    tests/            # Unit and integration tests
    notebooks/        # Marimo interactive notebooks
    docs/             # MkDocs documentation
    __init__.py       # Package marker
    README.md         # Project overview and instructions
    .env              # Environment variables (never commit secrets)
    pyproject.toml    # Linting, type checking, dependencies
    requirements.txt  # (Optional) requirements file
    mkdocs.yml        # Documentation site config
    PROJECT_PLAN.md   # Living project plan and checklist
    .gitignore        # Git ignore rules
```

## Setup Instructions

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/gradient-divergence/agentic-retail-foundations.git
    cd agentic-retail-foundations
    ```

2.  **Install `uv` (if you don't have it):**
    Follow the instructions at https://github.com/astral-sh/uv for installation. `uv` is used for fast Python package management and is required to create the virtual environment.

3.  **Create Virtual Environment and Install Dependencies:**
    Use the Makefile target which utilizes `uv` to create a `.venv` directory and install all project dependencies defined in `pyproject.toml`.
    ```sh
    make venv
    # or make install
    ```
    This installs both runtime and development dependencies into the `.venv` directory.

4.  **Activate the Virtual Environment (Choose one):**

    *   **Option A (Manual Activation - Recommended for interactive use):** Activate the environment in your current shell session. You'll need to do this each time you open a new terminal.
        ```sh
        source .venv/bin/activate
        ```
        Your prompt should now show `(.venv)` or similar.

    *   **Option B (Makefile Shell - Convenient shortcut):** Launch a *new* shell session with the environment already activated and the project name in the prompt.
        ```sh
        make shell
        ```
        Type `exit` to leave this specialized shell and return to your original terminal.

5.  **Set up Environment Variables:**
    - Copy `.env.example` to `.env`.
      ```sh
      cp .env.example .env
      ```
    - Edit `.env` and fill in required secrets/API keys (e.g., `OPENAI_API_KEY`). **Never commit the `.env` file.**

6.  **Download Models (if applicable):**
    - If the project requires specific pre-trained models (like for the `ShelfMonitoringAgent`), follow instructions provided separately to download and place them in the expected location (e.g., update `model_path` in the relevant notebook/script).

7.  **Install Pre-commit Hooks (Optional but Recommended):**
    This helps ensure code quality before commits. Activate your environment first if using the manual command.
    ```sh
    make precommit
    # Or, after activating: pre-commit install
    ```

## Usage

Common development tasks are managed via the `Makefile.mk`.

**Important:**
*   Most `make` commands (like `make lint`, `make test`) will automatically use the tools within the `.venv` directory.
*   If you need to run commands like `python`, `marimo`, or `pip` directly, ensure your virtual environment is activated first (either via `source .venv/bin/activate` or by using `make shell`).

-   **Launch an Activated Shell:**
    ```sh
    make shell         # Starts a new shell session with the venv active
    ```

-   **Run Marimo Notebooks (ensure venv is active):**
    ```sh
    # First, activate: source .venv/bin/activate OR use 'make shell'
    marimo edit notebooks/core-technologies-enabling-agentic-retail.py
    # Or use 'marimo run ...' for a read-only view
    ```

-   **Run Linters and Formatters:**
    ```sh
    make lint          # Check code style and quality with Ruff
    make format        # Auto-format code with Ruff
    make format-check  # Check formatting without applying changes (for CI)
    ```

-   **Run Type Checks:**
    ```sh
    make type-check    # Run MyPy static type checker
    ```

-   **Run Tests:**
    ```sh
    make test          # Run tests with pytest
    make coverage      # Run tests and generate a coverage report
    ```

-   **Build/Serve Documentation:**
    ```sh
    make docs-build    # Build the documentation site (outputs to 'site/')
    make docs-serve    # Serve documentation locally with live reload
    ```

-   **Clean Project:**
    ```sh
    make clean         # Remove cache files, build artifacts, etc.
    make clean-venv    # Remove the entire .venv directory
    ```

-   **See All Commands:**
    ```sh
    make help
    ```

## Development Best Practices

-   Use descriptive names and docstrings for all public classes/functions.
-   Encapsulate logic in modules (`agents/`, `connectors/`, etc.); avoid complex logic directly in notebooks.
-   Configure tools (`ruff`, `mypy`, `pytest`, `coverage`) in `pyproject.toml`.
-   Use the Makefile targets (`make lint`, `make type-check`, `make test`) regularly.
-   Write idempotent, reactive Marimo cells where possible.
-   Store secrets and environment-specific configurations in `.env` (and ensure `.env` is in `.gitignore`).
-   Follow the project plan in `PROJECT_PLAN.md` for phased development.

## Contribution Guidelines

-   Fork the repo and create a feature branch.
-   Write tests for new features and bug fixes in the `tests/` directory.
-   Ensure all checks pass (`make ci` or `make format-check lint type-check test coverage`).
-   Document new modules/functions and update the main documentation (`docs/`) if necessary.
-   Update the project plan (`PROJECT_PLAN.md`) as needed.
-   See `CONTRIBUTING.md` (to be added) for more details.

## Documentation

-   Main docs source: `docs/`
-   Build/serve docs using `make docs-build` / `make docs-serve`.
-   Online docs: (to be published)
-   Marimo: marimo docs

## GitHub Repository

-   https://github.com/gradient-divergence/agentic-retail-foundations