# Agentic Retail Foundations

A modular, extensible Python framework for building, simulating, and analyzing agent-based AI architectures tailored for the retail sector. This project provides reusable agent models (BDI, OODA), coordination protocols (Contract Net, Auctions), data models, utility functions (NLP, planning, monitoring), and interactive Marimo notebooks for rapid experimentation, research, and development of autonomous retail systems.

<table>
  <tr>
    <td width="60%">
      <strong>Featured Book:</strong><br>
      <a href="https://github.com/gradient-divergence/agentic-retail-foundations"><strong>Foundations of Agentic AI for Retail: Concepts, Technologies, and Architectures for Autonomous Retail Systems</strong></a> by Dr. Fatih Nayebi.
      <br><br>
      <em>Explore the future of retail powered by autonomous AI systems.</em>
      <br><br>
      <strong>Purchase on Amazon:</strong> <a href="https://www.amazon.com/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">US</a> | <a href="https://www.amazon.ca/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">CA</a> | <a href="https://www.amazon.co.jp/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">JP</a> | <a href="https://www.amazon.co.uk/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">UK</a> | <a href="https://www.amazon.de/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">DE</a> | <a href="https://www.amazon.fr/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">FR</a> | <a href="https://www.amazon.in/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">IN</a> | <a href="https://www.amazon.it/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">IT</a> | <a href="https://www.amazon.es/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">ES</a>
      <br>
    </td>
    <td width="40%" align="center" valign="center">
      <a href="https://www.amazon.com/Foundations-Agentic-Retail-Technologies-Architectures/dp/1069422606">
        <img src="https://github.com/gradient-divergence/.github/blob/main/book-cover.png" alt="Book Cover: Foundations of Agentic AI for Retail" width="300"/>
      </a>
    </td>
  </tr>
</table>

## Key Features

*   **Modular Agent Architectures:** Implementations of common agent paradigms like Belief-Desire-Intention (BDI) and Observe-Orient-Decide-Act (OODA), adapted for retail scenarios (e.g., `StoreAgent`, `InventoryAgent`, `LLM`-based `RetailCustomerServiceAgent`).
*   **Coordination Protocols:** Examples of multi-agent coordination mechanisms:
    *   **Contract Net Protocol (CNP):** For task allocation (e.g., `RetailCoordinator`, `StoreAgent` bidding).
    *   **Auction Mechanisms:** For procurement and supplier selection (e.g., `ProcurementAuction`).
    *   **Inventory Sharing:** Collaborative inventory management across locations (`InventoryCollaborationNetwork`).
*   **Retail-Specific Data Models:** Pydantic models for core retail concepts like `Product`, `InventoryPosition`, `PurchaseOrder`, `Task`, `AgentMessage`, `Store`, `Supplier`, etc. located in the `models/` directory.
*   **Utility Functions:** Helpers for common tasks:
    *   **NLP:** Intent classification, entity extraction (order ID, product ID), sentiment analysis using LLMs (`utils/nlp.py`).
    *   **Planning:** Fulfillment planning, timeline calculation (`utils/planning.py`).
    *   **Monitoring:** Agent metric tracking and alerting (`utils/monitoring.py`).
    *   **OpenAI Integration:** Safe and robust wrappers for interacting with OpenAI APIs (`utils/openai_utils.py`).
    *   **Event Bus:** Simple pub/sub mechanism for inter-agent communication (`utils/event_bus.py`).
    *   **CRDTs:** Example Conflict-free Replicated Data Type (PN-Counter) for distributed state (`utils/crdt.py`).
*   **Interactive Notebooks:** Marimo notebooks (`notebooks/`) demonstrating concepts, agent interactions, and framework usage.
*   **Demo Scripts:** Standalone Python scripts (`demos/`) showcasing specific agent workflows and protocol examples (e.g., task allocation, procurement auction, inventory sharing).
*   **Testing Framework:** Unit and integration tests using `pytest` (`tests/`).
*   **Documentation:** Project documentation using MkDocs (`docs/`).
*   **Standardized Tooling:** Uses `ruff` for formatting/linting and `mypy` for type checking, configured via `pyproject.toml`. Dependency management via `uv`.

## Directory Structure

```
agentic-retail-foundations/
├── agents/               # Core agent logic, protocols, and specific agent types
│   ├── coordinators/     # Coordinator agent implementations
│   ├── cross_functional/ # Agents spanning multiple business functions
│   ├── protocols/        # Implementations of coordination protocols (CNP, Auction)
│   ├── __init__.py
│   └── ...               # Specific agent files (bdi.py, llm.py, ooda.py, store.py etc.)
├── connectors/           # Interfaces to external systems (databases, APIs) - currently mocks
├── demos/                # Standalone demo scripts for specific workflows
├── docs/                 # MkDocs documentation source files
├── environments/         # Simulation environments (e.g., MDP for RL)
├── models/               # Pydantic data models for retail concepts
├── notebooks/            # Marimo interactive notebooks for exploration and visualization
├── tests/                # Unit and integration tests (using pytest)
│   ├── agents/
│   ├── __init__.py       # Makes 'tests' a package
│   └── mocks.py          # Mock objects for testing dependencies
├── utils/                # Common utility functions (NLP, planning, monitoring, etc.)
├── .env.example          # Example environment variables template
├── .env                  # Local environment variables (GITIGNORED - add your secrets here)
├── .gitignore
├── .pre-commit-config.yaml # Configuration for pre-commit hooks
├── LICENSE               # Project License (e.g., MIT, Apache 2.0) - Needs to be added
├── Makefile.mk           # Makefile for common development tasks
├── PROJECT_PLAN.md       # Phased development plan and task tracking
├── README.md             # This file
├── mkdocs.yml            # MkDocs configuration
├── pyproject.toml        # Project metadata, dependencies, and tool configurations (ruff, mypy, etc.)
└── requirements.txt      # (Optional) For compatibility or specific deployment needs
```

## Setup Instructions

1.  **Prerequisites:**
    *   Git
    *   Python 3.10+
    *   `uv` (recommended for fast environment/package management)

2.  **Clone the repository:**
    ```sh
    git clone https://github.com/gradient-divergence/agentic-retail-foundations.git
    cd agentic-retail-foundations
    ```

3.  **Install `uv` (if not already installed):**
    Follow the official instructions: https://github.com/astral-sh/uv

4.  **Create Virtual Environment & Install Dependencies:**
    This command uses `uv` to create a virtual environment named `.venv` in the project root and install all dependencies listed in `pyproject.toml`.
    ```sh
    make install
    # or: uv venv && uv sync
    ```

5.  **Activate the Virtual Environment:**
    You need to activate the environment to use the installed packages and tools directly in your shell.
    ```sh
    source .venv/bin/activate
    ```
    Your shell prompt should now indicate the active environment (e.g., `(.venv) ...`). Alternatively, use `make shell` to start a new sub-shell with the environment automatically activated.

6.  **Set up Environment Variables:**
    *   Copy the example environment file:
        ```sh
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your necessary API keys or configuration secrets (e.g., `OPENAI_API_KEY`).
    *   **Important:** The `.env` file is listed in `.gitignore` and should **never** be committed to version control.

7.  **Install Pre-commit Hooks (Recommended):**
    This ensures code quality checks (like formatting and linting) run automatically before each commit.
    ```sh
    # Ensure your virtual environment is active
    make precommit
    # or: pre-commit install
    ```

## Usage

Common development tasks are streamlined using the `Makefile.mk`. Ensure your virtual environment is active (`source .venv/bin/activate` or `make shell`) when running Python scripts or tools like `marimo` directly.

*   **Run Marimo Notebooks:**
    ```sh
    # Make sure venv is active!
    marimo edit notebooks/<notebook_name>.py
    # e.g., marimo edit notebooks/multi-agent-systems-in-retail.py
    ```
    Use `marimo run ...` for a read-only view.

*   **Run Demo Scripts:**
    ```sh
    # Make sure venv is active!
    python demos/<demo_name>.py
    # e.g., python demos/task_allocation_cnp_demo.py
    ```

*   **Run Linters / Formatters / Type Checks:**
    ```sh
    make lint          # Run Ruff linter
    make format        # Run Ruff formatter
    make format-check  # Check formatting without making changes (for CI)
    make type-check    # Run MyPy static type checker
    ```

*   **Run Tests:**
    ```sh
    make test          # Run pytest test suite
    make coverage      # Run tests and generate coverage report
    make ci            # Run format-check, lint, type-check, and test (CI pipeline simulation)
    ```

*   **Build / Serve Documentation:**
    ```sh
    make docs-build    # Build MkDocs site (outputs to site/)
    make docs-serve    # Serve docs locally with live reload (http://127.0.0.1:8000)
    ```

*   **Manage Environment:**
    ```sh
    make shell         # Start a new shell with venv activated
    make clean         # Remove cache files (__pycache__), build artifacts
    make clean-venv    # Remove the .venv directory entirely
    make venv          # Recreate the virtual environment (if deleted)
    make install       # Sync dependencies into the existing venv
    ```

*   **List All Commands:**
    ```sh
    make help
    ```

## Development Best Practices

*   **Modularity:** Keep agent logic, data models, utility functions, and connectors in their respective directories. Avoid complex logic directly within notebooks; use them primarily for demonstration, visualization, and orchestration of underlying modules.
*   **Configuration:** Use environment variables (`.env` file loaded via `python-dotenv`) for secrets and environment-specific settings. Avoid hardcoding API keys or sensitive paths.
*   **Typing:** Use Python type hints extensively. Run `make type-check` (`mypy`) regularly.
*   **Linting & Formatting:** Adhere to the styles enforced by `ruff`. Run `make format` and `make lint` frequently. Use pre-commit hooks (`make precommit`).
*   **Testing:** Write unit tests (`pytest`) for individual functions/classes and integration tests for components working together. Aim for reasonable test coverage. Run tests via `make test`.
*   **Documentation:** Write clear docstrings for public APIs (functions, classes, methods). Maintain project documentation in the `docs/` directory using MkDocs. Keep the README up-to-date.
*   **Git:** Use feature branches for development. Write clear, concise commit messages. Ensure `make ci` passes before merging/pushing.
*   **Project Planning:** Refer to `PROJECT_PLAN.md` for the development roadmap and task tracking.

## Contribution Guidelines

Please refer to `CONTRIBUTING.md` (to be added) for details on how to contribute to this project. General expectations include following the development best practices outlined above, ensuring tests pass, and documenting changes.

## Documentation Website

The full project documentation, generated using MkDocs, is available at: [Placeholder - Link to be added once deployed]

You can also build and serve the documentation locally using `make docs-serve`.

## GitHub Repository

*   Main repository: https://github.com/gradient-divergence/agentic-retail-foundations