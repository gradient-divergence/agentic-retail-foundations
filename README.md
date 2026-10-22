# Foundations of Agentic AI for Retail: Companion Code (Third Edition)

The runnable companion to *Foundations of Agentic AI for Retail*, Third Edition. Agent architectures, decision frameworks, coordination protocols, guardrails, evaluation and end-to-end demos, as the book prints them, with a regression test for each.

<table>
  <tr>
    <td width="60%">
      <a href="https://www.amazon.com/dp/1069422649"><strong>Foundations of Agentic AI for Retail: Designing, Building, and Operating Retail AI Agents, from First Principles to Production</strong></a>, Third Edition (October 2026), by Fatih Nayebi.
      <br><br>
      <em>A pricing agent marks down a winter coat in one region, exactly as its rules allow, and by morning the brand owns the outcome. The book asks when a retailer should hand a decision to software, and what evidence and controls make the handover worthwhile, from first principles to production.</em>
      <br><br>
      In paperback and hardcover, in full color. The Kindle edition will follow.<br>
      <strong>Buy on Amazon:</strong> <a href="https://www.amazon.com/dp/1069422649">US</a> | <a href="https://www.amazon.ca/dp/1069422649">CA</a> | <a href="https://www.amazon.co.uk/dp/1069422649">UK</a> | <a href="https://www.amazon.de/dp/1069422649">DE</a> | <a href="https://www.amazon.fr/dp/1069422649">FR</a> | <a href="https://www.amazon.it/dp/1069422649">IT</a> | <a href="https://www.amazon.es/dp/1069422649">ES</a> | <a href="https://www.amazon.co.jp/dp/1069422649">JP</a>
      <br><br>
      <sub>Earlier editions: <a href="https://www.amazon.com/dp/1069422622">second (2026)</a> | <a href="https://www.amazon.com/dp/1069422606">first (2025)</a></sub>
    </td>
    <td width="40%" align="center" valign="center">
      <a href="https://www.amazon.com/dp/1069422649">
        <img src="assets/book-cover-3rd-edition.jpg" alt="Book cover: Foundations of Agentic AI for Retail, Third Edition" width="300"/>
      </a>
    </td>
  </tr>
</table>

## Start Here

```sh
git clone https://github.com/gradient-divergence/agentic-retail-foundations.git
cd agentic-retail-foundations
uv sync --frozen                                    # Python 3.13, the locked environment

uv run --frozen python -m demos.intro_inventory_book_demo   # Chapter 1's reorder agent
uv run --frozen python -m demos.capstone_scaffold_demo      # the capstone's approval-and-receipt record
uv run --frozen python -m demos.capstone_gate_review_demo   # Chapter 14's simulated gate review
```

The tag `third-edition` is the code exactly as printed. For the full test suite, install every extra (`uv sync --frozen --all-extras`) and run `MPLBACKEND=Agg uv run --frozen python -m pytest`: 1,029 tests.

## The Book, Chapter by Chapter

Where each chapter's printed listings live. Every runnable listing in the book comes from this repository.

| Ch. | Chapter | Main files |
|---|---|---|
| 1 | Introduction: The Age of Agency | `demos/intro_inventory_book_demo.py` |
| 2 | Agent Architectures and Operating Models | `agents/raom_minimal.py`, `agents/bdi.py`, `agents/ooda.py`, `demos/bdi_inventory_demo_short.py` |
| 3 | Decision Frameworks I: Probabilistic Reasoning and Optimization | `demos/bayesian_reco_book_demo.py`, `demos/bayes_uplift_demo.py` |
| 4 | Decision Frameworks II: Sequential Decision-Making | `demos/mdp_dynamic_pricing_book_demo.py`, `demos/capstone_mdp_simulator_demo.py` |
| 5 | Decision Frameworks III: Learning and Planning | `demos/agent_optimization_ladder_demo.py`, `demos/store_fulfillment_book_demo.py` |
| 6 | Core Technologies Enabling Agentic Retail | `demos/structured_output_schema_demo.py`, `demos/tool_calling_rag_demo.py`, `demos/shelf_monitoring_book_demo.py` |
| 7 | Physical Perception and Sensor Intelligence | `agents/sensor.py`, `demos/knowledge_graph_book_demo.py`, `demos/promotion_causal_book_demo.py` |
| 8 | Multi-Agent Coordination and Interoperability | `demos/agent_communication_demo.py`, `demos/task_allocation_cnp_demo.py`, `demos/procurement_auction_demo_short.py`, `demos/inventory_sharing_demo_short.py` |
| 9 | End-to-End Agent Integration | `demos/order_orchestration_demo.py`, `demos/inventory_event_api_demo.py`, `demos/api_gateway_book_demo.py`, `demos/inventory_crdt_demo.py` |
| 10 | Building Production Agents | `demos/agent_session_state_demo.py`, `demos/inventory_agent_unit_test_demo.py`, `demos/monitoring_dashboard_api_demo.py` |
| 11 | Operating Agents (AgentOps) | `demos/agent_eval_hook_demo.py`, `demos/agentops_latency_middleware.py`, `demos/preference_pair_builder_demo.py` |
| 12 | Governance and Responsible Autonomy | `demos/hitl_approval_api_demo.py`, `demos/pricing_explainability_shap_demo.py` |
| 13 | Playbooks and Case Studies | `demos/inventory_agents_sdk_demo.py`, `demos/dynamic_pricing_agents_sdk_demo.py`, `demos/virtual_shopping_responses_demo.py` |
| 14 | Strategy, Maturity, and Futures | `demos/capstone_gate_review_demo.py` |

## What's New in the Third Edition

- **Repaired code.** The demos, agents and utilities were repaired for this edition, and the book's printed listings are drawn from this code.
- **Regression tests.** Each repair came with a test; the suite grew from 36 test files to 113, and all 1,029 tests pass.
- **Demos that say what they need.** A demo that needs an API key or a running service now names it and stops, instead of failing midway.
- **Current stack.** Python 3.13 and the latest compatible releases of every dependency, locked in `uv.lock`.
- **Tags.** `third-edition` (and `v3.1.0`) marks the code as printed in the published third edition; `v3.0.0` is the 7 October pre-release state, and `v2.0.0` is the second edition's.

<details>
<summary><strong>What the second edition added</strong></summary>

- **Retail Agent Operating Model (RAOM)** as the narrative spine across chapters and code.
- **Capstone workflow** that accumulates across chapters (inventory risk → supplier outreach → price protection → customer messaging → audit trail).
- **Three-plane architecture**: capability, control, and integration planes for system design.
- **Contract-first schemas** with runnable companion code and drift controls.
- **Evaluation pyramid + red-team packs** for safety, regression control, and release readiness.
- **Agent learning ladder**: prompt/tool tuning → SFT/PEFT → DPO/GRPO → RLHF/RLAIF → RFT.
- **Interoperability & standards**: MCP, agent-to-agent messaging, and emerging commerce protocols (ACP, AP2).
- **Context engineering** as a first-class design discipline (memory, retention boundaries, policies).
- **Expanded coverage**: offline RL, constrained RL, conformal prediction, GNNs, Mamba/SSM.
- **Teaching assets**: case packets, notation guide, glossary, and companion resources.

</details>

## Table of Contents

- [Start Here](#start-here)
- [The Book, Chapter by Chapter](#the-book-chapter-by-chapter)
- [What's New in the Third Edition](#whats-new-in-the-third-edition)
- [Companion Code Highlights](#companion-code-highlights)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Development Best Practices](#development-best-practices)
- [Contribution Guidelines](#contribution-guidelines)
- [Documentation](#documentation)

## Companion Code Highlights

*   **Retail Agent Operating Model (RAOM):** Minimal RAOM implementation and examples (`agents/raom_minimal.py`).
*   **Capstone Workflow:** End-to-end orchestration scaffolding with schemas, policies, tools, and tracing (`capstone/`).
*   **Agent Architectures:** BDI, OODA, Q-learning, Bayesian, causal, and LLM-based agents (`agents/`).
*   **Coordination Protocols:** Contract Net, auctions, and inventory sharing (`agents/protocols/`).
*   **Interoperability & Standards:** MCP tooling contracts and commerce protocols (ACP/AP2) via demos and manifests (`demos/`).
*   **Evaluation & Red Teaming:** Scenario packs and scoring rubrics (`evals/`, `redteam/`).
*   **Retail Data Models:** Pydantic models for core retail concepts (`models/`).
*   **Utilities & Observability:** Monitoring, event bus, planning, CRDTs, and NLP helpers (`utils/`).
*   **Notebooks & Demos:** Marimo notebooks and runnable demos aligned to the book (`notebooks/`, `demos/`).
*   **Testing & Tooling:** Pytest suite, Ruff, MyPy, and Makefile automation (`tests/`, `Makefile`).

## Directory Structure

```
agentic-retail-foundations/
├── agents/               # Core agent logic, protocols, and specific agent types
│   ├── coordinators/     # Coordinator agent implementations
│   ├── cross_functional/ # Agents spanning multiple business functions
│   ├── protocols/        # Coordination protocols (CNP, Auction, Inventory Sharing)
│   └── ...
├── assets/               # Diagrams and static assets (book cover, figures)
├── capstone/             # End-to-end capstone workflow scaffolding
├── config/               # Configuration helpers
├── connectors/           # Interfaces to external systems (mocked connectors)
├── demos/                # Standalone demo scripts and protocol examples
├── docs/                 # MkDocs documentation source files
├── environments/         # Simulation environments (e.g., MDP for RL)
├── evals/                # Evaluation scenarios and scoring rubrics
├── models/               # Pydantic data models for retail concepts
├── notebooks/            # Marimo notebooks for exploration and visualization
├── redteam/              # Red-team scenarios for safety testing
├── tests/                # Unit and integration tests (pytest)
├── utils/                # Utilities (monitoring, planning, NLP, event bus, etc.)
├── .env.example          # Example environment variables template
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile
├── mkdocs.yml
├── pyproject.toml
└── README.md
```

## Setup Instructions

1.  **Prerequisites:**
    *   Git
    *   Python 3.13 (the current dependency stack excludes 3.14)
    *   `uv` (recommended for fast environment/package management)

2.  **Clone the repository:**
    ```sh
    git clone https://github.com/gradient-divergence/agentic-retail-foundations.git
    cd agentic-retail-foundations
    ```

3.  **Install `uv` (if not already installed):**
    Follow the official instructions: https://github.com/astral-sh/uv

4.  **Create Virtual Environment & Install Dependencies:**
    ```sh
    uv sync --frozen
    # Complete test environment:
    uv sync --frozen --all-extras
    ```

5.  **Activate the Virtual Environment:**
    ```sh
    source .venv/bin/activate
    ```
    Run commands with `uv run --frozen` to use the locked environment without activation.

6.  **Set up Environment Variables:**
    *   Copy the example environment file:
        ```sh
        cp .env.example .env
        ```
    *   Edit `.env` and add your necessary API keys or configuration secrets (e.g., `OPENAI_API_KEY`).
    *   **Important:** `.env` is listed in `.gitignore` and should **never** be committed.

7.  **Optional: OpenAI Agents SDK demos**
    *   The SDK dependencies are included in the frozen installation above.
    *   SDK-specific demos (for example, `demos/*_agents_sdk_demo.py`) import the SDK via
        `demos/openai_agents_sdk_import.py` to avoid collisions with this repo's local
        `agents/` package.

8.  **Optional: Protocol SDKs and reference implementations**
    *   Install protocol SDK dependencies:
        ```sh
        uv sync --frozen --extra agent_protocols
        ```
    *   ACP is treated as a spec-only protocol reference (no SDK dependency).
    *   Other optional stacks have named extras: `streaming` (Redis/Kafka), `spark`,
        `gnn`, `monitoring` (Prometheus), `cloud` (Postgres/Supabase), `auth`, and `explainability` (SHAP).
        Install one with `uv sync --frozen --extra <name>`; use `uv sync --frozen --all-extras` for the complete test environment.

9.  **Install Pre-commit Hooks (Recommended):**
    ```sh
    make precommit
    # or: pre-commit install
    ```

## Usage

Common development tasks are streamlined using the `Makefile`. Ensure your virtual environment is active (`source .venv/bin/activate` or `make shell`) when running Python scripts or tools like `marimo` directly.

*   **Run Marimo Notebooks:**
    ```sh
    marimo edit notebooks/<notebook_name>.py
    # e.g., marimo edit notebooks/multi-agent-systems-in-retail.py
    ```
    Use `marimo run ...` for a read-only view.

*   **Run Demo Scripts:**
    Run from the companion repository root:
    ```sh
    python -m demos.<demo_name>
    # e.g., python -m demos.task_allocation_cnp_demo
    ```

    Chapter 8's complete `approval_request` in `demos/book_pricing_contracts.json`
    proposes ending SKU-123's discount from Friday 16 October 2026 at 00:00 UTC
    until Monday 19 October at 00:00 UTC (exclusive), covering the promotion's
    remaining Friday-to-Sunday window. Its unit-margin check is gross margin:
    `(price - unit_cost_usd) / price`, about 45.45% at the proposed $22 price
    against $12 merchandise cost, above the 40% floor. Contribution per unit
    also deducts $4 in other variable selling costs: $20 - $12 - $4 = $4
    currently, or $22 - $12 - $4 = $6 if approved. These per-unit contributions
    exclude the transfer's $20 transport/handling cost and the donor's $8
    expected contribution loss, deducted separately in the decision record.

*   **Run Linters / Formatters / Type Checks:**
    ```sh
    make lint          # Run Ruff linter
    make format        # Run Ruff formatter
    make format-check  # Check formatting without making changes (for CI)
    make type-check    # Run MyPy static type checker
    ```

*   **Run Tests:**
    ```sh
    make test          # Run offline pytest tests with the existing .venv
    make test PYTHON=/path/to/python  # Use an existing interpreter; installs nothing
    make check         # Run Ruff on packages/tests, then offline tests
    make coverage      # Run tests and generate coverage report
    make ci            # Run format-check, type-check, tests, coverage, docs build
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

*   **Modularity:** Keep agent logic, data models, utilities, and connectors in their respective directories.
*   **Configuration:** Use environment variables (`.env` file loaded via `python-dotenv`) for secrets and environment-specific settings.
*   **Typing:** Use Python type hints extensively. Run `make type-check` (`mypy`) regularly.
*   **Linting & Formatting:** Adhere to styles enforced by `ruff`. Run `make format` and `make lint` frequently.
*   **Testing:** Write unit tests (`pytest`) for individual functions/classes and integration tests for components working together.
    `make test` and `make check` use installed tools and never create an environment or install packages.
    Full-suite tests require the locked extras (`uv sync --all-extras`); provider calls use fake clients.
    Coroutine tests with synchronous fixtures can run offline using `asyncio.run` when `pytest-asyncio` is absent.
    Coverage is opt-in (`make coverage`, requires `pytest-cov`); normal tests use the installed runtime dependencies.
*   **Documentation:** Maintain docs in `docs/` and keep this README in sync with the book and code.

## Contribution Guidelines

Please refer to [`CONTRIBUTING.md`](CONTRIBUTING.md) for details on how to contribute to this project.

## Documentation

Build and browse the MkDocs documentation locally:

```sh
make docs-serve    # http://127.0.0.1:8000
```

Repository: https://github.com/gradient-divergence/agentic-retail-foundations
