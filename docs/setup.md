# Setup and Usage

This guide explains how to set up the project environment and run the examples.

## Prerequisites

*   Python 3.10 or higher
*   `pip` (or ideally `uv` for faster dependency management)
*   Git
*   (Optional but Recommended) Docker and Docker Compose (for running dependencies like Redis, Kafka easily)
*   (Optional) Access to an OpenAI API Key (for LLM features), set as `OPENAI_API_KEY` environment variable.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/gradient-divergence/agentic-retail-foundations.git
    cd agentic-retail-foundations
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv .venv         # Create venv
    source .venv/bin/activate  # Activate (Linux/macOS)
    # .venv\Scripts\activate   # Activate (Windows - Command Prompt)
    # .venv\Scripts\Activate.ps1 # Activate (Windows - PowerShell)
    ```

3.  **Install Dependencies:**
    We recommend using `uv` for faster installation.
    ```bash
    # Install uv if you don't have it
    pip install uv 
    
    # Install core + development dependencies (linters, types, testing, docs)
    uv pip install -e ".[dev,docs]"
    ```
    *Alternatively, using pip:* 
    ```bash
    pip install -e ".[dev,docs]"
    ```
    The `-e` flag installs the project in editable mode.

4.  **Environment Variables:**
    Copy the example environment file and fill in your details (especially the OpenAI key if using LLM features):
    ```bash
    cp .env.example .env
    # Now edit .env with your text editor
    nano .env 
    ```

5.  **(Optional) Start Services:**
    Some demos require external services like Redis or Kafka. If you have Docker installed, you might use a `docker-compose.yml` file (if provided in the project) to start these easily:
    ```bash
    # docker-compose up -d # Uncomment if docker-compose.yml exists
    ```

## Running Demos

Most demonstration scripts are located in the `demos/` directory. Ensure your virtual environment is activated.

*   **Standard Demos:** Run directly with Python.
    ```bash
    python demos/agent_communication_demo.py
    python demos/task_allocation_cnp_demo.py
    # ... and others
    ```
*   **API/Service Demos:** Use `uvicorn`.
    ```bash
    uvicorn demos.inventory_api_demo:app --reload --port 8001
    uvicorn demos.api_gateway_demo:app --reload --port 8000 
    # ... etc. Requires dependencies like Redis to be running
    ```
*   **Spark Demo:** Requires a Spark environment.
    ```bash
    # Example spark-submit command (adjust master, packages as needed)
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 demos/spark_streaming_demo.py
    ```
*   **Feedback Loop Demo:** Runs indefinitely until stopped (Ctrl+C).
    ```bash
    python demos/dynamic_pricing_feedback_demo.py
    ```

Refer to the [Examples Overview](examples/index.md) for details on each demo.

## Running Notebooks

Interactive notebooks explaining concepts and running demos are in the `notebooks/` directory.

```bash
# Ensure venv is active
marimo edit notebooks/multi-agent-systems-in-retail.py
# Or run headlessly (less common for exploration)
# marimo run notebooks/multi-agent-systems-in-retail.py --headless
```

## Running Tests & Checks

Use the Makefile for convenience (or run commands directly):

*   **Run all tests:** `make test` (runs `pytest .`)
*   **Check types:** `make typecheck` (runs `mypy .`)
*   **Check formatting/linting:** `make lint` (runs `ruff check .` and `ruff format --check .`)
*   **Apply formatting/linting fixes:** `make format` (runs `ruff check --fix .` and `ruff format .`)

## Building Documentation

*   **Preview locally:** `make docs-serve` (runs `mkdocs serve`)
*   **Build static site:** `make docs-build` (runs `mkdocs build` - output in `site/` directory)

```sh
# Run linters
make lint

# Run tests
make test

# Run a notebook (after activating)
marimo edit notebooks/your_notebook.py