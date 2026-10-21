# Setup and Usage

This guide explains how to set up the project environment and run the examples.

## Prerequisites

*   Python 3.13 (required by the current dependency stack)
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

2.  **Create the Locked Environment:** Run from the companion repository root.
    Install `uv` before this step, then let it create the local `.venv`.
    ```bash
    uv sync --frozen --extra dev --extra docs
    ```
    Select additional named extras for the demonstrations you run: `streaming`,
    `auth`, `cloud`, `monitoring`, `gnn`, or `spark`. Provider demos also require
    their explicitly documented credentials; the offline examples do not.

3.  **Run an Offline Example:**
    ```bash
    uv run --frozen python -m demos.preference_pair_builder_demo
    ```
    The logical output is one pair choosing “Returns accepted within 30 days
    with receipt.” over “Sure, just send it back anytime.”, with reason
    `preference_eval`. Start from a verified, release-pinned checkout; the
    published third-edition ref requires separate verification.

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

Run these commands from the companion root. Each command uses the locked environment.

*   **Standard Demos:** Run directly with Python.
    ```bash
    uv run --frozen python -m demos.agent_communication_demo
    uv run --frozen python -m demos.task_allocation_cnp_demo
    # ... and others
    ```
*   **API/Service Demos:** Use `uvicorn`.
    ```bash
    uv run --frozen --extra streaming uvicorn demos.inventory_api_demo:app --reload --port 8001
    uv run --frozen --extra streaming --extra auth uvicorn demos.api_gateway_demo:app --reload --port 8000
    # ... etc. Requires dependencies like Redis to be running
    ```
*   **Spark Demo:** Requires a Spark environment.
    ```bash
    # Candidate coordinates for locked Spark 4.2.0 / Scala 2.13; requires a qualified JVM and Kafka.
    .venv/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0 demos/stream_processing_sales_velocity_demo.py
    ```
*   **Feedback Loop Demo:** Runs indefinitely until stopped (Ctrl+C).
    ```bash
    uv run --frozen --extra streaming python -m demos.dynamic_pricing_feedback_demo
    ```

Refer to the [Examples Overview](examples/index.md) for details on each demo.

## Running Notebooks

Interactive notebooks explaining concepts and running demos are in the `notebooks/` directory.

```bash
# Ensure venv is active
uv run --frozen marimo edit notebooks/multi-agent-systems-in-retail.py
# Or run headlessly (less common for exploration)
# marimo run notebooks/multi-agent-systems-in-retail.py --headless
```

## Running Tests & Checks

Use the Makefile for convenience (or run commands directly):

*   **Run all tests:** `make test` (runs the `tests/` suite)
*   **Check types:** `make type-check` (runs `mypy .`)
*   **Check linting:** `make lint`. Check formatting separately with `make format-check`.
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
uv run --frozen marimo edit notebooks/your_notebook.py
```

### Synthetic streaming fixture

The book includes `demos/stream_processing_sales_velocity_demo.py`, not the older
`spark_streaming_demo.py`. Create topics `sales-transactions` and
`sales-velocity-metrics`; mount a writable `/checkpoints/sales-velocity`. Submit
three events with the same SKU/store and UTC timestamps 00:00:00, 00:04:00, and
00:16:00, quantities 2, 3, and 1, prices 10.0, and corresponding total values
20.0, 30.0, and 10.0. Advance event time past the first window's end plus the
one-minute watermark delay. The [00:00,00:15) window should report total quantity
5, average quantity per transaction 2.5, and transaction count 2. Its sales rate
is 5/15 units per minute; it is not uncensored demand.

Watermarks advance with arriving event times; idle streams may delay finalization.
Sufficiently late events may be omitted. Overlapping windows cannot be added,
and absent zero-sale windows require explicit treatment by consumers.

The locked fixture uses PySpark 4.2.0 and the Scala 2.13 connector. The
[Spark runtime guide](https://spark.apache.org/docs/4.2.0/index.html) lists Java
17, 21 and 25; the [Kafka integration guide](https://spark.apache.org/docs/4.2.0/streaming/structured-streaming-kafka-integration.html)
specifies the matching connector coordinates. These are prerequisites, not
evidence of a successful broker run.

For a local qualification, use an existing JDK and Kafka installation. The demo
resolves the broker as `kafka:9092`; run inside the broker's container network or
configure that hostname in your test environment. From the companion root:

```bash
java -version
.venv/bin/python -c 'import pyspark; print(pyspark.__version__)'
kafka-topics.sh --bootstrap-server kafka:9092 --create --if-not-exists --topic sales-transactions
kafka-topics.sh --bootstrap-server kafka:9092 --create --if-not-exists --topic sales-velocity-metrics
# Provision a writable /checkpoints/sales-velocity before submitting.
.venv/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0 demos/stream_processing_sales_velocity_demo.py
```

In another terminal, publish the synthetic events described above using
`kafka-console-producer.sh --bootstrap-server kafka:9092 --topic sales-transactions`.
Consume the result with
`kafka-console-consumer.sh --bootstrap-server kafka:9092 --topic sales-velocity-metrics --from-beginning`.
Verify the exact first-window totals after replay from the same checkpoint.
Kafka sinks can redeliver records, so downstream consumers must deduplicate by
product, store and window. JVM startup, connector resolution, broker delivery
and checkpoint recovery still require a live qualification.


### Full offline companion suite

The suite collects optional streaming, authentication, cloud, monitoring, GNN,
and Spark examples even though service and provider calls are mocked. Select
those extras when running the entire suite:

```bash
uv run --frozen --extra dev --extra auth --extra streaming --extra cloud --extra monitoring --extra gnn --extra spark python -m pytest
```

Service/broker/provider integration remains a separate qualification step.
