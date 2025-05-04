import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Keep essential imports for notebook structure and potential future use
    import marimo as mo
    import logging
    import asyncio
    import pandas as pd # Keep if used in any display logic
    from datetime import datetime, timedelta # Keep if used

    # Configure logging once
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("EndToEndIntegrationNotebook") 

    # Return only commonly needed modules for UI/basic ops
    return asyncio, datetime, logger, logging, mo, pd, timedelta


@app.cell
def _(mo):
    mo.md(
        r"""
        # End-to-End Integration for Autonomous Retail\index{end-to-end integration}\index{autonomous retail}

        Understand the principles and practices essential for end-to-end integration in autonomous retail systems. This chapter provides you with frameworks for system-wide coordination, real-time decision-making, and effective agent orchestration, positioning you to overcome integration challenges and optimize retail operations comprehensively.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("### Order Orchestration Simulation")
    run_ord_orch_button = mo.ui.button(label="Run Order Orchestration Demo")
    ord_orch_logs = mo.state([]) # To display completion message
    return ord_orch_logs, run_ord_orch_button


@app.cell
def _(asyncio, mo, ord_orch_logs, run_ord_orch_button):
    from demos.order_orchestration_demo import run_orchestration_simulation

    async def run_demo():
        # Demo prints logs to console
        await run_orchestration_simulation()
        ord_orch_logs.set_value(["Order Orchestration Demo Completed (check console)."])

    _ = run_ord_orch_button.on_click(lambda: asyncio.create_task(run_demo()))

    return mo.vstack([run_ord_orch_button, mo.md("```\n{ord_orch_logs.value}\n```")])


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Event-Driven Inventory API
        
        This demo runs a FastAPI service. Start it separately:
        ```bash
        uvicorn demos.inventory_api_demo:app --reload --port 8001
        ```
        You can then interact with it using tools like `curl` or Postman, or run the `inventory_api_client_demo.py` (if created).
        (No interactive button here as it's a background service).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### API Gateway Demo
        
        This demo runs a FastAPI API Gateway service. Start it separately:
        ```bash
        uvicorn demos.api_gateway_demo:app --reload --port 8000
        ```
        Requires backend services (like Inventory API) to be running.
        (No interactive button here as it's a background service).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Distributed State Management Demo (CRDT)
        
        This demo runs a FastAPI service demonstrating CRDT state updates via Redis. Start it separately:
        ```bash
        uvicorn demos.state_manager_demo:app --reload --port 8004
        ```
        (No interactive button here as it's a background service).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Stream Processing Demo (Spark)
        
        This demo runs a PySpark Streaming job. Requires a configured Spark environment with Kafka.
        Run using `spark-submit`:
        ```bash
        spark-submit demos/spark_streaming_demo.py
        ```
        (No interactive button here as it requires external execution).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Dynamic Pricing Feedback Loop Demo
        
        This demo runs a dynamic pricing agent loop. Requires Kafka and Redis.
        Run the script directly:
        ```bash
        python demos/dynamic_pricing_feedback_demo.py
        ```
        (No interactive button here as it runs indefinitely until stopped).
        """
    )
    return


@app.cell
def _(mo):
    # Keep final summary markdown if desired
    mo.md("End-to-end integration demonstrations extracted to separate scripts.")
    return


if __name__ == "__main__":
    app.run()
