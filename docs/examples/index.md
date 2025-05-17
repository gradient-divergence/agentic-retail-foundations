# Examples and Demos

This section provides standalone scripts demonstrating various capabilities of the agentic retail framework.

These scripts are located in the `demos/` directory and can typically be run directly using `python demos/<script_name>.py` (unless otherwise noted). Ensure you have installed dependencies (`uv pip install -e .[dev]`) and configured any necessary environment variables (see `.env.example`).

## Agent Communication & Protocols

-   **`agent_communication_demo.py`:** 
    -   *Purpose:* Demonstrates basic FIPA-inspired messaging via `MessageBroker`.
    -   *Run:* `python demos/agent_communication_demo.py`
-   **`task_allocation_cnp_demo.py`:** 
    -   *Purpose:* Shows Contract Net Protocol for task allocation with `RetailCoordinator` and `StoreAgent`.
    -   *Run:* `python demos/task_allocation_cnp_demo.py`
-   **`procurement_auction_demo.py`:** 
    -   *Purpose:* Demonstrates a reverse auction for supplier selection via `ProcurementAuction`.
    -   *Run:* `python demos/procurement_auction_demo.py`
-   **`inventory_sharing_demo.py`:**
    -   *Purpose:* Simulates cooperative inventory transfers using `InventoryCollaborationNetwork`.
    -   *Run:* `python demos/inventory_sharing_demo.py`
-   **`openai_agents_multi_agent_demo.py`:**
    -   *Purpose:* Demonstrates coordinating multiple OpenAI agents via the Agents SDK.
    -   *Run:* `python demos/openai_agents_multi_agent_demo.py`

## Cross-Functional Coordination

-   **`product_launch_demo.py`:** 
    -   *Purpose:* Showcases `ProductLaunchCoordinator` orchestrating functional agents.
    -   *Run:* `python demos/product_launch_demo.py`

## Decision-Making Frameworks

-   **`recommendation.py`:** 
    -   *Purpose:* Contains logic for the Bayesian Recommendation Agent (needs separate demo script).
    -   *Run:* (No direct demo script yet)
-   **`dynamic_pricing.py`:** 
    -   *Purpose:* Trains and visualizes a Q-Learning agent for dynamic pricing.
    -   *Run:* `python demos/dynamic_pricing.py`
-   **`dynamic_pricing_feedback_demo.py`:** 
    -   *Purpose:* Runs a dynamic pricing agent with a real-time feedback loop.
    -   *Requires:* Running Redis and Kafka.
    -   *Run:* `python demos/dynamic_pricing_feedback_demo.py`

## Fulfillment Planning

-   **`fulfillment_planning_demo.py`:** 
    -   *Purpose:* Demonstrates order batching, assignment, and pathfinding using `FulfillmentPlanner`.
    -   *Run:* `python demos/fulfillment_planning_demo.py`

## Supporting Services & Integration (Run Separately)

-   **`inventory_api_demo.py`:** 
    -   *Purpose:* Event-driven inventory service (FastAPI).
    -   *Run:* `uvicorn demos.inventory_api_demo:app --reload --port 8001`
-   **`api_gateway_demo.py`:** 
    -   *Purpose:* API Gateway simulation (FastAPI).
    -   *Requires:* Backend services (e.g., inventory API) running.
    -   *Run:* `uvicorn demos.api_gateway_demo:app --reload --port 8000`
-   **`state_manager_demo.py`:** 
    -   *Purpose:* Distributed state demo using CRDTs/Redis (FastAPI).
    -   *Run:* `uvicorn demos.state_manager_demo:app --reload --port 8004`
-   **`spark_streaming_demo.py`:** 
    -   *Purpose:* Real-time sales velocity calculation (PySpark).
    -   *Requires:* Spark & Kafka environment.
    -   *Run:* `spark-submit demos/spark_streaming_demo.py`

## Notebooks

The `notebooks/` directory contains Marimo notebooks that provide narrative explanations and interactive visualizations related to the concepts demonstrated in these scripts.