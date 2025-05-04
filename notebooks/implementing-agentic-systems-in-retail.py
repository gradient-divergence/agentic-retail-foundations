import marimo as mo
import marimo  # Ensure marimo is imported for app definition

__generated_with = "0.1.69"
app = marimo.App()


@app.cell
def __():
    # Import all required modules
    from fastapi import FastAPI, HTTPException
    from psycopg2.extras import RealDictCursor
    from supabase import create_client
    import os
    import psycopg2
    import marimo as mo

    # Return all modules needed by other cells
    return mo, FastAPI, HTTPException, RealDictCursor, create_client, os, psycopg2


@app.cell
def __(FastAPI, os, create_client):
    # Initialize FastAPI app
    app_api = FastAPI()

    # Initialize Supabase client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv(
        "SUPABASE_SERVICE_ROLE_KEY"
    )  # using a service role key for full DB access
    supabase = None
    try:
        supabase = create_client(url, key)
    except Exception as e:
        print(f"Failed to create Supabase client: {e}")

    return app_api, supabase


@app.cell
def __(mo):
    mo.md(
        r"""
        # Implementing Agentic Systems in Retail

        In this practical-focused chapter, you'll learn how to systematically implement, test, deploy, and scale agentic systems in real-world retail environments. From agent-oriented development practices to CI/CD pipelines, you'll be equipped with actionable strategies and methodologies essential for successful implementation.

        ## Code Example: Testing Framework for Retail Agents

        To illustrate testing in practice, let's create a simplified example. We'll write a small Python testing scenario for a hypothetical **InventoryAgent** that handles restocking logic. We assume the agent decides restock orders based on current stock and a predicted demand. We want to test that it issues a restock when stock is low relative to demand. We'll use a simple assert-based test (as one would in PyTest):
        """
    )
    return


@app.cell
def __(mo):
    # Define a simple InventoryAgent for testing
    class InventoryAgent:
        def __init__(self, safety_stock: int):
            # safety_stock: minimum units to keep as buffer
            self.safety_stock = safety_stock

        def evaluate_restock(self, current_stock: dict, predicted_demand: dict):
            """
            Decide restock orders for each item.
            Returns a dict of item -> order_quantity (0 if no restock needed).
            """
            orders = {}
            for item, stock in current_stock.items():
                demand = predicted_demand.get(item, 0)
                # If predicted demand exceeds current stock, plus a safety buffer, order more
                if stock < demand + self.safety_stock:
                    order_qty = (demand + self.safety_stock) - stock
                    if order_qty < 0:
                        order_qty = 0  # no negative orders
                    orders[item] = order_qty
                else:
                    orders[item] = 0
            return orders

    return InventoryAgent


@app.cell
def __(InventoryAgent, mo):
    # Test case for InventoryAgent
    def test_inventory_restock_logic():
        agent = InventoryAgent(safety_stock=10)
        # Scenario: low stock should trigger restock
        current_stock = {"Jeans": 5, "T-Shirt": 20}
        predicted_demand = {"Jeans": 15, "T-Shirt": 5}
        orders = agent.evaluate_restock(current_stock, predicted_demand)
        # The agent should order Jeans because 5 < 15+10, but not T-Shirt because 20 >= 5+10
        assert "Jeans" in orders and orders["Jeans"] >= 20, (
            "Jeans restock quantity should be at least 20"
        )
        assert orders["T-Shirt"] == 0, "T-Shirt should not be reordered"

        # Scenario: plenty of stock should result in no orders
        current_stock = {"Dress": 50}
        predicted_demand = {"Dress": 30}
        orders = agent.evaluate_restock(current_stock, predicted_demand)
        assert orders["Dress"] == 0, "No restock needed when stock is sufficient"

        print("All tests passed!")

    # Run the test function
    test_inventory_restock_logic()

    return test_inventory_restock_logic


@app.cell
def __(mo):
    mo.md(
        r"""
        *Explanation:* We defined a rudimentary `InventoryAgent` with a method `evaluate_restock` that decides how many units to order for each item. The logic here is: if current stock is less than predicted demand plus a safety buffer, it will calculate an order quantity to meet that demand and buffer. In `test_inventory_restock_logic()`, we create an agent with a safety stock of 10 units. We then test two scenarios:

        - In the first scenario, *Jeans* have current stock 5 and predicted demand 15. Since 5 is less than 15+10, the agent should decide to restock. We assert that the order for Jeans is at least 20 (in fact, it should be exactly 20 in this logic). For *T-Shirt*, stock is 20 and demand 5, which is above the threshold (5+10=15, stock is 20), so no restock – we assert the order is 0.
        - In the second scenario, stock is plenty relative to demand, so we expect no orders.

        ## Monitoring and Maintaining Agent Systems

        ### Code Example: Monitoring Dashboard Backend (FastAPI + Supabase)

        To support monitoring and maintenance, it's common to build a dashboard that displays the system's status. Supabase provides a convenient database (and you can also leverage its Auth and storage if needed), and FastAPI can serve as an API backend to query metrics or logs from the database and provide them to a frontend dashboard (in our case, maybe a SvelteKit app). Let's sketch a simple FastAPI endpoint that could serve as part of a monitoring API. This endpoint will retrieve some metrics from a Supabase (Postgres) database and return as JSON.
        """
    )
    return


@app.cell
def __(app_api, HTTPException, supabase):
    @app_api.get("/metrics/agents")
    def get_agent_metrics():
        if supabase is None:
            raise HTTPException(status_code=500, detail="Supabase client not available")
        try:
            res = supabase.table("agent_metrics").select("*").execute()
            if res.error:
                raise HTTPException(status_code=500, detail=res.error.message)
            return {"agents": res.data}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return get_agent_metrics


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Alternative Database Connection with psycopg2

        While we're using the Supabase Python client in our implementation, you can also connect directly to Supabase's PostgreSQL database using psycopg2:

        ```python
        # Database connection (Supabase Postgres)
        DB_URL = os.getenv("SUPABASE_DB_URL")  # e.g., "postgresql://user:pass@db.supabase.co:5432/postgres"
        try:
            conn = psycopg2.connect(DB_URL)
        except Exception as e:
            print("Failed to connect to Supabase DB:", e)
            conn = None


        @app.get("/metrics/agents")
        def get_agent_metrics():
            if conn is None:
                raise HTTPException(status_code=500, detail="DB connection not available")
            cur = conn.cursor(cursor_factory=RealDictCursor)
            # Fetch the latest metrics for each agent
            # For demonstration, assume one row per agent with current metrics
            cur.execute("SELECT agent_id, tasks_completed, avg_response_time, last_updated FROM agent_metrics;")
            rows = cur.fetchall()
            cur.close()
            # Convert to list of dict for JSON serialization (RealDictCursor already gives dict per row)
            return {"agents": rows}
        ```

        A few notes on this code:

        - We use `psycopg2` to connect to the Postgres database provided by Supabase. In a real deployment, you might use connection pooling (to reuse connections) and handle credentials securely (likely using environment variables as shown).
        - The `/metrics/agents` endpoint queries the `agent_metrics` table and returns all rows. Each row might look like `{"agent_id": "InventoryAgent-Store123", "tasks_completed": 250, "avg_response_time": 1.2, "last_updated": "2025-03-18T20:00:00Z"}`. The FastAPI framework automatically serializes the Python dict to JSON.
        - We wrap in an HTTPException if the DB connection isn't available, to return a proper 500 error.

        This is a simplistic example. In practice, you may want to add query parameters to filter or sort (e.g., `?agent_id=InventoryAgent-Store123` to get specific agent, or build more endpoints for different types of data). You might also join with other tables, e.g., an `agent_errors` table to get count of errors.

        The Supabase approach we're using in our implementation uses Supabase's REST interface under the hood. FastAPI can be the layer where you can implement any business logic or aggregation before sending to the front-end. 

        For instance, you might not want to ship raw data to the frontend. The FastAPI could compute some summaries: say, calculate a global tasks_completed total or percentage of agents meeting a certain threshold, etc., and return those in the JSON.

        The front-end (built with SvelteKit + Tailwind in this scenario) would call this API (via fetch in Svelte, perhaps using Supabase's client for real-time updates if we used that). It can then display in a nice UI table or graphs (maybe using a chart library). The ShadCN UI components could be used to style tables or cards showing each agent's metrics.

        Additionally, Supabase has a feature called *Realtime* that can stream changes from the database to clients via websockets. A more advanced dashboard might subscribe to changes on the `agent_metrics` table. For example, if an agent updates its row with new stats every minute, the frontend could get live updates without polling. Supabase's JS library can handle that easily with something like:
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ```javascript
        supabase.channel('public:agent_metrics')
          .on('postgres_changes', { event: '*', schema: 'public', table: 'agent_metrics' }, payload => {
            // Update the corresponding agent's metrics on the dashboard in real-time
          })
          .subscribe();
        ```
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Deployment Strategies and DevOps for Agents\index{deployment strategies}\index{DevOps!for agents}


        ### Code Example: CI/CD Pipeline Using GitHub and Vercel

        Let's illustrate a simple CI/CD pipeline with GitHub Actions for our agentic retail system. This pipeline will run tests and then deploy both the frontend (to Vercel) and backend (perhaps to Vercel or another server). We assume the frontend (SvelteKit) is connected to Vercel via Git integration, so it deploys automatically on pushes to `main`. For the backend (FastAPI + agent services), we'll use GitHub Actions to build and deploy to some environment – for demonstration, maybe deploying a Docker image to a registry or using Vercel's serverless functions if feasible.

        Below is a YAML snippet for GitHub Actions (placed in `.github/workflows/ci-cd.yml`):
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ```yaml
        name: CI-CD Pipeline

        on:
          push:
            branches: [main]
          pull_request:
            branches: [main]

        jobs:
          # Job 1: Run tests (CI)
          test-build:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v3
              - name: Setup Python
                uses: actions/setup-python@v4
                with:
                  python-version: 3.9
              - name: Install backend dependencies
                run: pip install -r backend/requirements.txt
              - name: Run backend tests
                run: pytest backend/tests
              - name: Install frontend dependencies
                run: npm ci --prefix frontend
              - name: Run frontend build (to catch compile errors)
                run: npm run build --prefix frontend
        ```
    """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ```yaml
          # Job 2: Deploy to Vercel (CD)
          deploy:
            needs: test-build
            runs-on: ubuntu-latest
            if: github.ref == 'refs/heads/main' && needs.test-build.result == 'success'
            steps:
              - uses: actions/checkout@v3
              # Assuming using Vercel CLI for backend or a custom deployment, for example:
              - name: Install Vercel CLI
                run: npm install -g vercel@latest
              - name: Build FastAPI container
                run: docker build -t myregistry/retail-backend:${{ github.sha }} -f backend/Dockerfile .
              - name: Push Container to Registry
                run: |
                  echo "${{ secrets.REGISTRY_PASSWORD }}" | docker login myregistry.io -u ${{ secrets.REGISTRY_USER }} --password-stdin
                  docker push myregistry/retail-backend:${{ github.sha }}
              - name: Deploy to Vercel (Frontend)
                uses: amondnet/vercel-action@v20
                with:
                  vercel-token: ${{ secrets.VERCEL_TOKEN }}
                  vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
                  vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
                  working-directory: frontend
                  alias-domains: "dashboard.mystore.com"
              - name: Deploy Backend to Server
                run: |
                  # Example: trigger a remote deploy script or update a Kubernetes deployment
                  ssh user@backend-server "docker pull myregistry/retail-backend:${{ github.sha }} && docker stop retail-backend && docker run -d --rm -p 80:80 myregistry/retail-backend:${{ github.sha }}"
        ```

        """
    )
    return



if __name__ == "__main__":
    app.run()
