import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    """
    This cell holds:
    - Core imports for the notebook
    - Logging setup
    - Return frequently used objects so other cells can receive them
    """
    import logging
    import random
    from datetime import datetime, timedelta

    import marimo as mo

    from agents.bdi import InventoryBDIAgent
    from agents.ooda import OODAPricingAgent

    # Configure logging once at the start
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("AgentFrameworks")  # Main logger for the notebook

    # Return all modules/classes/objects needed in subsequent cells
    return (
        InventoryBDIAgent,
        OODAPricingAgent,
        datetime,
        logger,
        mo,
        random,
        timedelta,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Agent Architectures and Frameworks

        Overview of BDI and OODA agent models for retail AI, including BDI agent structure and applications in retail.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""BDI Agent Data Models""")
    return


@app.cell
def _(mo):
    # --- BDI Agent Class ---
    mo.md(
        """
    **InventoryBDIAgent**

    Belief-Desire-Intention agent for inventory management. Encapsulates the BDI cycle for retail decision-making. Imported from `agents.bdi`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""BDI Agent Simulation Controls""")
    return


@app.cell
def _(mo):
    mo.md("""OODA Agent Data Models""")
    return


@app.cell
def _(mo):
    # --- OODA Agent Class ---
    mo.md(
        """
    **OODAPricingAgent**

    OODA (Observe-Orient-Decide-Act) agent for dynamic pricing. Implements the OODA loop for retail pricing. Imported from `agents.ooda`.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""OODA Agent Simulation""")
    return


@app.cell
def _(mo):
    mo.md("""## Summary and Next Steps""")
    return


@app.cell
def _(mo):
    mo.md(
        """
        # Agent Architectures and Frameworks

        This chapter explores two key agent architectures—Belief-Desire-Intention (BDI) and Observe-Orient-Decide-Act (OODA)—and their applications in retail AI. You'll learn how BDI agents manage inventory and how OODA agents handle dynamic pricing, gaining insights into the practical implementation of these models.
        """
    )
    return


@app.cell
def _(mo):
    # --- Data Models for BDI Agent ---
    mo.md(
        """
    **ProductInfo, InventoryItem, SalesData**

    Core data models for inventory, product, and sales information. Used as beliefs in the BDI agent. Imported from `models.inventory`.
    """
    )
    from models.inventory import InventoryItem, ProductInfo, SalesData

    return InventoryItem, ProductInfo, SalesData


@app.cell
def _(mo):
    mo.md(
        r"""
        **Explanation**:

        1. **`ProductInfo`**: Holds critical product specifications like price, cost, lead time, and supplier details. Includes `current_price` which might change dynamically.
        2. **`InventoryItem`**: Tracks the physical stock levels, reorder thresholds, and any pending incoming orders for each product.
        3. **`SalesData`**: Stores recent sales history and provides methods to calculate `average_daily_sales` and `trend`.

        These classes form the agent's **"Beliefs"**—its internal representation of the current state of the retail environment.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Part B: BDI Agent Class
        Defines the main `InventoryBDIAgent` class, encapsulating its beliefs, desires, intentions, and the core BDI cycle logic.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Explanation**

        The `InventoryBDIAgent` class structure:

        - **`__init__`**: Initializes beliefs (product, inventory, sales data), desires (goals with weights), and an empty list for intentions.
        - **`update_beliefs`**: Method to load new data into the agent's belief state.
        - **`observe` / `orient`**: (Code not shown here, but would be similar to the OODA example or previous BDI fragments) Gathers data and analyzes the situation for specific products or the overall market. These steps inform the deliberation.
        - **`deliberate`**: Core BDI logic. Evaluates the current state (using beliefs) against the weighted goals (desires) to determine which goals are most important *right now*. Uses helper methods (`_evaluate_*`) to calculate urgency for each goal. Returns a prioritized list of goal names.
        - **`generate_intentions`**: Takes the prioritized goals and creates concrete action plans (intentions). Uses helper methods (`_plan_*`) to generate specific actions like reordering or discounting. Stores these plans in `self.active_intentions`.
        - **`execute_intentions`**: Takes the generated intentions and carries them out. Uses helper methods (`_execute_*`) to simulate the actual interaction with external systems (like placing an order or changing a price).
        - **`run_cycle`**: Orchestrates the full Deliberate -> Generate Intentions -> Execute Intentions sequence.
        - **Helper Methods**: Internal methods (`_evaluate_*`, `_plan_*`, `_execute_*`, `_fetch_*`) encapsulate specific logic for clarity and reusability.
        """
    )
    return


@app.cell
def _(mo):
    # --- UI Controls ---
    stockout_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=1.0, label="Minimize Stockouts")
    excess_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=0.7, label="Minimize Excess Inventory")
    profit_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5, label="Maximize Profit Margin")
    fresh_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=0.8, label="Ensure Fresh Products")

    apples_stock = mo.ui.number(0, 200, value=25, label="Apples: Initial Stock")
    apples_reorder = mo.ui.number(0, 200, value=20, label="Apples: Reorder Point")
    apples_optimal = mo.ui.number(0, 200, value=50, label="Apples: Optimal Stock")
    apples_sales = mo.ui.text(
        label="Apples: 14-Day Sales (comma-separated)",
        value="8,7,9,8,10,12,9,8,7,6,8,9,10,11",
    )

    bread_stock = mo.ui.number(0, 200, value=5, label="Bread: Initial Stock")
    bread_reorder = mo.ui.number(0, 200, value=10, label="Bread: Reorder Point")
    bread_optimal = mo.ui.number(0, 200, value=30, label="Bread: Optimal Stock")
    bread_sales = mo.ui.text(
        label="Bread: 14-Day Sales (comma-separated)",
        value="6,5,7,8,6,5,4,6,7,8,6,5,4,5",
    )

    sim_days = mo.ui.slider(1, 10, value=3, label="Number of Simulation Days")
    run_button = mo.ui.run_button(label="Run BDI Simulation")

    controls = mo.vstack(
        [
            mo.md("### BDI Agent Simulation Controls"),
            stockout_slider,
            excess_slider,
            profit_slider,
            fresh_slider,
            mo.md("---"),
            apples_stock,
            apples_reorder,
            apples_optimal,
            apples_sales,
            bread_stock,
            bread_reorder,
            bread_optimal,
            bread_sales,
            mo.md("---"),
            sim_days,
            run_button,
        ]
    )
    controls

    return (
        apples_optimal,
        apples_reorder,
        apples_sales,
        apples_stock,
        bread_optimal,
        bread_reorder,
        bread_sales,
        bread_stock,
        excess_slider,
        fresh_slider,
        profit_slider,
        run_button,
        sim_days,
        stockout_slider,
    )


@app.cell
def _(mo, run_button):
    # --- BDI Agent Simulation Logic (Separate Cell) ---
    mo.stop(not run_button.value, "Click 'Run BDI Simulation' to start.")
    return


@app.cell
def _(
    InventoryBDIAgent,
    InventoryItem,
    ProductInfo,
    SalesData,
    apples_optimal,
    apples_reorder,
    apples_sales,
    apples_stock,
    bread_optimal,
    bread_reorder,
    bread_sales,
    bread_stock,
    datetime,
    excess_slider,
    fresh_slider,
    mo,
    profit_slider,
    random,
    sim_days,
    stockout_slider,
    timedelta,
):
    def parse_sales(s: str) -> list[int]:
        try:
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        except Exception:
            return [1] * 14

    try:
        # Setup agent with user parameters
        agent = InventoryBDIAgent()
        agent.goals = {
            "minimize_stockouts": stockout_slider.value,
            "minimize_excess_inventory": excess_slider.value,
            "maximize_profit_margin": profit_slider.value,
            "ensure_fresh_products": fresh_slider.value,
        }
        products_data = {
            "P001": ProductInfo(
                product_id="P001",
                name="Organic Apples",
                category="Produce",
                price=2.99,
                cost=1.50,
                lead_time_days=2,
                shelf_life_days=14,
                supplier_id="S1",
                min_order_quantity=10,
            ),
            "P002": ProductInfo(
                product_id="P002",
                name="Whole Grain Bread",
                category="Bakery",
                price=3.49,
                cost=1.25,
                lead_time_days=1,
                shelf_life_days=5,
                supplier_id="S2",
                min_order_quantity=20,
            ),
        }
        inventory_data = {
            "P001": InventoryItem(
                product_id="P001",
                current_stock=apples_stock.value,
                reorder_point=apples_reorder.value,
                optimal_stock=apples_optimal.value,
            ),
            "P002": InventoryItem(
                product_id="P002",
                current_stock=bread_stock.value,
                reorder_point=bread_reorder.value,
                optimal_stock=bread_optimal.value,
            ),
        }
        sales_data_hist = {
            "P001": SalesData(
                product_id="P001",
                daily_sales=parse_sales(apples_sales.value),
            ),
            "P002": SalesData(
                product_id="P002",
                daily_sales=parse_sales(bread_sales.value),
            ),
        }
        initial_date = datetime(2023, 10, 26)
        agent.update_beliefs(
            new_products=products_data,
            new_inventory=inventory_data,
            new_sales=sales_data_hist,
            new_date=initial_date,
        )
        output = []
        for day in range(sim_days.value):
            current_sim_date = agent.current_date
            output.append(mo.md(f"**Day {day + 1} ({current_sim_date.strftime('%Y-%m-%d')})**"))
            # Show inventory before
            inv_status = []
            for pid, item in sorted(agent.inventory.items()):
                inv_status.append(f"{products_data[pid].name}: Stock={item.current_stock}, Pending={item.pending_order_quantity}")
            output.append(mo.md("  ".join(inv_status)))
            # Run agent cycle
            actions = agent.run_cycle()
            if not actions:
                output.append(mo.md("No actions executed."))
            else:
                for i, act in enumerate(actions, 1):
                    output.append(mo.md(f"{i}. {act}"))
            # Simulate next day
            if day < sim_days.value - 1:
                new_inv = {}
                new_sales = agent.sales_data.copy()
                for pid, it in agent.inventory.items():
                    # Deliver order if due
                    if it.pending_order_quantity > 0 and it.expected_delivery_date and it.expected_delivery_date.date() == current_sim_date.date():
                        delivered = it.pending_order_quantity
                        it.current_stock += delivered
                        it.pending_order_quantity = 0
                        it.expected_delivery_date = None
                    # Simulate a random daily sale
                    sale_amount = random.randint(0, 3)
                    new_stock = max(it.current_stock - sale_amount, 0)
                    new_inv[pid] = InventoryItem(
                        product_id=pid,
                        current_stock=new_stock,
                        reorder_point=it.reorder_point,
                        optimal_stock=it.optimal_stock,
                        last_reorder_date=it.last_reorder_date,
                        expected_delivery_date=it.expected_delivery_date,
                        pending_order_quantity=it.pending_order_quantity,
                    )
                    # Update sales
                    old_sales = agent.sales_data[pid].daily_sales
                    new_sales_daily = old_sales[1:] + [sale_amount]
                    new_sales[pid] = SalesData(pid, new_sales_daily)
                agent.update_beliefs(
                    new_inventory=new_inv,
                    new_sales=new_sales,
                    new_date=current_sim_date + timedelta(days=1),
                )
        result = mo.vstack(output)
    except Exception as e:
        result = mo.md(f"**Error:** {e}")
    result
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        #### Part C: Demonstration Function
        Sets up a sample scenario with products, inventory, and sales data, then runs the BDI agent through a simulated multi-day period.
        """
    )
    return


@app.cell
def _(
    InventoryBDIAgent,
    InventoryItem,
    ProductInfo,
    SalesData,
    datetime,
    logger,
    random,
    timedelta,
):
    """
    Demonstration function for BDI agent.
    """

    def demonstrate_bdi_agent():
        agent = InventoryBDIAgent()

        # Setup sample products, inventory, sales
        products_data = {
            "P001": ProductInfo(
                product_id="P001",
                name="Organic Apples",
                category="Produce",
                price=2.99,
                cost=1.50,
                lead_time_days=2,
                shelf_life_days=14,
                supplier_id="S1",
                min_order_quantity=10,
            ),
            "P002": ProductInfo(
                product_id="P002",
                name="Whole Grain Bread",
                category="Bakery",
                price=3.49,
                cost=1.25,
                lead_time_days=1,
                shelf_life_days=5,
                supplier_id="S2",
                min_order_quantity=20,
            ),
        }
        inventory_data = {
            "P001": InventoryItem(
                product_id="P001",
                current_stock=25,
                reorder_point=20,
                optimal_stock=50,
            ),
            "P002": InventoryItem(
                product_id="P002",
                current_stock=5,
                reorder_point=10,
                optimal_stock=30,
            ),
        }
        sales_data_hist = {
            "P001": SalesData(
                product_id="P001",
                daily_sales=[8, 7, 9, 8, 10, 12, 9, 8, 7, 6, 8, 9, 10, 11],
            ),
            "P002": SalesData(
                product_id="P002",
                daily_sales=[6, 5, 7, 8, 6, 5, 4, 6, 7, 8, 6, 5, 4, 5],
            ),
        }

        initial_date = datetime(2023, 10, 26)
        agent.update_beliefs(
            new_products=products_data,
            new_inventory=inventory_data,
            new_sales=sales_data_hist,
            new_date=initial_date,
        )

        logger.info("\n===== BDI Agent Demonstration Start =====")

        for day in range(3):
            current_sim_date = agent.current_date
            logger.info(f"\n--- Day {day + 1} ({current_sim_date.strftime('%Y-%m-%d')}) ---")
            print(f"\n[Day {day + 1} - {current_sim_date.strftime('%Y-%m-%d')}] BEFORE Cycle:")
            for pid, item in sorted(agent.inventory.items()):
                print(
                    f"  {pid} Stock={item.current_stock:<3} Pending={item.pending_order_quantity:<3}"
                    f" (Optimal={item.optimal_stock}, Reorder={item.reorder_point})"
                )

            # Run the agent cycle
            print(f"\n[Day {day + 1}] Running BDI Cycle...")
            actions = agent.run_cycle()

            if not actions:
                print("No actions executed.")
            else:
                print("[Actions Executed]")
                for i, act in enumerate(actions, 1):
                    print(f"  {i}. {act}")

            # Simulate next day
            if day < 2:
                new_inv = {}
                new_sales = agent.sales_data.copy()
                for pid, it in agent.inventory.items():
                    # Maybe we deliver an order
                    if it.pending_order_quantity > 0 and it.expected_delivery_date and it.expected_delivery_date.date() == current_sim_date.date():
                        delivered = it.pending_order_quantity
                        it.current_stock += delivered
                        it.pending_order_quantity = 0
                        it.expected_delivery_date = None

                    # simulate a random daily sale
                    sale_amount = random.randint(0, 3)
                    new_stock = max(it.current_stock - sale_amount, 0)
                    new_inv[pid] = InventoryItem(
                        product_id=pid,
                        current_stock=new_stock,
                        reorder_point=it.reorder_point,
                        optimal_stock=it.optimal_stock,
                        last_reorder_date=it.last_reorder_date,
                        expected_delivery_date=it.expected_delivery_date,
                        pending_order_quantity=it.pending_order_quantity,
                    )
                    # Update sales
                    old_sales = agent.sales_data[pid].daily_sales
                    new_sales_daily = old_sales[1:] + [sale_amount]
                    new_sales[pid] = SalesData(pid, new_sales_daily)

                # Advance date
                agent.update_beliefs(
                    new_inventory=new_inv,
                    new_sales=new_sales,
                    new_date=current_sim_date + timedelta(days=1),
                )

        logger.info("\n===== BDI Agent Demonstration End =====")

    # Just call the function to run demonstration in one go
    demonstrate_bdi_agent()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Explanation**

                - This **demonstration** function serves as a miniature simulation:                
                  - **Initial Setup**: We define products (apples, bread, coffee, cheddar), their initial inventory levels (some low, some high), and 30 days of sales history.
                  - **Simulation Loop**: We run the agent for 3 simulated "days".
                  - **Inside the Loop**:
                    - Print the inventory status **before** the agent acts.
                    - Call `agent.run_cycle()`. The agent internally performs deliberation, generates intentions, and executes them.
                    - Print the actions the agent decided to take (reorder, discount, etc.).
                    - Simulate the environment changing: check for deliveries, simulate random daily sales based on history, update inventory, and advance the date.
                    - Update the agent's beliefs with the new state for the next day's cycle.

                This illustrates how the agent adapts its behavior day-to-day based on changing stock levels, sales trends (implicitly updated), and deliveries.

                ### BDI Summary

                This example highlights how a **BDI agent** can manage inventory autonomously. By modeling **Beliefs** (data classes), **Desires** (weighted goals evaluated in `deliberate`), and **Intentions** (action plans created in `generate_intentions` and executed via `execute_intentions`), we build a system that reacts rationally to its environment. The agent continuously updates its knowledge, prioritizes conflicting objectives (like avoiding stockouts vs. reducing excess), and takes actions aligned with retail strategy. Real-world integration would replace simulated data fetching and action execution with calls to actual databases and APIs.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## OODA: Observe-Orient-Decide-Act

        The OODA (Observe-Orient-Decide-Act) loop is a decision-making framework that emphasizes rapid, iterative cycles of information processing and action. This section introduces the OODA loop and its application to retail agent decision-making, particularly in dynamic pricing.
        """
    )
    return


@app.cell
def _(mo):
    # --- Data Model for OODA Agent ---
    mo.md(
        """
    **PricingProduct**

    Data model for products used in dynamic pricing. Imported from `models.pricing`.
    """
    )
    from models.pricing import PricingProduct

    return (PricingProduct,)


@app.cell
def _(mo):
    mo.md(
        """
        ## OODA Pricing Agent: Class and Demonstration

        Overview of the OODA pricing agent class and its demonstration for dynamic retail pricing.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Explanation**

        - **`__init__`**: Sets up weighting factors (how much inventory, competitors, sales influence price) and a `max_price_change_pct` to prevent excessive volatility.
        - **`observe`**: Gathers current data (competitor prices, inventory, sales) for a product (simulated here).
        - **`orient`**: Analyzes the observed data to classify the situation (e.g., price position, inventory status, sales assessment) and determine the overall `market_situation`.
        - **`decide`**: Calculates the desired price change based on the `orientation`. It computes components for inventory, competitor price difference, and sales velocity, combines them using the weights, caps the change percentage, and calculates the potential `new_price`, ensuring it stays within the product's min/max bounds. It also applies psychological pricing (like ending in .99).
        - **`act`**: Implements the `new_price` from the `decision` phase. It checks if the change is significant, simulates calling an external pricing API, updates the agent's internal belief (product's `current_price`), and logs the action.
        - **`run_cycle_for_product`**: Orchestrates one full OODA loop for a specific product.
        - **Helpers**: `_fetch_*` methods simulate data retrieval, and `_apply_price_psychology` adjusts the final price.

        #### Part C: OODA Demonstration

        Sets up a scenario with a few products and runs the OODA pricing agent through several cycles to show dynamic adjustments.
        """
    )
    return


@app.cell
def _(OODAPricingAgent, PricingProduct, logger, random):
    """
    Demonstration function for the OODAPricingAgent
    """

    def demonstrate_ooda_pricing():
        logger.info("\n===== OODA Pricing Agent Demonstration Start =====")

        agent = OODAPricingAgent(
            inventory_weight=0.2,
            competitor_weight=0.5,
            sales_weight=0.3,
            max_price_change_pct=7.0,
        )

        products = {
            "DYN-01": PricingProduct(
                product_id="DYN-01",
                name="Wireless Mouse",
                category="Electronics",
                cost=12.50,
                current_price=24.99,
                min_price=19.99,
                max_price=34.99,
                inventory=60,
                target_profit_margin=0.4,
                sales_last_7_days=[5, 6, 5, 7, 8, 6, 7],
            ),
            "DYN-02": PricingProduct(
                product_id="DYN-02",
                name="USB-C Hub",
                category="Accessories",
                cost=25.00,
                current_price=49.99,
                min_price=39.99,
                max_price=69.99,
                inventory=8,
                target_profit_margin=0.5,
                sales_last_7_days=[10, 12, 15, 11, 13, 14, 16],
            ),
            "DYN-03": PricingProduct(
                product_id="DYN-03",
                name="Gaming Keyboard",
                category="Electronics",
                cost=75.00,
                current_price=129.99,
                min_price=99.99,
                max_price=179.99,
                inventory=40,
                target_profit_margin=0.35,
                sales_last_7_days=[1, 0, 2, 1, 0, 1, 2],
            ),
        }
        agent.update_products(products)

        num_cycles = 5
        product_ids = list(products.keys())

        for cycle in range(num_cycles):
            print(f"\n--- OODA Cycle {cycle + 1} ---")
            for pid in product_ids:
                agent.run_cycle_for_product(pid)
                # Simulate small changes
                if agent.products[pid].inventory > 0:
                    agent.products[pid].inventory -= random.randint(0, 1)
                if agent.products[pid].sales_last_7_days:
                    last_sale = agent.products[pid].sales_last_7_days[-1]
                    new_sale = max(0, last_sale + random.randint(-1, 1))
                    new_hist = agent.products[pid].sales_last_7_days[1:] + [new_sale]
                    agent.products[pid].sales_last_7_days = new_hist

            print(f"\nState after Cycle {cycle + 1}:")
            for pid, prod in sorted(agent.products.items()):
                print(f"  {prod.name:<18} [{pid}] Price=${prod.current_price:.2f} Inv={prod.inventory}")

        logger.info("\n===== OODA Pricing Agent Demonstration End =====")

    demonstrate_ooda_pricing()
    return


if __name__ == "__main__":
    app.run()
