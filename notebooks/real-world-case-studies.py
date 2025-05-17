import marimo  # Ensure marimo is imported for app definition
import marimo as mo

from demos.dynamic_pricing_agent_demo import run_pricing_demo

# Import functions from the new demo scripts
from demos.inventory_management_agent_demo import run_inventory_demo
from demos.virtual_shopping_assistant_demo import run_assistant_demo

__generated_with = "0.1.69"
app = marimo.App()


@app.cell
def __():
    # Import necessary libraries
    import marimo as mo

    return (
        mo,
        run_assistant_demo,
        run_pricing_demo,
        run_inventory_demo,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        # Chapter 13: Real-World Case Studies

        This chapter examines successful implementations of agentic AI systems in retail environments. Through detailed case studies, we analyze how retailers have deployed autonomous agents to transform inventory management, pricing strategies, and customer engagement. By examining both successes and challenges from real-world deployments, this chapter provides practical insights on implementation approaches, technical architectures, and organizational considerations that lead to successful outcomes.

        The following sections contain interactive demos based on the examples discussed. Click the buttons to run the simulations.
        """
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Case Study 1: Inventory Management Agent

        Autonomous agents can monitor stock levels and automatically trigger reorders to prevent stockouts and optimize inventory holding. This example simulates an agent ensuring a minimum stock level for a product.

        *(Note: This demo uses a simplified, placeholder agent framework for illustration.)*
        """
    )


@app.cell
def __(mo, run_inventory_demo):
    # Button to trigger the inventory demo
    run_inventory_button = mo.ui.button(
        label="Run Inventory Agent Demo",
        on_click=lambda _: run_inventory_demo(),
        kind="success",
    )
    return (run_inventory_button,)


@app.cell
def __(run_inventory_button):
    # Display the button and its output value (which will update on click)
    mo.vstack([run_inventory_button, mo.md(f"**Demo Output:** {run_inventory_button.value}")])


@app.cell
def __(mo):
    mo.md(
        r"""
        **Explanation:** The inventory agent demo simulates checking the stock for `product_123` against a target level (50 units). If the stock (initially 20) is below the target, the agent uses a (simulated) tool to order the difference (30 units). The output confirms the order and the new stock level. In a real system, the agent's tools would interact with actual inventory and procurement systems.
        """
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Case Study 2: Dynamic Pricing Agent

        Pricing agents can adjust prices based on real-time factors like competitor pricing, demand signals, and inventory levels to maximize revenue or clear stock. This example simulates a pricing agent adjusting `product_456`'s price.

        *(Note: This demo uses a simplified, placeholder agent framework for illustration.)*
        """
    )


@app.cell
def __(mo, run_pricing_demo):
    # Button to trigger the pricing demo
    run_pricing_button = mo.ui.button(
        label="Run Dynamic Pricing Agent Demo",
        on_click=lambda _: run_pricing_demo(),
        kind="success",
    )
    return (run_pricing_button,)


@app.cell
def __(run_pricing_button):
    # Display the button and its output value
    mo.vstack([run_pricing_button, mo.md(f"**Demo Output:** {run_pricing_button.value}")])


@app.cell
def __(mo):
    mo.md(
        r"""
        **Explanation:** The pricing agent demo checks the competitor price ($120) and inventory level (5 units) for `product_456` (initially priced at $100). Based on its instructions (e.g., raise price if stock is low and competitor is higher), the agent decides on a new price (e.g., $110) and uses a (simulated) tool to update it. The output confirms the price change.
        """
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Case Study 3: Virtual Shopping Assistant

        Customer-facing agents, like chatbots or virtual assistants, can provide personalized recommendations, answer questions, and guide users through the shopping process, enhancing engagement and sales. This example uses OpenAI's API to simulate an assistant recommending an outfit.

        *(Note: This demo requires a configured OpenAI API key.)*
        """
    )


@app.cell
def __(mo, run_assistant_demo):
    # Input for user query
    user_query = mo.ui.text(label="Ask the assistant:", value="I need an outfit idea for a summer party.")

    # Button to trigger the assistant demo
    run_assistant_button = mo.ui.button(
        label="Run Virtual Assistant Demo",
        on_click=lambda _: run_assistant_demo(user_query.value),
        kind="success",
    )
    return run_assistant_button, user_query


@app.cell
def __(run_assistant_button, user_query):
    # Display the input, button, and output value
    mo.vstack(
        [
            user_query,
            run_assistant_button,
            mo.md(f"**Assistant Response:** {run_assistant_button.value}"),
        ]
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        **Explanation:** When you click the button, the user's query is sent to the OpenAI API. The AI model determines it needs to call the `recommend_outfit` function (defined in the demo script) with the extracted style ("summer party"). The demo script executes this function (which returns simulated outfit suggestions) and sends the results back to the AI. The AI then formulates a natural language response incorporating the suggestions, which is displayed above. This demonstrates how function calling enables AI assistants to leverage external tools and knowledge.
        """
    )


if __name__ == "__main__":
    app.run()
