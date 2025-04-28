

import marimo

__generated_with = "0.13.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Chapter 1: Introduction

        In this chapter, we explore what makes AI "agentic," transitioning from traditional methods to autonomous decision-making systems. 

        We'll discuss foundational concepts, the AI lifecycle, and the essential building blocks that position agentic AI as a transformative force in retail. 

        Readers will gain clarity on how proactive intelligence reshapes inventory management, pricing, and customer experiences, setting the stage for deeper exploration in subsequent chapters​.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Imports""")
    return


@app.cell
def _():
    # Import InventoryAgent from the agents module (modularized)
    from agents.inventory import InventoryAgent

    return (InventoryAgent,)


@app.cell
def _(mo):
    mo.md(r"""## Example: Simulation of agent in an environment loop""")
    return


@app.cell
def _(InventoryAgent):
    agent = InventoryAgent(reorder_threshold=50, max_capacity=100)
    environment = {"stock_level": 60}  # initial stock

    for day in range(1, 8):  # simulate a week of daily checks
        print(f"\nDay {day}: Stock level = {environment['stock_level']}")
        agent.perceive(environment)  # Agent observes the current stock level
        decision = agent.decide()  # Agent decides whether to reorder
        agent.act(decision)  # Agent takes action if needed
        # Simulate environment changes (e.g., daily sales reducing stock by random amount)
        sales = 15 if day == 3 else 5  # example: a big sale happened on day 3
        environment["stock_level"] = max(agent.current_stock - sales, 0)
        agent.learn(feedback=None)  # No learning implemented in this simple example
    # Define returned variables if needed, e.g.:
    # return var1, var2
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Explanation:** This agent checks a product's stock each day and autonomously decides to place a reorder when stock falls below a threshold. 

        After acting, it updates its internal stock state. (In a real scenario, learning could be implemented to adjust the reorder threshold or predict optimal order quantities over time.)
        """
    )
    return


if __name__ == "__main__":
    app.run()
