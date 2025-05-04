import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell
def __(): # type: ignore[no-redef]
    import marimo as mo
    return (mo,)


@app.cell
def __(mo): # type: ignore[no-redef]
    mo.md(
        r"""
        # Ethical Considerations and Governance

        Explore essential ethical considerations and governance frameworks critical to responsible agentic AI deployment in retail. You'll understand transparency, accountability, human oversight, and regulatory compliance, ensuring that your AI initiatives align with societal values and legal standards​.

        ## Code Example: Human-in-the-Loop Approval Workflow

        Let's demonstrate how a human-in-the-loop approval process might be implemented in code. We will sketch a simple backend API (using Python with a FastAPI-like style) and a snippet of a frontend interface (perhaps using SvelteKit with a Supabase database) to handle an AI agent's decisions that require human approval. The scenario: an AI pricing agent proposes price changes, but if the change is above a certain threshold (e.g., more than 20% discount), it requires a human manager's approval.
    """
    )
    return


@app.cell
def __(mo): # type: ignore[no-redef]
    mo.md(
        r"""
        **Backend (Python/FastAPI)** – managing suggestions and approvals:
        """
    )
    # Ensure mo is returned if other cells use it directly
    # return mo


@app.cell
def __(FastAPI, Dict, mo): # type: ignore[no-redef]
    from fastapi import FastAPI
    from typing import Dict

    app = FastAPI()
    pending_reviews: Dict[
        int, dict
    ] = {}  # In-memory store for pending decisions (for demo)

    # Endpoint for AI to propose a price change
    @app.post("/ai/propose_price")
    def propose_price(product_id: int, current_price: float, suggested_price: float):
        change_percent = (current_price - suggested_price) / current_price * 100
        if change_percent > 20:  # >20% markdown, require human approval
            review_id = len(pending_reviews) + 1
            pending_reviews[review_id] = {
                "product_id": product_id,
                "current_price": current_price,
                "suggested_price": suggested_price,
                "reason": "High discount > 20%, pending approval",
            }
            return {
                "status": "pending",
                "review_id": review_id,
                "message": "Escalated for human approval",
            }
        else:
            # Auto-approve minor price changes
            # (In a real system, code to update the price in database would go here)
            return {"status": "auto_approved", "new_price": suggested_price}

    # Endpoint for a human manager to get the list of pending price changes
    @app.get("/admin/pending_reviews")
    def list_pending():
        return pending_reviews

    # Endpoint for a human to approve a pending price change
    @app.post("/admin/review/{review_id}/approve")
    def approve_price(review_id: int):
        review = pending_reviews.pop(review_id, None)
        if not review:
            return {"error": "Review not found or already processed"}
        # Here we would apply the price change, e.g., update product price in database
        return {
            "status": "approved",
            "product_id": review["product_id"],
            "new_price": review["suggested_price"],
        }

    # Endpoint for a human to reject/modify a pending price change
    @app.post("/admin/review/{review_id}/reject")
    def reject_price(review_id: int, new_price: float = None):
        review = pending_reviews.pop(review_id, None)
        if not review:
            return {"error": "Review not found or already processed"}
        action = {}
        if new_price:
            # Human provided an alternative price
            action = {
                "status": "modified",
                "product_id": review["product_id"],
                "new_price": new_price,
            }
            # Update price to new_price in database (not shown)
        else:
            # Human outright rejected the suggestion
            action = {
                "status": "rejected",
                "product_id": review["product_id"],
                "reason": "Human rejected AI suggestion",
            }
        return action

    # Define returned variables if needed, e.g.:
    # return app, pending_reviews # if used by other cells
    # It seems these are self-contained examples, so likely no return needed
    # Ensure FastAPI and Dict are passed if defined earlier, or import locally if needed
    # Assuming they are imported in the first cell as per analysis
    return


@app.cell
def __(mo): # type: ignore[no-redef]
    mo.md(
        r"""
        In this backend code, the AI system would call `/ai/propose_price` whenever it has a price recommendation. The logic checks the size of the discount; if it's above 20%, instead of approving automatically, it stores the suggestion in a `pending_reviews` dictionary and returns a status that it's pending. A real system might push a notification to a review dashboard at this point. There are also endpoints for an admin (human) to list all pending reviews, approve them, or reject/modify them. This way, a human can fetch the list (perhaps via the frontend) and take actions.

        **Frontend (SvelteKit + Supabase)** – a simple UI for managers to review suggestions:
        """
    )
    return


@app.cell
def __(mo): # type: ignore[no-redef]
    mo.md(
        r"""
        ```svelte
        <script lang="ts">
          import { onMount } from 'svelte';
          let pending = [];

          // Fetch pending reviews on component mount
          onMount(async () => {
            const res = await fetch('/admin/pending_reviews');
            pending = await res.json();
          });

          // Approve a suggestion
          async function approve(reviewId: number) {
            await fetch(`/admin/review/${reviewId}/approve`, { method: 'POST' });
            pending = pending.filter(item => item[0] !== reviewId);
          }

          // Reject a suggestion (with optional new price)
          async function reject(reviewId: number, productId: number, alternativePrice: number | null = null) {
            const url = alternativePrice
              ? `/admin/review/${reviewId}/reject?new_price=${alternativePrice}`
              : `/admin/review/${reviewId}/reject`;
            await fetch(url, { method: 'POST' });
            pending = pending.filter(item => item[0] !== reviewId);
          }
        </script>

        <h2>AI Price Change Suggestions Requiring Approval</h2>
        {#if pending.length === 0}
          <p>No pending reviews. AI suggestions are up-to-date.</p>
        {:else}
          <table>
            <tr><th>Product</th><th>Current Price</th><th>Suggested Price</th><th>Action</th></tr>
            {#each Object.entries(pending) as [id, review]}
              <tr>
                <td>{review.product_id}</td>
                <td>${review.current_price}</td>
                <td>${review.suggested_price}</td>
                <td>
                  <button on:click={() => approve(Number(id))}>Approve</button>
                  <button on:click={() => reject(Number(id), review.product_id)}>Reject</button>
                </td>
              </tr>
            {/each}
          </table>
        {/if}
        ```

    """
    )
    return


@app.cell
def __(mo): # type: ignore[no-redef]
    mo.md(
        r"""
        In this Svelte component, when the page loads (`onMount`), it fetches the pending reviews from our backend and stores them in a `pending` array. It then displays them in a table with product ID, current price, and suggested price. The manager can click **Approve** to call the approve API, or **Reject** to call the reject API (we also allow an optional flow to provide an alternative price – for brevity, we show a reject with or without suggesting an alternative; in a real UI, we'd provide an input to capture the new price). Once an action is taken, we update the `pending` list in the UI by removing that review.

        This simple example shows the scaffolding of a human-in-loop workflow:
        
        1. The AI defers certain decisions to humans based on rules (here, >20% discount).
        2. Those decisions are queued for human review.
        3. A human interface lists the queued decisions and allows one-click approval or modification.
        4. The system updates accordingly.

        In practice, this could be enhanced with real databases (Supabase could store the pending decisions so that multiple managers can view them in real-time and so that data persists), authentication (only authorized staff can access the `/admin` endpoints or UI), and notifications (e.g., send an email or Slack message when a new review is pending). Frontend libraries like ShadCN UI could style the table and buttons consistently with the company's design system. But the core logic remains: **the human is looped in before the AI's decision is finalized.**

        This approach ensures that for sensitive cases, human judgment is applied. It also serves as a feedback mechanism; if humans consistently approve some type of suggestion, the threshold might be adjusted to let AI auto-approve next time (or vice versa). Over time, the line of autonomy can shift as trust in the AI grows, but with this setup, that shift is controlled and observable.


        ### Code Example: Explainability Module for Pricing Decisions

        To illustrate explainability in practice, below is a simplified example of a Python module that explains a pricing agent's decisions. In this scenario, assume we have an AI model that suggests optimal prices for products based on features like inventory levels, competitor pricing, and days remaining in the season. We'll use the SHAP library\index{SHAP!implementation} to interpret a trained model's price prediction for a specific product. This could be part of a backend service (perhaps a FastAPI endpoint) that returns an explanation for why the AI suggested a certain price.
        """
    )
    return


@app.cell
def __(mo): # type: ignore[no-redef]
    import pandas as pd
    import numpy as np
    import shap
    from sklearn.ensemble import RandomForestRegressor

    # Sample training data for a pricing model (for illustration purposes)
    data = pd.DataFrame(
        {
            "inventory_level": [200, 50, 120, 80, 300],  # units in stock
            "competitor_price": [50, 45, 60, 55, 40],  # competitor's price in $
            "days_to_season_end": [10, 5, 30, 20, 15],  # days until end-of-season
        }
    )
    target = np.array(
        [45, 40, 60, 50, 35]
    )  # historical optimal prices for those scenarios

    # Train a simple model (Random Forest) to predict optimal price
    model = RandomForestRegressor(random_state=0).fit(data, target)

    # Suppose the agent suggests a new price for a product with the following features:
    product = pd.DataFrame(
        {
            "inventory_level": [150],  # current stock
            "competitor_price": [48],  # competitor's price for similar item
            "days_to_season_end": [7],  # days left in season
        }
    )

    predicted_price = model.predict(product)[0]
    print(f"AI-predicted optimal price: ${predicted_price:.2f}")

    # Use SHAP to explain the prediction
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(product)

    # Pair each feature with its SHAP contribution value
    explanation = {}
    for feature_name, value, shap_val in zip(
        product.columns, product.iloc[0], shap_values[0]
    ):
        explanation[feature_name] = round(shap_val, 2)
        print(f"  {feature_name}: {value} -> contribution {shap_val:+.2f}")

    # The explanation dict now holds feature contributions to the price prediction.
    # Define returned variables if needed, e.g.:
    # return var1, var2
    return explanation # Return the explanation dict for potential use



if __name__ == "__main__":
    app.run()
