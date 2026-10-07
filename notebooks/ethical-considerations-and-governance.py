import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Ethical Considerations and Governance

    Explore essential ethical considerations and governance frameworks critical to responsible agentic AI deployment in retail. You'll understand transparency, accountability, human oversight, and regulatory compliance, ensuring that your AI initiatives align with societal values and legal standards​.

    ## Code Example: Human-in-the-Loop Approval Workflow

    Let's demonstrate how a human-in-the-loop approval process might be implemented in code. We will sketch a simple backend API (using Python with a FastAPI-like style) and a snippet of a frontend interface (perhaps using SvelteKit with a Supabase database) to handle an AI agent's decisions that require human approval. The scenario: an AI pricing agent proposes price changes, but if the change is above a certain threshold (e.g., more than 20% discount), it requires a human manager's approval.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Backend (Python/FastAPI)** – managing suggestions and approvals:
    """)  # Ensure mo is returned if other cells use it directly
    # return mo
    return


@app.cell
def _():
    from demos.hitl_approval_api_demo import app as approval_app

    return (approval_app,)


@app.cell
def _(mo):
    mo.md(r"""
    The backend uses `demos/hitl_approval_api_demo.py`, including requester/reviewer separation and a recorded outcome before returning an approved price. The identities here are synthetic; a deployed API must obtain them from authentication. Proposals include a `requester` field, and reviews require a distinct `reviewer`. The AI system would call `/ai/propose_price` whenever it has a price recommendation. The logic checks the size of the discount; if it's above 20%, instead of approving automatically, it stores the suggestion in a `pending_reviews` dictionary and returns a status that it's pending. A real system might push a notification to a review dashboard at this point. There are also endpoints for an admin (human) to list all pending reviews, approve them, or reject/modify them. This way, a human can fetch the list (perhaps via the frontend) and take actions.

    **Frontend (SvelteKit + Supabase)** – a simple UI for managers to review suggestions:
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ```svelte
    <script lang="ts">
      import { onMount } from 'svelte';
      let pending = [];

      // Fetch pending reviews on component mount
      onMount(async () => {
        const res = await fetch('/admin/pending_reviews');
        pending = (await res.json()).reviews;
      });

      // Approve a suggestion
      async function approve(reviewId: number) {
        const res = await fetch(`/admin/review/${reviewId}/approve?reviewer=pricing-lead`, { method: 'POST' });
        if (!res.ok) throw new Error('Approval failed');
        pending = pending.filter(item => item.review_id !== reviewId);
      }

      // Reject a suggestion (with optional new price)
      async function reject(reviewId: number, productId: number, alternativePrice: number | null = null) {
        const res = await fetch(`/admin/review/${reviewId}/reject?reviewer=pricing-lead`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ new_price: alternativePrice })
        });
        if (!res.ok) throw new Error('Review failed');
        pending = pending.filter(item => item.review_id !== reviewId);
      }
    </script>

    <h2>AI Price Change Suggestions Requiring Approval</h2>
    {#if pending.length === 0}
      <p>No pending reviews. AI suggestions are up-to-date.</p>
    {:else}
      <table>
        <tr><th>Product</th><th>Current Price</th><th>Suggested Price</th><th>Action</th></tr>
        {#each pending as review}
          <tr>
            <td>{review.product_id}</td>
            <td>${review.current_price}</td>
            <td>${review.suggested_price}</td>
            <td>
              <button on:click={() => approve(review.review_id)}>Approve</button>
              <button on:click={() => reject(review.review_id, review.product_id)}>Reject</button>
            </td>
          </tr>
        {/each}
      </table>
    {/if}
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    In this Svelte component, when the page loads (`onMount`), it fetches the pending reviews from our backend and stores the response's `reviews` list in a `pending` array. It then displays them in a table with product ID, current price, and suggested price. The manager can click **Approve** to call the approve API, or **Reject** to call the reject API (we also allow an optional flow to provide an alternative price – for brevity, we show a reject with or without suggesting an alternative; in a real UI, we'd provide an input to capture the new price). Once an action is taken, we update the `pending` list in the UI by removing that review.

    This simple example shows the scaffolding of a human-in-loop workflow:

    1. The AI defers certain decisions to humans based on rules (here, >20% discount).
    2. Those decisions are queued for human review.
    3. A human interface lists the queued decisions and allows one-click approval or modification.
    4. The system updates accordingly.

    In practice, this could be enhanced with real databases (Supabase could store the pending decisions so that multiple managers can view them in real-time and so that data persists), authentication (only authorized staff can access the `/admin` endpoints or UI), and notifications (e.g., send an email or Slack message when a new review is pending). Frontend libraries like ShadCN UI could style the table and buttons consistently with the company's design system. But the core logic remains: **the human is looped in before the AI's decision is finalized.**

    This approach ensures that for sensitive cases, human judgment is applied. It also serves as a feedback mechanism; if humans consistently approve some type of suggestion, the threshold might be adjusted to let AI auto-approve next time (or vice versa). Over time, the line of autonomy can shift as trust in the AI grows, but with this setup, that shift is controlled and observable.


    ### Code Example: Explainability Module for Pricing Decisions

    To illustrate explainability in practice, below is a simplified example of a Python module that explains a pricing agent's decisions. In this scenario, assume we have an AI model that suggests optimal prices for products based on features like inventory levels, competitor pricing, and days remaining in the season. We'll use the SHAP library\index{SHAP!implementation} to interpret a trained model's price prediction for a specific product. This could be part of a backend service (perhaps a FastAPI endpoint) that returns an explanation for why the AI suggested a certain price.
    """)
    return


@app.cell
def _():
    from runpy import run_module

    run_module("demos.pricing_explainability_shap_demo", run_name="__main__")
    return


if __name__ == "__main__":
    app.run()
