import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import logging

    # Configure logging once at the start
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("AgentFrameworks")  # Main logger for the notebook

    return logger, mo


@app.cell
def _(mo):
    mo.md(r"""# Decision-making Frameworks for Retail Agents""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Chapter 3: Decision-Making Frameworks - Statistical & Causal

        ### BayesianRecommendationAgent
        See the agent's code in `agents/bayesian.py` for full implementation details.
        """
    )
    return


@app.cell
def _():
    from demos.recommendation import demo_bayesian_recommendations

    return (demo_bayesian_recommendations,)


@app.cell
def _(demo_bayesian_recommendations):
    # Example usage
    def demonstrate_bayesian_recommendations(mo_instance):
        recommendations_c1, agent, all_products, product_catalog = (
            demo_bayesian_recommendations()
        )

        output_c1 = [mo_instance.md("**Top 5 recommendations for customer C1:**")]
        for i, pid in enumerate(recommendations_c1):
            explain = agent.explain_recommendation("C1", pid)
            prod_name = product_catalog[pid]["name"]
            reason = explain["explanation"]
            output_c1.append(mo_instance.md(f"  {i + 1}. `{prod_name}` -> {reason}"))

        # Generate recommendations for new customer C2
        print("\nGenerating recommendations for new customer C2...")
        recommendations_c2 = agent.recommend("C2", all_products, num_recommendations=5)

        output_c2 = [mo_instance.md("**Top 5 recommendations for new customer C2:**")]
        for i, pid in enumerate(recommendations_c2):
            explain = agent.explain_recommendation("C2", pid)
            prod_name = product_catalog[pid]["name"]
            reason = explain["explanation"]
            output_c2.append(mo_instance.md(f"  {i + 1}. `{prod_name}` -> {reason}"))

        # Visualize preferences for C1
        print("\nVisualizing C1's preference distributions...")
        fig_c1 = agent.visualize_customer_preferences("C1")

        # Combine outputs for Marimo
        return mo_instance.vstack(
            [
                *output_c1,
                mo_instance.md("---"),
                *output_c2,
                mo_instance.md("### Customer C1 Preference Distributions"),
                (
                    fig_c1
                    if fig_c1
                    else mo_instance.md("_No interactions to visualize for C1._")
                ),
            ]
        )

    return (demonstrate_bayesian_recommendations,)


@app.cell
def _(demonstrate_bayesian_recommendations, mo):
    # Call the demonstration function and display its output
    demonstration_output = demonstrate_bayesian_recommendations(mo)
    return (demonstration_output,)


@app.cell
def _(demonstration_output):
    demonstration_output
    return


@app.cell
def _(mo):
    mo.md(r"""## Chapter 4: Decision-Making Frameworks - Sequential""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### MDP/Q-Learning Dynamic Pricing Demo

        Configure the environment and agent hyperparameters below, then click 
        **Run Simulation** to train a Q-learning agent for dynamic pricing.
        Results will include a learning curve and a sample of the learned policy.

        - For `DynamicPricingMDP`:
            ```python
                from environments.mdp import DynamicPricingMDP
            ```
        - For `QLearningAgent` :
            ```python
                from agents.qlearning import QLearningAgent
            ```
        - For Configurations:
            ```python
              from config.config import DynamicPricingMDPConfig, QLearningAgentConfig
            ```
        - For Demo (`demonstrate_mdp_dynamic_pricing`):
            ```python
                from demos.dynamic_pricing import demonstrate_mdp_dynamic_pricing
            ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Environment hyperparameters""")
    return


@app.cell
def _(mo):
    initial_inventory = mo.ui.slider(10, 500, value=100, label="Initial Inventory")
    season_length_weeks = mo.ui.slider(4, 52, value=12, label="Season Length (weeks)")
    base_price = mo.ui.slider(10, 200, value=50, step=1, label="Base Price ($)")
    base_demand = mo.ui.slider(1, 50, value=10, step=1, label="Base Demand")
    price_elasticity = mo.ui.slider(
        0.1, 5.0, value=1.5, step=0.1, label="Price Elasticity"
    )
    holding_cost_per_unit = mo.ui.slider(
        0.0, 5.0, value=0.5, step=0.1, label="Holding Cost per Unit"
    )
    end_season_salvage_value = mo.ui.slider(
        0.0, 100.0, value=15.0, step=1, label="End-of-Season Salvage Value"
    )
    available_discounts = mo.ui.multiselect(
        options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        value=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        label="Available Discounts (%)",
    )

    # Agent hyperparameters
    learning_rate = mo.ui.slider(
        0.01, 1.0, value=0.1, step=0.01, label="Learning Rate (alpha)"
    )
    discount_factor = mo.ui.slider(
        0.5, 1.0, value=0.95, step=0.01, label="Discount Factor (gamma)"
    )
    exploration_rate = mo.ui.slider(
        0.0, 1.0, value=1.0, step=0.01, label="Initial Exploration Rate (epsilon)"
    )
    exploration_decay = mo.ui.slider(
        0.90, 1.0, value=0.995, step=0.001, label="Exploration Decay"
    )
    min_exploration_rate = mo.ui.slider(
        0.0, 0.5, value=0.01, step=0.01, label="Min Exploration Rate"
    )
    num_training_episodes = mo.ui.slider(
        100, 20000, value=5000, step=100, label="Training Episodes"
    )
    run_button = mo.ui.run_button(label="Run Simulation")

    return (
        available_discounts,
        base_demand,
        base_price,
        discount_factor,
        end_season_salvage_value,
        exploration_decay,
        exploration_rate,
        holding_cost_per_unit,
        initial_inventory,
        learning_rate,
        min_exploration_rate,
        num_training_episodes,
        price_elasticity,
        run_button,
        season_length_weeks,
    )


@app.cell
def _(
    available_discounts,
    base_demand,
    base_price,
    discount_factor,
    end_season_salvage_value,
    exploration_decay,
    exploration_rate,
    holding_cost_per_unit,
    initial_inventory,
    learning_rate,
    min_exploration_rate,
    mo,
    num_training_episodes,
    price_elasticity,
    run_button,
    season_length_weeks,
):
    mo.md(
        f"""
        ### Configure Environment
        {initial_inventory} {season_length_weeks} {base_price} {base_demand} 
        {price_elasticity} {holding_cost_per_unit}{end_season_salvage_value} 
        {available_discounts}

        ### Configure Agent
        {learning_rate} {discount_factor} {exploration_rate} {exploration_decay} 
        {min_exploration_rate} {num_training_episodes}

        {run_button}
        """
    )

    return


@app.cell
def _():
    from config.config import DynamicPricingMDPConfig, QLearningAgentConfig
    from demos.dynamic_pricing import demonstrate_mdp_dynamic_pricing

    return (
        DynamicPricingMDPConfig,
        QLearningAgentConfig,
        demonstrate_mdp_dynamic_pricing,
    )


@app.cell
def _(
    DynamicPricingMDPConfig,
    QLearningAgentConfig,
    available_discounts,
    base_demand,
    base_price,
    demonstrate_mdp_dynamic_pricing,
    discount_factor,
    end_season_salvage_value,
    exploration_decay,
    exploration_rate,
    holding_cost_per_unit,
    initial_inventory,
    learning_rate,
    min_exploration_rate,
    mo,
    num_training_episodes,
    price_elasticity,
    run_button,
    season_length_weeks,
):
    # Stop execution if button not clicked
    mo.stop(not run_button.value, "Click 'Run Simulation' to start.")

    # Prepare config objects
    env_config = DynamicPricingMDPConfig(
        initial_inventory=initial_inventory.value,
        season_length_weeks=season_length_weeks.value,
        base_price=base_price.value,
        base_demand=base_demand.value,
        price_elasticity=price_elasticity.value,
        holding_cost_per_unit=holding_cost_per_unit.value,
        end_season_salvage_value=end_season_salvage_value.value,
        available_discounts=sorted(list(available_discounts.value)),
    )
    agent_config = QLearningAgentConfig(
        learning_rate=learning_rate.value,
        discount_factor=discount_factor.value,
        exploration_rate=exploration_rate.value,
        exploration_decay=exploration_decay.value,
        min_exploration_rate=min_exploration_rate.value,
        action_space_size=len(env_config.available_discounts),
    )

    # Run the demonstration

    # Run simulation and return results
    # Make sure demonstrate_mdp_dynamic_pricing returns the dict as shown before
    results = demonstrate_mdp_dynamic_pricing(
        env_config=env_config,
        agent_config=agent_config,
        num_training_episodes=num_training_episodes.value,
        verbose=True,  # Keep verbose logging in console
    )

    # Return the raw results needed for visualization
    # Ensure results dict contains 'episode_returns', 'policy', 'env', 'agent'
    return agent_config, results


@app.cell
def _(mo, results):
    env = results.get("env")

    # --- Define UI elements for visualization control ---
    week_slider = mo.ui.slider(
        1,
        env.season_length_weeks,
        value=env.season_length_weeks // 2,
        label="Select Week for Policy Heatmap",
    )
    q_week = mo.ui.slider(
        1,
        env.season_length_weeks,
        value=env.season_length_weeks // 2,
        label="Q-Explorer: Week",
    )
    q_inv = mo.ui.slider(
        0,
        env.initial_inventory,
        value=env.initial_inventory // 2,
        label="Q-Explorer: Inventory",
    )
    q_disc = mo.ui.slider(
        0, len(env.available_discounts) - 1, value=0, label="Q-Explorer: Discount Index"
    )

    return env, q_disc, q_inv, q_week, week_slider


@app.cell
def _(mo, q_disc, q_inv, q_week, week_slider):
    mo.md(
        f"""
        {week_slider} {q_week} {q_inv} {q_disc}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""### Check if UI elements were created 
        (handles case where Cell 3 returned None)"""
    )
    return


@app.cell
def _(mo, week_slider):
    if week_slider is None:
        mo.md("**Waiting for simulation results...**")

    import altair as alt

    return (alt,)


@app.cell
def _(results):
    episode_returns = results.get("episode_returns", [])
    policy = results.get("policy", {})
    pricing_agent = results.get("agent")
    return episode_returns, policy, pricing_agent


@app.cell
def _(mo):
    mo.md(r"""### Check if results are valid before proceeding""")
    return


@app.cell
def _(env, episode_returns, mo, policy, pricing_agent):
    if not episode_returns or not policy or not env or not pricing_agent:
        mo.md("**Error:** Simulation results are missing or invalid.")
    num_training_episodes_val = len(episode_returns)
    return (num_training_episodes_val,)


@app.cell
def _(mo):
    mo.md("""### 1. Learning Curve""")
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    return np, pd, plt


@app.cell
def _(episode_returns, num_training_episodes_val, pd, plt):
    fig_curve, ax_curve = plt.subplots(figsize=(10, 5))
    ax_curve.plot(episode_returns, label="Episode Return", alpha=0.6)
    window_size = max(1, num_training_episodes_val // 20)
    if len(episode_returns) >= window_size:
        moving_avg = pd.Series(episode_returns).rolling(window=window_size).mean()
        ax_curve.plot(
            moving_avg,
            label=f"Moving Average (Window={window_size})",
            color="red",
            linewidth=2,
        )
    ax_curve.set_xlabel("Episode")
    ax_curve.set_ylabel("Total Return (Profit)")
    ax_curve.set_title("Q-Learning Agent Performance Over Training Episodes")
    ax_curve.legend()
    ax_curve.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.close(fig_curve)  # Prevent double rendering
    return (fig_curve,)


@app.cell
def _(mo):
    mo.md(r"""### 2. Exploration Rate Over Time""")
    return


@app.cell
def _(agent_config, num_training_episodes_val, plt):
    epsilons = [agent_config.exploration_rate]
    for _ in range(num_training_episodes_val):  # Use actual number of episodes
        next_epsilon = max(
            agent_config.min_exploration_rate,
            epsilons[-1] * agent_config.exploration_decay,
        )
        epsilons.append(next_epsilon)
    # Adjust length if needed
    if len(epsilons) > num_training_episodes_val + 1:
        epsilons = epsilons[1:]
    fig_epsilon, ax_epsilon = plt.subplots(figsize=(10, 3))
    ax_epsilon.plot(
        range(num_training_episodes_val + 1),
        epsilons[: num_training_episodes_val + 1],
        color="purple",
    )
    ax_epsilon.set_xlabel("Episode")
    ax_epsilon.set_ylabel("Exploration Rate (epsilon)")
    ax_epsilon.set_title("Exploration Rate Decay Over Time")
    ax_epsilon.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.close(fig_epsilon)
    return (fig_epsilon,)


@app.cell
def _(mo):
    mo.md(r"""### 3. Episode Reward Histogram""")
    return


@app.cell
def _(episode_returns, plt):
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    ax_hist.hist(
        episode_returns, bins=30, color="skyblue", edgecolor="black", alpha=0.7
    )
    ax_hist.set_xlabel("Total Return (Profit)")
    ax_hist.set_ylabel("Episode Count")
    ax_hist.set_title("Distribution of Episode Rewards")
    plt.tight_layout()
    plt.close(fig_hist)
    return (fig_hist,)


@app.cell
def _(mo):
    mo.md(r"""#### 4. Policy Heatmap""")
    return


@app.cell
def _(alt, env, mo, pd, policy, week_slider):
    heatmap_data = []
    inv_step = max(1, env.initial_inventory // 20)
    for inv in range(0, env.initial_inventory + 1, inv_step):
        for disc_idx in range(len(env.available_discounts)):
            # NOW uses the week_slider passed in as an argument
            state = (week_slider.value, inv, disc_idx)
            best_action = policy.get(state, None)
            heatmap_data.append(
                {
                    "Inventory": inv,
                    "Current Discount Index": disc_idx,
                    "Best Action": best_action if best_action is not None else -1,
                }
            )
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_chart = (
        alt.Chart(heatmap_df)
        .mark_rect()
        .encode(
            x=alt.X(
                "Inventory:O", title="Inventory Level", sort=alt.SortField("Inventory")
            ),
            y=alt.Y("Current Discount Index:O", title="Current Discount Index"),
            color=alt.Color(
                "Best Action:O",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title="Best Action Index"),
            ),
            tooltip=["Inventory", "Current Discount Index", "Best Action"],
        )
        .properties(
            width=400, height=300, title=f"Policy Heatmap (Week={week_slider.value})"
        )
        .interactive()
    )
    heatmap_ui = mo.ui.altair_chart(heatmap_chart)
    return (heatmap_ui,)


@app.cell
def _(mo):
    mo.md(r"""### 5. Q-Value Explorer""")
    return


@app.cell
def _(env, mo, np, pd, pricing_agent, q_disc, q_inv, q_week):
    q_state = (q_week.value, q_inv.value, q_disc.value)
    q_values = pricing_agent.q_table.get(
        q_state, np.zeros(len(env.available_discounts))
    )
    q_df = pd.DataFrame(
        {
            "Discount Index": list(range(len(env.available_discounts))),
            "Discount (%)": [
                env.available_discounts[i] * 100
                for i in range(len(env.available_discounts))
            ],
            "Q-Value": q_values,
        }
    )
    q_best_action_idx = (
        int(np.argmax(q_values))
        if len(q_values) > 0 and q_state in pricing_agent.q_table
        else None
    )
    q_explorer_md_content = (
        f"""
    **Q-Explorer State:** Week={q_week.value}, Inventory={q_inv.value}, 
    Discount Index={q_disc.value}
    **Q-Values:**
    {mo.ui.table(q_df).to_html()}
    **Best Action:** {q_best_action_idx} (Discount 
    {env.available_discounts[q_best_action_idx] * 100:.0f}%)
    """
        if q_best_action_idx is not None
        else f"""
    **Q-Explorer State:** Week={q_week.value}, Inventory={q_inv.value}, 
    Discount Index={q_disc.value}
    *No Q-values learned for this specific state.*
    """
    )
    q_explorer_md = mo.md(q_explorer_md_content)
    return (q_explorer_md,)


@app.cell
def _(mo):
    mo.md(r"""### 6. Policy Table Sample""")
    return


@app.cell
def _(env, mo, policy, pricing_agent):
    sample_states = [
        (env.season_length_weeks - 1, env.initial_inventory - 5, 0),
        (env.season_length_weeks // 2, env.initial_inventory // 2, 0),
        (env.season_length_weeks // 2, env.initial_inventory - 20, 1),
        (2, env.initial_inventory // 4, 2),
        (1, 10, 3),
    ]
    policy_rows = []
    for sample_state in sample_states:
        valid_state = (
            max(1, min(env.season_length_weeks, sample_state[0])),
            max(0, min(env.initial_inventory, sample_state[1])),
            max(0, min(len(env.available_discounts) - 1, sample_state[2])),
        )
        if valid_state in policy:
            action_idx = policy[valid_state]
            action_discount = env.available_discounts[action_idx] * 100
            q_values_str = ", ".join(
                [f"{q:.2f}" for q in pricing_agent.q_table.get(valid_state, [])]
            )
            policy_rows.append(
                f"| {valid_state} | {action_idx} ({action_discount:.0f}%) | "
                f"[{q_values_str}] |"
            )
        else:
            policy_rows.append(f"| {valid_state} | *State not visited* | - |")
    policy_md_content = r"""
                ### Policy Insights (Sample States)
                | State (Weeks, Inv, DiscIdx) | Best Action (Disc %) | Q-Values (Approx) |
                | :-------------------------- | :------------------- | :---------------- |
                """ + "\n".join(policy_rows)
    policy_md = mo.md(policy_md_content)
    return (policy_md,)


@app.cell
def _(mo):
    mo.md(r"""### 7. Q-table sample""")
    return


@app.cell
def _(env, pricing_agent):
    q_table_df = (
        pricing_agent.get_q_table_df(env.available_discounts)
        if hasattr(pricing_agent, "get_q_table_df")
        else None
    )  # Check if method exists
    return (q_table_df,)


@app.cell
def _(q_table_df):
    q_table_df
    return


@app.cell
def _(mo, q_table_df):
    q_table_md_content = "### Q-Table Sample\n\n"
    if q_table_df is not None and not q_table_df.empty:
        q_table_md_content += q_table_df.head(15).to_html()
        q_table_md_content += (
            f"\n\n*(Showing first 15 rows of {len(q_table_df)} total learned states)*"
        )
    else:
        q_table_md_content += "*Q-table is empty or agent did not learn any states.*"
    q_table_md = mo.md(q_table_md_content)
    return (q_table_md,)


@app.cell
def _(mo):
    mo.md(r"""### Organize all visualizations in tabs""")
    return


@app.cell
def _(
    fig_curve,
    fig_epsilon,
    fig_hist,
    heatmap_ui,
    mo,
    policy_md,
    q_disc,
    q_explorer_md,
    q_inv,
    q_table_md,
    q_week,
    week_slider,
):
    tabs = mo.ui.tabs(
        {
            "Learning Curve": mo.vstack(
                [
                    mo.md("## MDP Dynamic Pricing Results"),
                    mo.md("### Learning Curve"),
                    mo.mpl.interactive(fig_curve),  # Use interactive wrapper
                ]
            ),
            "Exploration Rate": mo.mpl.interactive(fig_epsilon),
            "Reward Histogram": mo.mpl.interactive(fig_hist),
            "Policy Heatmap": mo.vstack([week_slider, heatmap_ui]),
            "Q-Value Explorer": mo.vstack([q_week, q_inv, q_disc, q_explorer_md]),
            "Policy Table": policy_md,
            "Q-Table Sample": q_table_md,
        }
    )
    return (tabs,)


@app.cell
def _(tabs):
    tabs
    return


@app.cell
def _():
    def _(mo):
        # This cell can be used to export functions or variables if needed.
        # For example, if you wanted to use the trained agent or policy elsewhere.

        # Or simply run final checks or summaries.
        mo.md("All decision-making framework demonstrations are complete.")

    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Chapter 5: Decision-Making - RL & Planning

        ### Store Fulfillment Optimization Demonstration

        This demonstrates the planning algorithms for optimizing in-store order 
        fulfillment, including order batching, associate assignment, and 
        path optimization.
        
        - For Models (`Item`, `Order`, `Associate`):
            ```python
                from models.fulfillment import Item, Order, Associate
            ```
        - For Planning (`StoreLayout`, `FulfillmentPlanner`):
            ```python
                from utils.planning import StoreLayout, FulfillmentPlanner
            ```
        - For Demo (`demo_fulfillment_system`):
             ```python
                from demos.fulfillment_planning_demo import demo_fulfillment_system
            ```
        """
    )
    return


@app.cell
def _(mo):
    # Import the demo function from the new location
    from demos.fulfillment_planning_demo import demo_fulfillment_system
    
    # Run the demo, passing the Marimo instance for output rendering
    fulfillment_output = demo_fulfillment_system(mo)
    
    # Display the output returned by the demo function
    return fulfillment_output


if __name__ == "__main__":
    app.run()
