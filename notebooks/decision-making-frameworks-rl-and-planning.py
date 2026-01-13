import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Decision-Making Frameworks: RL and Planning

    This notebook provides a compact, reproducible entry point for
    reinforcement learning and planning examples that align with the chapter.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    from config.config import DynamicPricingMDPConfig, QLearningAgentConfig
    from demos.dynamic_pricing import demonstrate_mdp_dynamic_pricing
    return (
        DynamicPricingMDPConfig,
        QLearningAgentConfig,
        demonstrate_mdp_dynamic_pricing,
        np,
        pd,
    )


@app.cell
def _(mo):
    run_button = mo.ui.button(label="Run RL/Planning Demo")
    return (run_button,)


@app.cell
def _(
    DynamicPricingMDPConfig,
    QLearningAgentConfig,
    demonstrate_mdp_dynamic_pricing,
    mo,
    np,
    pd,
    run_button,
):
    mo.stop(not run_button.value, "Click 'Run RL/Planning Demo' to start.")

    np.random.seed(7)

    env_config = DynamicPricingMDPConfig(
        initial_inventory=50,
        season_length_weeks=8,
        base_price=40.0,
        base_demand=30,
        price_elasticity=0.15,
        holding_cost_per_unit=1.0,
        end_season_salvage_value=5.0,
        available_discounts=[0.0, 0.1, 0.2],
    )
    agent_config = QLearningAgentConfig(
        learning_rate=0.15,
        discount_factor=0.95,
        exploration_rate=0.8,
        exploration_decay=0.995,
        min_exploration_rate=0.05,
        action_space_size=len(env_config.available_discounts),
    )

    results = demonstrate_mdp_dynamic_pricing(
        env_config=env_config,
        agent_config=agent_config,
        num_training_episodes=300,
        verbose=False,
    )

    episode_returns = results.episode_returns
    summary_df = pd.DataFrame(
        [
            {"metric": "episodes", "value": len(episode_returns)},
            {
                "metric": "mean_return",
                "value": round(float(np.mean(episode_returns)), 2) if episode_returns else 0.0,
            },
            {
                "metric": "max_return",
                "value": round(float(np.max(episode_returns)), 2) if episode_returns else 0.0,
            },
        ]
    )
    return


if __name__ == "__main__":
    app.run()
