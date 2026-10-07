"""
Demonstration and training logic for dynamic pricing MDP and Q-learning agent.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict

from agents.qlearning import QLearningAgent
from config.config import DynamicPricingMDPConfig, QLearningAgentConfig
from environments.mdp import DynamicPricingMDP
from utils.logger import get_logger

logger = get_logger("demos.dynamic_pricing")


class TrainingResult(BaseModel):
    episode_returns: list[float]
    policy: dict[tuple[int, int, int], int]


class DemoResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    episode_returns: list[float]
    policy: dict[tuple[int, int, int], int]
    env: DynamicPricingMDP
    agent: QLearningAgent


def train_agent(
    env: DynamicPricingMDP,
    agent: QLearningAgent,
    num_episodes: int = 5000,
    verbose: bool = False,
) -> TrainingResult:
    """
    Train a Q-learning agent on the Dynamic Pricing MDP.
    Returns:
        Tuple containing:
        - List of total returns for each episode.
        - The learned policy (dictionary mapping state to best action index).
    """
    episode_returns: list[float] = []
    print_every = max(1, num_episodes // 20)
    logger.info(f"Starting training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_return: float = 0.0
        steps = 0
        while not done:
            available_actions = env.get_available_actions()
            action = agent.choose_action(state, available_actions)
            next_state, reward, done, info = env.step(action)
            episode_return += float(reward)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            steps += 1
            if steps > env.season_length_weeks * 2:
                logger.warning(f"Episode {episode + 1} exceeded max steps. Breaking.")
                break
        agent.decay_exploration()
        episode_returns.append(episode_return)
        if verbose and (episode + 1) % print_every == 0:
            avg_return = np.mean(episode_returns[-print_every:])
            logger.info(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Avg Return (last {print_every}): {avg_return:.2f} | "
                f"Exploration Rate: {agent.exploration_rate:.4f}"
            )
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Avg Return (last {print_every}): {avg_return:.2f} | "
                f"Exploration Rate: {agent.exploration_rate:.4f}"
            )
    logger.info("Training complete.")
    policy: dict[tuple[int, int, int], int] = agent.get_policy()
    logger.info(f"Learned policy has {len(policy)} state entries.")
    return TrainingResult(episode_returns=episode_returns, policy=policy)


def demonstrate_mdp_dynamic_pricing(
    env_config: DynamicPricingMDPConfig | None = None,
    agent_config: QLearningAgentConfig | None = None,
    num_training_episodes: int = 10000,
    verbose: bool = True,
) -> DemoResult:
    """
    Demonstrate the MDP for dynamic pricing with Q-learning.
    Returns a dictionary with results and artifacts for further use or display.
    """
    logger.info("--- Starting MDP Dynamic Pricing Demonstration ---")
    if env_config is None:
        env_config = DynamicPricingMDPConfig()

    if agent_config is None:
        agent_config = QLearningAgentConfig(action_space_size=len(env_config.available_discounts))
    env = DynamicPricingMDP(env_config)
    agent = QLearningAgent(agent_config)
    training = train_agent(env, agent, num_episodes=num_training_episodes, verbose=verbose)
    # Learning curve data
    results = DemoResult(
        episode_returns=training.episode_returns,
        policy=training.policy,
        env=env,
        agent=agent,
    )
    logger.info("--- MDP Dynamic Pricing Demonstration Complete ---")
    return results


if __name__ == "__main__":
    demonstrate_mdp_dynamic_pricing()
