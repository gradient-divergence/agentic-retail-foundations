# region book:mdp-dynamic-imports
import random
from collections import defaultdict

import numpy as np
from pydantic import BaseModel


class DynamicPricingMDP:
    """
    An MDP formulation for dynamic pricing of a seasonal product.

    States: (weeks_remaining, inventory_level, current_discount)
    Actions: Set discount to 0%, 20%, 40%, or 60%
    Rewards: Revenue from sales minus inventory holding costs
    """

    # endregion book:mdp-dynamic-imports

    # region book:mdp-dynamic-init
    def __init__(
        self,
        initial_inventory: int = 100,
        season_length_weeks: int = 10,
        base_price: float = 50.0,
        base_demand: float = 10.0,
        price_elasticity: float = 1.5,
        holding_cost_per_unit: float = 0.5,
        end_season_salvage_value: float = 15.0,
        available_discounts: list[float] = None,
    ):
        """
        Initialize the Dynamic Pricing MDP.
        """
        self.initial_inventory = initial_inventory
        self.season_length_weeks = season_length_weeks
        self.base_price = base_price
        self.base_demand = base_demand
        self.price_elasticity = price_elasticity
        self.holding_cost_per_unit = holding_cost_per_unit
        self.end_season_salvage_value = end_season_salvage_value
        # endregion book:mdp-dynamic-init

        # region book:mdp-dynamic-discounts
        # Available discount levels
        self.available_discounts = available_discounts or [0.0, 0.2, 0.4, 0.6]
        # Define state space dimensions
        self.max_inventory = initial_inventory
        # For tracking performance
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []
        # endregion book:mdp-dynamic-discounts

    # region book:mdp-dynamic-reset
    def reset(self) -> tuple[int, int, float]:
        """Reset the environment to the initial state and return it."""
        self.current_week = 0
        self.current_inventory = self.initial_inventory
        self.current_discount = 0.0
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []

        # Return initial state: (weeks_remaining, inventory_level, current_discount)
        return (
            self.season_length_weeks - self.current_week,
            self.current_inventory,
            self.current_discount,
        )

    # endregion book:mdp-dynamic-reset

    # region book:mdp-dynamic-step
    def step(self, action_idx: int) -> tuple[tuple, float, bool, dict]:
        """
        Take an action (set a discount) and transition to the next state.
        """
        if self.current_week >= self.season_length_weeks:
            raise RuntimeError("Episode is complete; call reset() before stepping again.")
        if not 0 <= action_idx < len(self.available_discounts):
            raise ValueError(f"Invalid action index: {action_idx}")
        # Get the discount percentage from the action index
        new_discount = self.available_discounts[action_idx]
        # Apply the discount and calculate sales
        discounted_price = self.base_price * (1 - new_discount)
        # Calculate expected demand based on price elasticity
        # Higher discount → higher demand, with elasticity controlling sensitivity
        price_ratio = (self.base_price / discounted_price) if discounted_price > 0 else 1.0
        expected_demand = self.base_demand * (price_ratio**self.price_elasticity)
        # Add randomness to demand (normally distributed around expected_demand)
        # Standard deviation is 20% of expected demand
        actual_demand = max(0, np.random.normal(expected_demand, 0.2 * expected_demand))
        # Season week effect: demand increases mid-season and then decreases
        week_effect = 1.0 + 0.2 * np.sin(np.pi * self.current_week / self.season_length_weeks)
        actual_demand *= week_effect

        # Calculate sales, revenue, holding costs, and update state for the next time step
        # Limit sales by available inventory
        sales = min(self.current_inventory, int(actual_demand))
        # Calculate revenue
        revenue = sales * discounted_price
        # Update inventory
        self.current_inventory -= sales
        # Calculate holding cost for remaining inventory
        holding_cost = self.current_inventory * self.holding_cost_per_unit
        # Calculate reward (revenue minus holding cost)
        reward = revenue - holding_cost
        self.current_week += 1
        self.current_discount = new_discount
        # Check if the season is over
        done = self.current_week >= self.season_length_weeks

        # Add end-of-season salvage value and return next state, reward, done, and debug info
        # End-of-season salvage value
        if done and self.current_inventory > 0:
            salvage_revenue = self.current_inventory * self.end_season_salvage_value
            reward += salvage_revenue

        next_state = (
            self.season_length_weeks - self.current_week,
            self.current_inventory,
            self.current_discount,
        )

        # Store for episode tracking
        self.episode_rewards.append(reward)
        self.episode_states.append(next_state)
        self.episode_actions.append(action_idx)

        # Additional info for debugging
        info = {
            "sales": sales,
            "revenue": revenue,
            "holding_cost": holding_cost,
            "expected_demand": expected_demand,
            "actual_demand": actual_demand,
            "discounted_price": discounted_price,
        }

        return next_state, reward, done, info

    def get_available_actions(self) -> list[int]:
        """Return indices of all available actions."""
        return list(range(len(self.available_discounts)))

    # endregion book:mdp-dynamic-step


# region book:mdp-dynamic-agent-class
class QLearningAgent:
    """
    A Q-learning agent for solving the Dynamic Pricing MDP.
    Q-learning is a model-free reinforcement learning algorithm that learns
    a policy by directly estimating the Q-values (expected future rewards)
    for each state-action pair.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.3,
        exploration_decay: float = 0.99,
    ):
        """Initialize the Q-learning agent."""
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    # endregion book:mdp-dynamic-agent-class

    # region book:mdp-dynamic-choose-action
    def choose_action(self, state, available_actions) -> int:
        """
        Select an action using an epsilon-greedy policy.
        With probability exploration_rate, choose a random action.
        Otherwise, choose the action with the highest Q-value.
        """
        # Exploration: choose a random action
        if np.random.random() < self.exploration_rate:
            return random.choice(available_actions)

        # Exploitation: choose the best action based on Q-values
        # If multiple actions have the same Q-value, choose randomly among them
        q_values = [self.q_table[state][a] for a in available_actions]
        max_q = max(q_values)
        # Find all actions with the max Q-value
        best_actions = [a for a, q in zip(available_actions, q_values, strict=False) if q == max_q]

        return random.choice(best_actions)

    # endregion book:mdp-dynamic-choose-action

    # region book:mdp-dynamic-update
    def update(self, state, action, reward, next_state, next_available_actions, done):
        """
        Update Q-values using the Q-learning update rule.
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        # Calculate best next action's Q-value
        if done:
            max_next_q = 0  # Terminal state has no future reward
        else:
            # Best Q-value for any action in the next state
            next_q_values = [self.q_table[next_state][a] for a in next_available_actions]
            max_next_q = max(next_q_values) if next_q_values else 0

        # Calculate the TD (Temporal Difference) target
        td_target = reward + self.discount_factor * max_next_q
        # Calculate the TD error
        td_error = td_target - self.q_table[state][action]
        # Update the Q-value
        self.q_table[state][action] += self.learning_rate * td_error

        return td_error

    # endregion book:mdp-dynamic-update

    # region book:mdp-dynamic-decay
    def decay_exploration(self):
        """Decrease the exploration rate over time."""
        self.exploration_rate *= self.exploration_decay

    # endregion book:mdp-dynamic-decay

    # region book:mdp-dynamic-policy
    def get_policy(self) -> "PolicyMap":
        """Extract the learned policy from the Q-table."""
        policy = {}

        for state in self.q_table:
            # Find the action with the highest Q-value for this state
            best_action = max(self.q_table[state], key=self.q_table[state].get, default=0)
            policy[state] = best_action

        return PolicyMap(policy=policy)

    # endregion book:mdp-dynamic-policy


# region book:mdp-dynamic-train
class PolicyMap(BaseModel):
    policy: dict[tuple[int, int, float], int]


class TrainingResult(BaseModel):
    episode_returns: list[float]
    policy: PolicyMap


def train_agent(
    env: DynamicPricingMDP,
    agent: QLearningAgent,
    num_episodes: int = 1000,
    verbose: bool = False,
) -> TrainingResult:
    """
    Train a Q-learning agent on the Dynamic Pricing MDP.
    """
    episode_returns = []

    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()
        done = False
        episode_return = 0

        while not done:
            # Choose an action
            available_actions = env.get_available_actions()
            action = agent.choose_action(state, available_actions)
            # Take the action
            next_state, reward, done, _ = env.step(action)
            # Update the agent
            next_available_actions = env.get_available_actions()
            agent.update(state, action, reward, next_state, next_available_actions, done)
            # Update state and total return
            state = next_state
            episode_return += reward

        # Complete the training loop with exploration decay and progress tracking
        # Decay exploration rate
        agent.decay_exploration()
        # Record the total return for this episode
        episode_returns.append(episode_return)

        if verbose and (episode + 1) % max(1, num_episodes // 10) == 0:
            print(
                f"Episode {episode + 1}/{num_episodes}, "
                + f"Return: {episode_return:.2f}, "
                + f"Exploration rate: {agent.exploration_rate:.4f}"
            )

    # Extract the learned policy
    policy = agent.get_policy()

    return TrainingResult(episode_returns=episode_returns, policy=policy)


# endregion book:mdp-dynamic-train


# region book:mdp-dynamic-demo
def demonstrate_mdp_dynamic_pricing():
    """Demonstrate the MDP for dynamic pricing."""
    # Create the environment
    env = DynamicPricingMDP(
        initial_inventory=100,
        season_length_weeks=12,
        base_price=50.0,
        base_demand=10.0,
        price_elasticity=1.5,
        holding_cost_per_unit=0.5,
        end_season_salvage_value=15.0,
        available_discounts=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    # endregion book:mdp-dynamic-demo

    # region book:mdp-dynamic-create-agent
    # Create the agent
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=0.3,
        exploration_decay=0.99,
    )

    # Train the agent
    train_agent(env, agent, num_episodes=500, verbose=True)

    # Test the policy
    # Sample insight from the learned policy:
    # - Early in season: Minimal discounts unless inventory is very high
    # - Mid-season: Moderate discounts if inventory is above target trajectory
    # - End of season: Deep discounts to clear remaining inventory

    # Key pattern observed: The optimal policy tends to maintain regular price
    # when inventory follows expected sales trajectory, and only applies
    # discounts when inventory levels exceed target levels for the given week
    # endregion book:mdp-dynamic-create-agent


if __name__ == "__main__":
    demonstrate_mdp_dynamic_pricing()
