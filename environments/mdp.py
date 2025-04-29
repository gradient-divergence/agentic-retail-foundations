"""
MDP environment for dynamic pricing of a seasonal product.
Refactored for modular use in agentic-retail-foundations.
"""

import numpy as np
from config.config import DynamicPricingMDPConfig
from utils.logger import get_logger
import pickle


class DynamicPricingMDP:
    """
    An MDP formulation for dynamic pricing of a seasonal product.

    States: (weeks_remaining, inventory_level, current_discount_index)
    Actions: Index corresponding to an available discount in config.available_discounts
    Rewards: Revenue from sales minus inventory holding costs plus end-of-season salvage value.
    """

    def __init__(self, config: DynamicPricingMDPConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.initial_inventory = config.initial_inventory
        self.season_length_weeks = config.season_length_weeks
        self.base_price = config.base_price
        self.base_demand = config.base_demand
        self.price_elasticity = config.price_elasticity
        self.holding_cost_per_unit = config.holding_cost_per_unit
        self.end_season_salvage_value = config.end_season_salvage_value
        self.available_discounts = config.available_discounts
        self.max_inventory = self.initial_inventory
        self.current_week = 0
        self.current_inventory = self.initial_inventory
        self.current_discount_index = 0
        self._episode_rewards = []
        self._episode_states = []
        self._episode_actions = []
        self.logger.info(
            f"DynamicPricingMDP initialized: {self.season_length_weeks} weeks, {self.initial_inventory} units, discounts: {self.available_discounts}"
        )

    def _get_state(self) -> tuple[int, int, int]:
        inventory_state = max(0, self.current_inventory)
        inventory_state = min(self.initial_inventory, inventory_state)
        return (
            self.season_length_weeks - self.current_week,
            inventory_state,
            self.current_discount_index,
        )

    def reset(self) -> tuple[int, int, int]:
        self.current_week = 0
        self.current_inventory = self.initial_inventory
        self.current_discount_index = 0
        self._episode_rewards = []
        self._episode_states = []
        self._episode_actions = []
        initial_state = self._get_state()
        self.logger.debug(f"Environment reset. Initial state: {initial_state}")
        return initial_state

    def step(self, action_idx: int) -> tuple[tuple[int, int, int], float, bool, dict]:
        if not (0 <= action_idx < len(self.available_discounts)):
            self.logger.error(
                f"Invalid action index: {action_idx}. Available: {list(range(len(self.available_discounts)))}"
            )
            raise ValueError(f"Invalid action index: {action_idx}")
        new_discount = self.available_discounts[action_idx]
        discounted_price = self.base_price * (1 - new_discount)
        if discounted_price <= 0:
            expected_demand = self.base_demand * 10
        else:
            price_ratio = self.base_price / discounted_price
            expected_demand = self.base_demand * (price_ratio**self.price_elasticity)
        demand_std_dev = 0.15 * expected_demand
        actual_demand = max(0, np.random.normal(expected_demand, demand_std_dev))
        week_effect = 1.0 + 0.2 * np.sin(
            np.pi * self.current_week / self.season_length_weeks
        )
        actual_demand *= week_effect
        actual_demand = int(round(actual_demand))
        sales = min(self.current_inventory, actual_demand)
        revenue = sales * discounted_price
        previous_inventory = self.current_inventory
        self.current_inventory -= sales
        holding_cost = previous_inventory * self.holding_cost_per_unit
        reward = revenue - holding_cost
        self.current_week += 1
        self.current_discount_index = action_idx
        done = self.current_week >= self.season_length_weeks
        if done and self.current_inventory > 0:
            salvage_revenue = self.current_inventory * self.end_season_salvage_value
            reward += salvage_revenue
            self.logger.debug(
                f"End of season. Salvage value added: {salvage_revenue:.2f} for {self.current_inventory} units."
            )
        next_state = self._get_state()
        self._episode_rewards.append(reward)
        self._episode_states.append(next_state)
        self._episode_actions.append(action_idx)
        info = {
            "week": self.current_week - 1,
            "action_idx": action_idx,
            "discount_applied": new_discount,
            "discounted_price": discounted_price,
            "expected_demand": expected_demand,
            "actual_demand_real": actual_demand,
            "sales": sales,
            "revenue": revenue,
            "holding_cost": holding_cost,
            "start_inventory": previous_inventory,
            "end_inventory": self.current_inventory,
            "reward": reward,
            "done": done,
            "next_state": next_state,
        }
        self.logger.debug(
            f"Week {info['week']}: Action={action_idx}({new_discount * 100:.0f}%), Sales={sales}, Inv={previous_inventory}->{self.current_inventory}, Reward={reward:.2f}"
        )
        return next_state, reward, done, info

    def get_available_actions(self) -> list:
        return list(range(len(self.available_discounts)))

    def save(self, filepath: str):
        """Serialize the environment state to a file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.__dict__, f)
        self.logger.info(f"Environment state saved to {filepath}")

    def load(self, filepath: str):
        """Load the environment state from a file."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)
            self.__dict__.update(state)
        self.logger.info(f"Environment state loaded from {filepath}")
