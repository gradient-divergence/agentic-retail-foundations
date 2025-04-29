"""
Configuration classes for agentic-retail-foundations project.
Defines hyperparameters for environments and agents in a type-safe, extensible way.
"""

from dataclasses import dataclass, field


@dataclass
class DynamicPricingMDPConfig:
    initial_inventory: int = 100
    season_length_weeks: int = 12
    base_price: float = 50.0
    base_demand: float = 10.0
    price_elasticity: float = 1.5
    holding_cost_per_unit: float = 0.5
    end_season_salvage_value: float = 15.0
    available_discounts: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    )


@dataclass
class QLearningAgentConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 1.0
    min_exploration_rate: float = 0.01
    exploration_decay: float = 0.995
    action_space_size: int = 6  # Should match len(available_discounts)


# Example usage:
# env_config = DynamicPricingMDPConfig()
# agent_config = QLearningAgentConfig(action_space_size=len(env_config.available_discounts))
