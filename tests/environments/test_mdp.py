from pathlib import Path
from unittest.mock import patch

import numpy as np  # Needed for mocking
import pytest

from config.config import DynamicPricingMDPConfig

# Module to test
from environments.mdp import DynamicPricingMDP

# --- Test Fixtures --- #


@pytest.fixture
def mdp_config() -> DynamicPricingMDPConfig:
    """Provides a default DynamicPricingMDPConfig."""
    # Use smaller numbers for easier testing
    return DynamicPricingMDPConfig(
        initial_inventory=20,
        season_length_weeks=5,
        base_price=100.0,
        base_demand=5.0,
        available_discounts=[0.0, 0.1, 0.5],  # 3 actions
    )


@pytest.fixture
def mdp_env(mdp_config: DynamicPricingMDPConfig) -> DynamicPricingMDP:
    """Provides a DynamicPricingMDP instance with default config."""
    return DynamicPricingMDP(config=mdp_config)


# --- Test Initialization --- #


def test_mdp_initialization(mdp_env: DynamicPricingMDP, mdp_config: DynamicPricingMDPConfig):
    """Test that the MDP initializes with correct state and config."""
    assert mdp_env.config == mdp_config
    assert mdp_env.current_week == 0
    assert mdp_env.current_inventory == mdp_config.initial_inventory
    assert mdp_env.current_discount_index == 0
    assert mdp_env._episode_rewards == []
    assert mdp_env._episode_states == []
    assert mdp_env._episode_actions == []

    # Check initial state from _get_state()
    expected_initial_state = (
        mdp_config.season_length_weeks,  # weeks_remaining
        mdp_config.initial_inventory,
        0,  # initial discount index
    )
    assert mdp_env._get_state() == expected_initial_state


# --- Test reset() --- #


def test_mdp_reset(mdp_env: DynamicPricingMDP, mdp_config: DynamicPricingMDPConfig):
    """Test that reset() correctly resets the environment state."""
    # Modify the state slightly
    mdp_env.current_week = 2
    mdp_env.current_inventory = 10
    mdp_env.current_discount_index = 1
    mdp_env._episode_rewards = [100.0, 90.0]
    mdp_env._episode_states = [(4, 15, 0), (3, 10, 1)]
    mdp_env._episode_actions = [0, 1]

    # Reset the environment
    reset_state = mdp_env.reset()

    # Verify state is reset
    assert mdp_env.current_week == 0
    assert mdp_env.current_inventory == mdp_config.initial_inventory
    assert mdp_env.current_discount_index == 0
    assert mdp_env._episode_rewards == []
    assert mdp_env._episode_states == []
    assert mdp_env._episode_actions == []

    # Verify returned state is the correct initial state
    expected_initial_state = (
        mdp_config.season_length_weeks,
        mdp_config.initial_inventory,
        0,
    )
    assert reset_state == expected_initial_state


# --- Test step() --- #


# Helper to calculate expected demand for verification
def _calculate_expected_demand(config: DynamicPricingMDPConfig, discount: float) -> float:
    if discount >= 1.0:  # Avoid price <= 0
        return config.base_demand * 10  # Arbitrary high demand for free item
    discounted_price = config.base_price * (1.0 - discount)
    price_ratio = config.base_price / discounted_price
    return config.base_demand * (price_ratio**config.price_elasticity)


@patch("numpy.random.normal")
def test_mdp_step_basic(mock_np_normal, mdp_env: DynamicPricingMDP, mdp_config: DynamicPricingMDPConfig):
    """Test a basic step where inventory is sufficient."""
    action_idx = 1  # Corresponds to 0.1 discount in fixture
    discount = mdp_config.available_discounts[action_idx]
    start_inventory = mdp_config.initial_inventory
    start_week = mdp_env.current_week

    # Calculate expected demand before noise and week effect
    base_expected_demand = _calculate_expected_demand(mdp_config, discount)
    # Demand noise: mock normal to return expected value (mean = expected_demand)
    mock_np_normal.return_value = base_expected_demand

    # Calculate expected actual demand after week effect and rounding
    week_effect = 1.0 + 0.2 * np.sin(np.pi * start_week / mdp_config.season_length_weeks)
    expected_actual_demand = int(round(base_expected_demand * week_effect))

    # Expected sales (inventory > demand)
    expected_sales = expected_actual_demand
    expected_end_inventory = start_inventory - expected_sales
    expected_revenue = expected_sales * (mdp_config.base_price * (1.0 - discount))
    expected_holding_cost = start_inventory * mdp_config.holding_cost_per_unit
    expected_reward = expected_revenue - expected_holding_cost

    next_state, reward, done, info = mdp_env.step(action_idx)

    # Check return values
    assert next_state == (
        mdp_config.season_length_weeks - (start_week + 1),  # weeks_remaining
        expected_end_inventory,
        action_idx,
    )
    assert reward == pytest.approx(expected_reward)
    assert done is False
    assert info["sales"] == expected_sales
    assert info["reward"] == pytest.approx(expected_reward)
    assert info["end_inventory"] == expected_end_inventory

    # Check internal state update
    assert mdp_env.current_week == start_week + 1
    assert mdp_env.current_inventory == expected_end_inventory
    assert mdp_env.current_discount_index == action_idx


@patch("numpy.random.normal")
def test_mdp_step_inventory_limited(mock_np_normal, mdp_env: DynamicPricingMDP, mdp_config: DynamicPricingMDPConfig):
    """Test a step where sales are limited by current inventory."""
    action_idx = 0  # 0% discount
    start_inventory = 3  # Low starting inventory
    mdp_env.current_inventory = start_inventory
    start_week = mdp_env.current_week
    discount = mdp_config.available_discounts[action_idx]

    # Calculate expected demand and make mock return high value
    base_expected_demand = _calculate_expected_demand(mdp_config, discount)
    week_effect = 1.0 + 0.2 * np.sin(np.pi * start_week / mdp_config.season_length_weeks)
    high_demand = (base_expected_demand * week_effect) + 10  # Ensure demand > inventory
    mock_np_normal.return_value = high_demand

    # Expected sales capped by inventory
    expected_sales = start_inventory
    expected_end_inventory = 0
    expected_revenue = expected_sales * (mdp_config.base_price * (1.0 - discount))
    expected_holding_cost = start_inventory * mdp_config.holding_cost_per_unit
    expected_reward = expected_revenue - expected_holding_cost

    next_state, reward, done, info = mdp_env.step(action_idx)

    assert info["sales"] == expected_sales
    assert info["end_inventory"] == expected_end_inventory
    assert reward == pytest.approx(expected_reward)
    assert mdp_env.current_inventory == expected_end_inventory
    assert next_state[1] == expected_end_inventory
    assert done is False


@patch("numpy.random.normal")
def test_mdp_step_end_of_season_with_salvage(mock_np_normal, mdp_env: DynamicPricingMDP, mdp_config: DynamicPricingMDPConfig):
    """Test the last step includes salvage value in reward."""
    action_idx = 2  # 50% discount
    # Set state to be the start of the last week
    mdp_env.current_week = mdp_config.season_length_weeks - 1
    start_inventory = 10  # Some inventory left
    mdp_env.current_inventory = start_inventory
    discount = mdp_config.available_discounts[action_idx]

    # Mock demand to be less than inventory
    mock_np_normal.return_value = 1.0  # Low demand
    expected_sales = 1  # Expected sales based on mock demand and week effect

    inventory_after_sales = start_inventory - expected_sales
    expected_revenue = expected_sales * (mdp_config.base_price * (1.0 - discount))
    expected_holding_cost = start_inventory * mdp_config.holding_cost_per_unit
    expected_salvage = inventory_after_sales * mdp_config.end_season_salvage_value
    expected_reward = expected_revenue - expected_holding_cost + expected_salvage

    next_state, reward, done, info = mdp_env.step(action_idx)

    assert done is True
    assert info["sales"] == expected_sales
    assert mdp_env.current_inventory == inventory_after_sales
    assert reward == pytest.approx(expected_reward)
    assert next_state[0] == 0  # Weeks remaining is 0


@patch("numpy.random.normal")
def test_mdp_step_end_of_season_no_salvage(mock_np_normal, mdp_env: DynamicPricingMDP, mdp_config: DynamicPricingMDPConfig):
    """Test the last step has no salvage value if inventory is zero."""
    action_idx = 0  # 0% discount
    # Set state to be the start of the last week with low inventory
    mdp_env.current_week = mdp_config.season_length_weeks - 1
    start_inventory = 5
    mdp_env.current_inventory = start_inventory
    discount = mdp_config.available_discounts[action_idx]

    # Mock demand to sell exactly remaining inventory
    _base_expected_demand = _calculate_expected_demand(mdp_config, discount)
    week_effect = 1.0 + 0.2 * np.sin(np.pi * mdp_env.current_week / mdp_config.season_length_weeks)
    # Calculate the demand value needed before rounding to sell exactly 5
    demand_to_sell_5 = 5 / week_effect
    mock_np_normal.return_value = demand_to_sell_5
    expected_sales = 5

    _inventory_after_sales = start_inventory - expected_sales  # Should be 0
    expected_revenue = expected_sales * (mdp_config.base_price * (1.0 - discount))
    expected_holding_cost = start_inventory * mdp_config.holding_cost_per_unit
    expected_salvage = 0  # No salvage
    expected_reward = expected_revenue - expected_holding_cost + expected_salvage

    next_state, reward, done, info = mdp_env.step(action_idx)

    assert done is True
    assert info["sales"] == expected_sales
    assert mdp_env.current_inventory == 0
    assert reward == pytest.approx(expected_reward)
    assert next_state[1] == 0  # Inventory state is 0


def test_mdp_step_invalid_action(mdp_env: DynamicPricingMDP, mdp_config: DynamicPricingMDPConfig):
    """Test that an invalid action index raises ValueError."""
    invalid_action_idx = len(mdp_config.available_discounts)  # Out of bounds
    with pytest.raises(ValueError, match=f"Invalid action index: {invalid_action_idx}"):
        mdp_env.step(invalid_action_idx)


# --- Test get_available_actions() --- #


def test_mdp_get_available_actions(mdp_env: DynamicPricingMDP, mdp_config: DynamicPricingMDPConfig):
    """Test that get_available_actions returns the correct list of indices."""
    expected_actions = list(range(len(mdp_config.available_discounts)))
    assert mdp_env.get_available_actions() == expected_actions


# --- Test save() / load() --- #


def test_mdp_save_load(mdp_env: DynamicPricingMDP, mdp_config: DynamicPricingMDPConfig, tmp_path: Path):
    """Test saving and loading the MDP environment state."""
    # Modify state
    mdp_env.step(1)  # Take one step
    mdp_env.step(0)

    # Store current state details for comparison
    current_week = mdp_env.current_week
    current_inventory = mdp_env.current_inventory
    current_discount_index = mdp_env.current_discount_index
    current_rewards = mdp_env._episode_rewards.copy()
    current_states = mdp_env._episode_states.copy()
    current_actions = mdp_env._episode_actions.copy()

    save_file = tmp_path / "mdp_state.pkl"

    # Save
    mdp_env.save(str(save_file))
    assert save_file.exists()

    # Create a new instance and load
    new_mdp_env = DynamicPricingMDP(config=mdp_config)
    new_mdp_env.load(str(save_file))

    # Compare state
    assert new_mdp_env.current_week == current_week
    assert new_mdp_env.current_inventory == current_inventory
    assert new_mdp_env.current_discount_index == current_discount_index
    assert new_mdp_env._episode_rewards == current_rewards
    assert new_mdp_env._episode_states == current_states
    assert new_mdp_env._episode_actions == current_actions
    # Also check config is retained (although it should be)
    assert new_mdp_env.config == mdp_config


# Placeholder for actions tests
# def test_mdp_get_available_actions(): ...

# Placeholder for save/load tests
# def test_mdp_save_load(): ...
