import pytest
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch
import pandas as pd

# Module to test
from agents.qlearning import QLearningAgent
from config.config import QLearningAgentConfig

# --- Fixtures --- #

@pytest.fixture
def agent_config() -> QLearningAgentConfig:
    """Provides a default QLearningAgentConfig."""
    return QLearningAgentConfig(
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.5, # Start with some exploration
        min_exploration_rate=0.01,
        exploration_decay=0.99,
        action_space_size=3 # e.g., [0, 1, 2]
    )

@pytest.fixture
def q_agent(agent_config: QLearningAgentConfig) -> QLearningAgent:
    """Provides a QLearningAgent instance."""
    return QLearningAgent(config=agent_config)

# --- Test Initialization --- #

def test_qlearning_agent_initialization(q_agent: QLearningAgent, agent_config: QLearningAgentConfig):
    """Test agent initialization."""
    assert q_agent.config == agent_config
    assert q_agent.learning_rate == agent_config.learning_rate
    assert q_agent.discount_factor == agent_config.discount_factor
    assert q_agent.exploration_rate == agent_config.exploration_rate
    assert q_agent.min_exploration_rate == agent_config.min_exploration_rate
    assert q_agent.exploration_decay == agent_config.exploration_decay
    assert q_agent.action_space_size == agent_config.action_space_size
    assert isinstance(q_agent.q_table, defaultdict)

# --- Test choose_action --- #

@patch('numpy.random.random') # Mock the exploration check
@patch('numpy.random.choice') # Mock the action choice
def test_choose_action_explore(
    mock_choice, mock_random, q_agent: QLearningAgent
):
    """Test action selection when exploring."""
    mock_random.return_value = 0.1 # < exploration_rate (0.5)
    mock_choice.return_value = 1 # Mock random choice

    state = (10, 5, 0) # Example state
    available_actions = [0, 1, 2]
    chosen_action = q_agent.choose_action(state, available_actions)

    mock_random.assert_called_once()
    mock_choice.assert_called_once_with(available_actions)
    assert chosen_action == 1

@patch('numpy.random.random')
def test_choose_action_exploit(
    mock_random, q_agent: QLearningAgent
):
    """Test action selection when exploiting."""
    mock_random.return_value = 0.9 # > exploration_rate (0.5)

    state = (10, 5, 0)
    available_actions = [0, 1, 2]

    # Setup Q-table: action 1 has the highest Q-value
    q_agent.q_table[state][0] = 10.0
    q_agent.q_table[state][1] = 20.0
    q_agent.q_table[state][2] = 15.0

    chosen_action = q_agent.choose_action(state, available_actions)

    mock_random.assert_called_once()
    assert chosen_action == 1 # Should choose action with max Q-value

@patch('numpy.random.random')
@patch('numpy.random.choice') # Mock for tie-breaking
def test_choose_action_exploit_tie_breaking(
    mock_choice, mock_random, q_agent: QLearningAgent
):
    """Test random choice among actions with equal max Q-value when exploiting."""
    mock_random.return_value = 0.9 # Exploit
    # Mock choice to return a specific action from the tied best actions
    mock_choice.return_value = 2

    state = (5, 2, 1)
    available_actions = [0, 1, 2]

    # Setup Q-table: actions 0 and 2 have the same max Q-value
    q_agent.q_table[state][0] = 30.0
    q_agent.q_table[state][1] = 10.0
    q_agent.q_table[state][2] = 30.0

    chosen_action = q_agent.choose_action(state, available_actions)

    mock_random.assert_called_once()
    # Check that np.random.choice was called with the tied best actions [0, 2]
    mock_choice.assert_called_once()
    call_args = mock_choice.call_args.args[0]
    assert sorted(call_args) == [0, 2]
    # Check the returned action is the one mocked by choice
    assert chosen_action == 2

@patch('numpy.random.random')
def test_choose_action_exploit_with_unavailable_action(
    mock_random, q_agent: QLearningAgent
):
    """Test exploitation ignores unavailable actions."""
    mock_random.return_value = 0.9 # Exploit

    state = (10, 5, 0)
    available_actions = [0, 2] # Action 1 (highest Q) is NOT available

    # Setup Q-table: action 1 has the highest Q-value overall
    q_agent.q_table[state][0] = 10.0
    q_agent.q_table[state][1] = 20.0
    q_agent.q_table[state][2] = 15.0 # Highest Q among available actions

    chosen_action = q_agent.choose_action(state, available_actions)

    mock_random.assert_called_once()
    assert chosen_action == 2 # Should choose action 2 (max Q among available)

def test_choose_action_no_available_actions(q_agent: QLearningAgent):
    """Test that ValueError is raised if no actions are available."""
    state = (1, 1, 1)
    # Test with empty list
    with pytest.raises(ValueError, match="No available actions."):
        q_agent.choose_action(state, [])

# --- Test update() --- #

def test_q_agent_update(q_agent: QLearningAgent):
    """Test the Q-table update logic."""
    state = (5, 10, 0) # weeks_rem, inv, current_discount_idx
    action = 1
    reward = 50.0
    next_state = (4, 8, 1) # Next state after taking action 1
    done = False

    # Set known Q-values for next_state
    q_agent.q_table[next_state][0] = 100.0
    q_agent.q_table[next_state][1] = 120.0 # Max Q for next state
    q_agent.q_table[next_state][2] = 110.0
    max_next_q = 120.0

    # Initial Q-value for (state, action)
    initial_q = q_agent.q_table[state][action] # Should be 0.0 initially
    assert initial_q == 0.0

    # Calculate expected new Q-value
    alpha = q_agent.learning_rate # 0.1
    gamma = q_agent.discount_factor # 0.9
    td_target = reward + gamma * max_next_q
    # td_target = 50.0 + 0.9 * 120.0 = 50.0 + 108.0 = 158.0
    td_error = td_target - initial_q # 158.0 - 0.0 = 158.0
    expected_new_q = initial_q + alpha * td_error
    # expected_new_q = 0.0 + 0.1 * 158.0 = 15.8

    # Perform update
    q_agent.update(state, action, reward, next_state, done)

    # Verify the Q-value was updated correctly
    assert q_agent.q_table[state][action] == pytest.approx(expected_new_q)
    # Verify other actions for the state remain unchanged (still 0.0)
    assert q_agent.q_table[state][0] == 0.0
    assert q_agent.q_table[state][2] == 0.0

def test_q_agent_update_when_done(q_agent: QLearningAgent):
    """Test Q-table update when the episode is done."""
    state = (1, 2, 1) # Last week state
    action = 0
    reward = 25.0 # Final reward including salvage perhaps
    next_state = (0, 1, 0) # Terminal state representation
    done = True

    # Q-values for next_state don't matter when done=True
    q_agent.q_table[next_state][0] = 1000.0 # Should be ignored

    initial_q = q_agent.q_table[state][action]
    assert initial_q == 0.0

    # Calculate expected new Q-value (max_next_q is 0 because done=True)
    alpha = q_agent.learning_rate # 0.1
    gamma = q_agent.discount_factor # 0.9
    td_target = reward + gamma * 0.0 # max_next_q = 0
    # td_target = 25.0
    td_error = td_target - initial_q # 25.0 - 0.0 = 25.0
    expected_new_q = initial_q + alpha * td_error
    # expected_new_q = 0.0 + 0.1 * 25.0 = 2.5

    # Perform update
    q_agent.update(state, action, reward, next_state, done)

    # Verify the Q-value was updated correctly
    assert q_agent.q_table[state][action] == pytest.approx(expected_new_q)

# --- Test decay_exploration() --- #

def test_q_agent_decay_exploration(q_agent: QLearningAgent):
    """Test the exploration rate decay logic."""
    initial_epsilon = q_agent.exploration_rate # 0.5 from fixture
    decay = q_agent.exploration_decay # 0.99 from fixture
    min_epsilon = q_agent.min_exploration_rate # 0.01 from fixture

    # First decay
    q_agent.decay_exploration()
    expected_epsilon_1 = initial_epsilon * decay
    assert q_agent.exploration_rate == pytest.approx(expected_epsilon_1)

    # Second decay
    q_agent.decay_exploration()
    expected_epsilon_2 = expected_epsilon_1 * decay
    assert q_agent.exploration_rate == pytest.approx(expected_epsilon_2)

    # Decay until minimum is reached
    q_agent.exploration_rate = min_epsilon * 1.01 # Slightly above minimum
    q_agent.decay_exploration() # Decay should make it less than min
    assert q_agent.exploration_rate == min_epsilon # Should be clamped at min

    # Decay when already at minimum
    q_agent.exploration_rate = min_epsilon
    q_agent.decay_exploration() # Should have no effect
    assert q_agent.exploration_rate == min_epsilon

# --- Test get_policy() --- #

def test_get_policy(q_agent: QLearningAgent):
    """Test extracting the greedy policy from the Q-table."""
    state1 = (10, 15, 0)
    state2 = (5, 5, 1)
    state3 = (1, 1, 2)

    # Populate Q-table
    q_agent.q_table[state1][0] = 10
    q_agent.q_table[state1][1] = 20 # Best for state1
    q_agent.q_table[state1][2] = 15

    q_agent.q_table[state2][0] = -5
    q_agent.q_table[state2][1] = -2
    q_agent.q_table[state2][2] = 0 # Best for state2

    # State 3 only has Q-values for action 0 (defaultdict)
    q_agent.q_table[state3][0] = 5 # Best (and only) for state3

    # Get policy
    policy = q_agent.get_policy()

    # Verify policy dictionary
    assert isinstance(policy, dict)
    assert len(policy) == 3 # Only states with entries in q_table are included
    assert policy[state1] == 1 # Action index with max Q-value
    assert policy[state2] == 2
    assert policy[state3] == 0

# --- Test get_q_table_df() --- #

def test_get_q_table_df(q_agent: QLearningAgent, agent_config: QLearningAgentConfig):
    """Test converting the Q-table to a pandas DataFrame."""
    state1 = (2, 10, 0)
    state2 = (1, 5, 1)
    # Populate Q-table
    q_agent.q_table[state1][0] = 1.1
    q_agent.q_table[state1][1] = 1.2 # Best
    q_agent.q_table[state1][2] = 1.0
    q_agent.q_table[state2][0] = 2.5 # Best
    q_agent.q_table[state2][1] = 2.0
    q_agent.q_table[state2][2] = -1.0

    # Mock discount map (assuming 3 actions from fixture)
    discount_map = [0.0, 0.1, 0.5]
    assert len(discount_map) == agent_config.action_space_size

    df = q_agent.get_q_table_df(discount_map)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2 # Number of states in Q-table

    # Check columns
    expected_cols = [
        "weeks_remaining", "inventory_level", "current_discount_idx",
        "Q(disc=0%)", "Q(disc=10%)", "Q(disc=50%)",
        "best_action_idx", "best_discount"
    ]
    assert list(df.columns) == expected_cols

    # Check sorting (weeks_remaining DESC, inventory ASC, discount_idx ASC)
    assert df.iloc[0]["weeks_remaining"] == 2
    assert df.iloc[1]["weeks_remaining"] == 1

    # Check values for state1 (should be first row after sort)
    row1 = df.iloc[0]
    assert row1["weeks_remaining"] == state1[0]
    assert row1["inventory_level"] == state1[1]
    assert row1["current_discount_idx"] == state1[2]
    assert row1["Q(disc=0%)"] == pytest.approx(1.1)
    assert row1["Q(disc=10%)"] == pytest.approx(1.2)
    assert row1["Q(disc=50%)"] == pytest.approx(1.0)
    assert row1["best_action_idx"] == 1
    assert row1["best_discount"] == 10 # 0.1 * 100

    # Check values for state2 (should be second row)
    row2 = df.iloc[1]
    assert row2["best_action_idx"] == 0
    assert row2["best_discount"] == 0 # 0.0 * 100

def test_get_q_table_df_empty(q_agent: QLearningAgent):
    """Test getting DataFrame when Q-table is empty."""
    assert q_agent.q_table == {}
    df = q_agent.get_q_table_df([0.0, 0.1, 0.5])
    assert df is None

# --- Test save() / load() --- #

def test_q_agent_save_load(q_agent: QLearningAgent, agent_config: QLearningAgentConfig, tmp_path: Path):
    """Test saving and loading the Q-learning agent state."""
    # Modify the agent state somewhat
    state1 = (5, 10, 0)
    state2 = (4, 5, 1)
    q_agent.update(state1, 1, 50.0, state2, False)
    q_agent.update(state2, 0, 30.0, (3, 3, 0), False)
    q_agent.decay_exploration()
    q_agent.decay_exploration()

    # Store original values for comparison
    original_q_table = dict(q_agent.q_table) # Convert defaultdict for comparison
    original_epsilon = q_agent.exploration_rate
    original_lr = q_agent.learning_rate # Should not change, but check anyway

    save_file = tmp_path / "q_agent.pkl"

    # Save
    q_agent.save(str(save_file))
    assert save_file.exists()

    # Create a new agent and load
    new_agent = QLearningAgent(config=agent_config)
    # Verify state is different initially
    assert new_agent.exploration_rate != original_epsilon
    assert len(new_agent.q_table) == 0

    new_agent.load(str(save_file))

    # Verify loaded state matches original state
    assert new_agent.learning_rate == original_lr
    assert new_agent.discount_factor == agent_config.discount_factor # Loaded from state dict
    assert new_agent.exploration_rate == original_epsilon
    assert new_agent.min_exploration_rate == agent_config.min_exploration_rate
    assert new_agent.exploration_decay == agent_config.exploration_decay
    assert new_agent.action_space_size == agent_config.action_space_size

    # Compare Q-tables (handle defaultdict vs dict)
    assert isinstance(new_agent.q_table, defaultdict)
    loaded_q_table_dict = dict(new_agent.q_table)
    # Compare keys first
    assert set(loaded_q_table_dict.keys()) == set(original_q_table.keys())
    # Compare numpy arrays within the dict using np.testing
    for state in original_q_table:
        np.testing.assert_array_equal(loaded_q_table_dict[state], original_q_table[state])

# Placeholder tests
# def test_q_agent_save_load(): ... 