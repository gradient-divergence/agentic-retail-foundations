import os
import tempfile

from agents.qlearning import QLearningAgent
from config.config import DynamicPricingMDPConfig, QLearningAgentConfig
from environments.mdp import DynamicPricingMDP


def make_env_agent():
    env_config = DynamicPricingMDPConfig(initial_inventory=20, season_length_weeks=4)
    agent_config = QLearningAgentConfig(action_space_size=len(env_config.available_discounts))
    env = DynamicPricingMDP(env_config)
    agent = QLearningAgent(agent_config)
    return env, agent


def test_environment_reset_and_step():
    """Test that the environment resets and steps correctly."""
    env, _ = make_env_agent()
    state = env.reset()
    assert isinstance(state, tuple)
    next_state, reward, done, info = env.step(0)
    assert isinstance(next_state, tuple)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_agent_choose_action_and_update():
    """Test that the agent can choose actions and update Q-values."""
    env, agent = make_env_agent()
    state = env.reset()
    available_actions = env.get_available_actions()
    action = agent.choose_action(state, available_actions)
    assert action in available_actions
    next_state, reward, done, info = env.step(action)
    agent.update(state, action, reward, next_state, done)
    # Q-value should be updated
    assert agent.q_table[state][action] != 0.0


def test_training_loop_short():
    """Test a short training loop end-to-end."""
    env, agent = make_env_agent()
    num_episodes = 10
    for _episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, env.get_available_actions())
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.decay_exploration()
    # After training, Q-table should have some nonzero values
    q_vals = [v for arr in agent.q_table.values() for v in arr]
    assert any(abs(v) > 1e-6 for v in q_vals)


def test_environment_serialization():
    """Test saving and loading environment state."""
    env, _ = make_env_agent()
    env.step(0)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        env.save(tmp.name)
        path = tmp.name
    env2, _ = make_env_agent()
    env2.load(path)
    assert env2.current_week == env.current_week
    os.remove(path)


def test_agent_serialization():
    """Test saving and loading agent state and Q-table."""
    _, agent = make_env_agent()
    # Simulate some updates
    state = (4, 10, 0)
    agent.q_table[state][0] = 42.0
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        agent.save(tmp.name)
        path = tmp.name
    agent2 = QLearningAgent(agent.config)
    agent2.load(path)
    assert agent2.q_table[state][0] == 42.0
    os.remove(path)
