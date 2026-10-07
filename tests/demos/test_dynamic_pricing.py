import runpy

import numpy as np
import pytest

from agents.qlearning import QLearningAgent
from config.config import DynamicPricingMDPConfig, QLearningAgentConfig
from demos.dynamic_pricing import train_agent
from environments.mdp import DynamicPricingMDP


def test_library_training_and_q_updates_are_seeded():
    results = []
    for _ in range(2):
        np.random.seed(7)
        env = DynamicPricingMDP(DynamicPricingMDPConfig())
        agent = QLearningAgent(QLearningAgentConfig())
        results.append(train_agent(env, agent, num_episodes=2))
    assert results[0] == results[1]
    state, next_state = (2, 10, 0), (1, 8, 1)
    agent.learning_rate, agent.discount_factor = 0.5, 0.9
    agent.q_table[state][0] = 4
    agent.q_table[next_state][1] = 6
    agent.update(state, 0, 3, next_state, False)
    assert agent.q_table[state][0] == pytest.approx(6.2)
    agent.update(state, 0, 2, next_state, True)
    assert agent.q_table[state][0] == pytest.approx(4.1)


def test_module_runs_the_training_demo(capsys):
    runpy.run_path("demos/dynamic_pricing.py", run_name="__main__")
    assert "Episode 10000/10000" in capsys.readouterr().out
