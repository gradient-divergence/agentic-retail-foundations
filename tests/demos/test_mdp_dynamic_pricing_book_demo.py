import random
import subprocess
import sys

import numpy as np
import pytest

from demos.mdp_dynamic_pricing_book_demo import DynamicPricingMDP, QLearningAgent, train_agent


def test_policy_keeps_fractional_discount_states():
    agent = QLearningAgent()
    state = (2, 10, 0.2)
    agent.q_table[state][1] = 5
    assert agent.get_policy().policy == {state: 1}


def test_short_verbose_training_is_seeded_and_preserves_update_rule(capsys):
    results = []
    for _ in range(2):
        random.seed(7)
        np.random.seed(7)
        results.append(train_agent(DynamicPricingMDP(), QLearningAgent(), num_episodes=2, verbose=True))
    assert results[0] == results[1]
    assert "Episode 2/2" in capsys.readouterr().out
    agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9)
    state, next_state = (2, 10, 0.0), (1, 8, 0.2)
    agent.q_table[state][0] = 4
    agent.q_table[next_state][1] = 6
    assert agent.update(state, 0, 3, next_state, [1], False) == pytest.approx(4.4)
    assert agent.q_table[state][0] == pytest.approx(6.2)
    agent.update(state, 0, 2, next_state, [], True)
    assert agent.q_table[state][0] == pytest.approx(4.1)


def test_terminal_environment_cannot_pay_salvage_twice():
    env = DynamicPricingMDP(season_length_weeks=1, base_demand=0)
    env.reset()
    _, reward, done, _ = env.step(0)
    assert done
    assert reward == 100 * (15 - 0.5)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    assert env.current_week == 1


@pytest.mark.parametrize("action", [-1, 4])
def test_invalid_action_is_rejected_before_mutating_state(action):
    env = DynamicPricingMDP()
    state = env.reset()
    with pytest.raises(ValueError, match="action"):
        env.step(action)
    assert env.current_inventory == state[1]
    assert env.current_week == 0


def test_module_runs_the_training_demo():
    result = subprocess.run(
        [sys.executable, "-m", "demos.mdp_dynamic_pricing_book_demo"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Episode 500/500" in result.stdout
