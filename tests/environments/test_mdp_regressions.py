import pytest

from config.config import DynamicPricingMDPConfig
from environments import mdp


def test_terminal_environment_cannot_pay_salvage_twice():
    env = mdp.DynamicPricingMDP(DynamicPricingMDPConfig(season_length_weeks=1, base_demand=0))
    env.reset()
    _, reward, done, _ = env.step(0)
    assert done
    assert reward == 100 * (15 - 0.5)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    assert env.current_week == 1
