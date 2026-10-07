import subprocess
import sys

import pytest

from demos.capstone_mdp_simulator_demo import Action, InventoryRiskMDP, SupplierStatus


def test_module_prints_all_eight_weeks():
    result = subprocess.run(
        [sys.executable, "-m", "demos.capstone_mdp_simulator_demo"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 8
    assert "week=8 " in result.stdout


def test_terminal_capstone_requires_reset():
    env = InventoryRiskMDP(horizon_weeks=1)
    env.step(Action.WAIT)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(Action.REORDER)
    assert env.state.week == 1
    assert env.state.pipeline == []


def test_state_tracks_pending_deliveries_without_changing_previous_snapshots(monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.5)
    monkeypatch.setattr("random.gauss", lambda mean, sigma: 0)
    env = InventoryRiskMDP()
    initial = env.reset()
    ordered, _, _, _ = env.step(Action.REORDER)
    waiting, _, _, _ = env.step(Action.WAIT)
    delivered, _, _, _ = env.step(Action.WAIT)

    assert initial.pipeline == []
    assert ordered.pipeline == [(2, 80)]
    assert waiting.pipeline == [(1, 80)]
    assert delivered.pipeline == []
    assert delivered.inventory_on_hand == initial.inventory_on_hand + 80
    assert env.reset().pipeline == []


@pytest.mark.parametrize("initial,delayed", [(SupplierStatus.STABLE, 10), (SupplierStatus.DELAYED, 82)])
def test_supplier_transition_branches_form_a_complete_probability_distribution(monkeypatch, initial, delayed):
    counts = dict.fromkeys(SupplierStatus, 0)
    for first in range(10):
        for second in range(10):
            draws = iter([(first + 0.5) / 10, (second + 0.5) / 10])
            monkeypatch.setattr("random.random", lambda draws=draws: next(draws))
            env = InventoryRiskMDP()
            env.state.supplier_status = initial
            state, _, _, _ = env.step(Action.WAIT)
            counts[state.supplier_status] += 1
    assert sum(counts.values()) == 100
    assert counts[SupplierStatus.DELAYED] == delayed
