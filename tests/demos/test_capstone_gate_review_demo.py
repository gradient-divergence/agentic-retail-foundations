import importlib
import random

import pytest


def test_gate_review_demo_exists():
    assert importlib.util.find_spec("demos.capstone_gate_review_demo") is not None


@pytest.mark.parametrize(
    "holiday,receipt_day,quantity", [(False, 1, 20), (True, 1, 50), (False, 2, 35), (True, 2, 80)]
)
def test_forecast_drives_agent_quantity_while_rule_keeps_80(holiday, receipt_day, quantity):
    from demos.capstone_gate_review_demo import Scenario, simulate_episode

    scenario = Scenario(1, holiday, 30, 20, receipt_day, 0, False)
    assert simulate_episode(scenario, "agent")["inventory_trace"][0] == ("confirmed", 20, quantity)
    assert simulate_episode(scenario, "rule")["inventory_trace"][0] == ("confirmed", 20, 80)


def test_agent_nets_on_order_and_never_uses_realized_demand():
    from demos.capstone_gate_review_demo import Scenario, proposed_order_quantity

    ordinary = Scenario(1, False, 15, 10, 1, 0, False)
    shock = Scenario(1, False, 45, 40, 1, 0, False)
    assert proposed_order_quantity(ordinary, "agent", on_hand=20, on_order=10) == 10
    assert proposed_order_quantity(shock, "agent", on_hand=20, on_order=10) == 10
    assert proposed_order_quantity(shock, "agent", on_hand=30, on_order=10) == 0
    assert proposed_order_quantity(shock, "agent", on_hand=50, on_order=10) == 0
    assert proposed_order_quantity(shock, "rule", on_hand=30, on_order=0) == 0
    assert proposed_order_quantity(shock, "rule", on_hand=20, on_order=10) == 0
    with pytest.raises(ValueError):
        proposed_order_quantity(shock, "unknown")


@pytest.mark.parametrize("quantity,policy_stops", [(66, 0), (67, 1)])
def test_exact_quantity_passes_policy_and_retains_stage_1_buyer_review(quantity, policy_stops):
    from demos.capstone_gate_review_demo import Scenario, approved_order

    controls = dict(policy_stops=0, approvals=0)
    assert approved_order(Scenario(1, False, 30, 20, 1, 0, False), quantity, controls)
    assert controls == dict(policy_stops=policy_stops, approvals=1)


def test_episode_accounts_for_markdown_writeoff_and_idempotent_receipt():
    from demos.capstone_gate_review_demo import Scenario, simulate_episode

    scenario = Scenario(1, True, 60, 20, 1, 1, True)
    agent = simulate_episode(scenario, "agent")
    rule = simulate_episode(scenario, "rule")
    assert agent["fill_rate"] == 1
    assert agent["leftover_units"] == 10
    assert agent["leftover_cost"] == 750
    assert agent["post_sell_through"] == 1
    assert agent["waste_units"] == 0
    assert agent["contribution"] == 1317.5
    assert rule["contribution"] == -345
    assert agent["controls"] == {
        "policy_stops": 0,
        "approvals": 1,
        "retries": 1,
        "duplicates_ignored": 1,
        "timeouts": 1,
        "exhausted": 0,
        "recoveries": 1,
        "unauthorized": 0,
    }
    assert agent["inventory_trace"][0] == ("confirmed", 20, 50)
    assert agent["inventory_trace"][1] == ("received", 70, 0)


def test_supplier_delay_cannot_fulfill_holiday_demand():
    from demos.capstone_gate_review_demo import Scenario, simulate_episode

    outcome = simulate_episode(Scenario(2, True, 60, 20, 3, 0, False), "agent")
    assert outcome["fill_rate"] == pytest.approx(1 / 3)
    assert outcome["leftover_units"] == 80  # Confirmed incoming stock is capital too.
    assert outcome["post_sell_through"] == 0.25
    assert outcome["waste_units"] == 60


def test_retry_exhaustion_never_creates_an_order_or_receipt():
    from demos.capstone_gate_review_demo import Scenario, simulate_episode

    outcome = simulate_episode(Scenario(3, True, 60, 20, 1, 3, True), "agent")
    assert outcome["fill_rate"] == pytest.approx(1 / 3)
    assert outcome["controls"]["retries"] == 2
    assert outcome["controls"]["timeouts"] == 3
    assert outcome["controls"]["exhausted"] == 1
    assert outcome["controls"]["recoveries"] == 0
    assert outcome["controls"]["duplicates_ignored"] == 0
    assert outcome["inventory_trace"] == []
    assert outcome["contribution"] == 488.25


def test_paired_interval_uses_episode_differences():
    from demos.capstone_gate_review_demo import paired_interval

    mean, low, high = paired_interval([1, 2, 3])
    assert mean == 2
    assert low == pytest.approx(-0.4841377117)
    assert high == pytest.approx(4.4841377117)
    with pytest.raises(ValueError):
        paired_interval([1])


def test_review_is_paired_and_independent_of_global_randomness():
    from demos.capstone_gate_review_demo import run_review

    before = random.getstate()
    review = run_review()
    assert random.getstate() == before
    assert len(review["baseline"]) == 32
    assert len(review["agent"]) == len(review["rule"]) == 64
    assert [r["seed"] for r in review["agent"]] == [r["seed"] for r in review["rule"]]
    assert review["verdict"] == "hold"
    assert review["failed_gates"] == ["availability", "workload"]
    assert all(review["gates"][name] for name in ("value", "capital", "waste", "controls", "window"))


def test_gate_acceptance_can_advance_and_checks_uncertainty():
    from demos.capstone_gate_review_demo import Scenario, evaluate_gates

    # Literal counterfactual evidence: value improvement with all risk bounds met.
    paired = {
        "contribution": (10, 5, 15),
        "fill_rate": (0.1, 0.05, 0.15),
        "leftover_cost": (-100, -150, -50),
        "waste_units": (-2, -3, -1),
        "workload_minutes": (-1, -2, 0),
    }
    controls = dict(
        policy_stops=2,
        approvals=2,
        retries=3,
        timeouts=4,
        exhausted=1,
        recoveries=1,
        duplicates_ignored=1,
        unauthorized=0,
    )
    scenarios = [Scenario(1, True, 60, 20, 3, 1, True), Scenario(2, False, 30, 20, 1, 3, False)]
    assert all(evaluate_gates(paired, controls, scenarios).values())
    for metric, gate in (
        ("contribution", "value"),
        ("fill_rate", "availability"),
        ("leftover_cost", "capital"),
        ("waste_units", "waste"),
        ("workload_minutes", "workload"),
    ):
        uncertain = {**paired, metric: (0, -1, 1)}
        assert not evaluate_gates(uncertain, controls, scenarios)[gate]
    assert not evaluate_gates(paired, {**controls, "recoveries": 0}, scenarios)["controls"]
    assert not evaluate_gates(paired, {**controls, "unauthorized": 1}, scenarios)["controls"]
    assert not evaluate_gates(paired, controls, scenarios[:1])["window"]


def test_printed_gate_review_is_pinned(capsys):
    from demos.capstone_gate_review_demo import main

    main()
    assert capsys.readouterr().out == (
        "SIMULATION ONLY: SKU-123 Stage 1 -> 2 gate review (USD)\n"
        "Baseline: 32 cycles, seeds 1..32; paired: 64 cycles, seeds 1001..1064.\n"
        "Price=100.00; unit cost=75.00; markdown=60.00; terminal salvage=0.00.\n"
        "Rule: 80 units below a 30-unit point; agent: forecast over L+R + 10, net of stock.\n"
        "Forecast=30/60 per 2 days; L=1 stable/2 delayed days; R=1 day; opening on-order=0.\n"
        "Episode means; leftover includes confirmed incoming stock.\n"
        "baseline-rule: fill=83.80%; leftover=49.44 units (USD 3707.81); net=USD -1638.66; "
        "post-sell=38.29%; waste=29.00 units; workload=6.38 min\n"
        "paired-rule: fill=79.71%; leftover=46.52 units (USD 3488.67); net=USD -1796.13; "
        "post-sell=30.37%; waste=30.48 units; workload=6.47 min\n"
        "paired-agent: fill=78.77%; leftover=21.19 units (USD 1589.06); net=USD -149.68; "
        "post-sell=46.69%; waste=9.28 units; workload=7.47 min\n"
        "Paired agent-minus-rule differences [95% Student-t interval]:\n"
        "fill_rate: -0.94 pp [-1.79, -0.08]\n"
        "leftover_units: -25.33 units [-31.54, -19.11]\n"
        "leftover_cost: -1899.61 USD [-2365.77, -1433.45]\n"
        "contribution: +1646.45 USD [+1199.17, +2093.73]\n"
        "post_sell_through: +16.32 pp [+6.07, +26.56]\n"
        "waste_units: -21.20 units [-27.06, -15.35]\n"
        "workload_minutes: +1.00 min [+1.00, +1.00]\n"
        "baseline controls: policy_stops=32; approvals=32; retries=22; duplicates_ignored=17; "
        "timeouts=28; exhausted=6; recoveries=10; unauthorized=0\n"
        "rule controls: policy_stops=64; approvals=64; retries=47; duplicates_ignored=16; "
        "timeouts=65; exhausted=18; recoveries=11; unauthorized=0\n"
        "agent controls: policy_stops=15; approvals=64; retries=47; duplicates_ignored=16; "
        "timeouts=65; exhausted=18; recoveries=11; unauthorized=0\n"
        "Gates: value=pass; availability=fail; capital=pass; waste=pass; workload=fail; "
        "controls=pass; window=pass\n"
        "VERDICT: hold; failed gates: availability, workload.\n"
        "Synthetic owner: head of replenishment; buyer approval retained.\n"
        "Real advancement still requires field evidence and owner sign-off.\n"
    )
