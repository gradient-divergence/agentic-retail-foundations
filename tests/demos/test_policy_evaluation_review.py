import random
from copy import deepcopy

from demos.capstone_mdp_simulator_demo import InventoryRiskMDP
from demos.capstone_policy_comparison_demo import compare_policies, plan_action
from demos.mdp_dynamic_pricing_book_demo import DynamicPricingMDP, evaluate_policy


def test_planning_does_not_consume_evaluation_noise_or_mutate_environment():
    env = InventoryRiskMDP()
    state = deepcopy(env.state)
    noise = random.getstate()
    plan_action(env, env.state)
    assert env.state == state
    assert random.getstate() == noise


def test_frozen_policy_evaluation_replays_and_preserves_policy():
    env = DynamicPricingMDP(season_length_weeks=3)
    policy = {(3, 100, 0.0): 1}
    snapshot = dict(policy)
    result = evaluate_policy(env, policy)
    assert result == evaluate_policy(env, policy)
    assert policy == snapshot
    assert result["learned"][0]["trace"][0][1] == 1
    assert result["full_price"][0]["trace"][0][1] == 0


def test_all_three_policies_share_heldout_seeds_and_denominators():
    result = compare_policies(seeds=(1001, 1002, 1003), episodes=20)
    assert result["decision"] in {"retain_rule", "promote"}
    for record in result["policies"].values():
        assert len(record["runs"]) == 3
        assert record["policy_violations"] == 0
        assert record["fill_rate"] == sum(r["sales_units"] for r in record["runs"]) / sum(
            r["demand_units"] for r in record["runs"]
        )
    replay = compare_policies(seeds=(1001, 1002, 1003), episodes=20)
    for name in result["policies"]:
        assert result["policies"][name]["runs"] == replay["policies"][name]["runs"]


def test_governed_episode_requires_approval_and_deduplicates_receipts():
    from demos.capstone_scaffold_demo import run_governed_episode

    result = run_governed_episode()
    assert result["evaluation_passed"]
    assert result["on_hand"] == 100 and result["receipt_count"] == 1
    assert result["events"] == [
        "risk_lookup",
        "proposals_compared",
        "policy_rejected",
        "buyer_approved_exact_proposal",
        "execution_failed_before_commit",
        "retry_authorized",
        "execution_confirmed",
        "physical_receipt_recorded",
        "duplicate_receipt_ignored",
        "evaluation_passed",
    ]


def test_synthetic_order_episode_completes_after_recovery_without_reallocating_replay():
    import asyncio

    from demos.order_orchestration_demo import run_orchestration_simulation

    trace = asyncio.run(run_orchestration_simulation())
    assert trace[-1] == "order.completed"
    assert trace.count("order.allocated") == 1
    assert (
        trace.index("order.exception")
        < trace.index("inventory.reallocation_requested")
        < trace.index("order.allocated")
    )
