"""Behavior regressions for the third-edition technical review."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


@pytest.mark.parametrize("value", [True, "9.99", float("inf"), float("nan")])
def test_price_proposals_reject_coercion_and_nonfinite_values(value):
    from demos.structured_output_schema_demo import propose_price_update, validate_tool_payload

    with pytest.raises(ValidationError):
        validate_tool_payload({**propose_price_update().model_dump(), "new_price": value})


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["a", "b"])
def test_preference_pairs_reject_invalid_scores(score, side):
    from demos.preference_pair_builder_demo import EvalCase, to_preference_pairs

    scores = {"a": 0.9, "b": 0.5, side: score}
    with pytest.raises(ValueError, match="finite"):
        to_preference_pairs([EvalCase("prompt", "A", "B", scores["a"], scores["b"])])


def test_cost_ceiling_denies_unknown_cost():
    from agents.raom_minimal import ActionRequest, MaxCostRule

    assert MaxCostRule(1).evaluate(ActionRequest(tool_name="charge", payload={})) == ["cost_unknown"]


@pytest.mark.parametrize("price,status", [(10.0, "optimal"), (10.5, "high")])
def test_price_ending_preserves_direction_and_reports_realized_change(price, status):
    from agents.ooda import OODAPricingAgent
    from models.pricing import PricingProduct

    agent = OODAPricingAgent()
    agent.update_products({"A": PricingProduct("A", "Tee", "apparel", 5, price, 8, 15)})
    result = agent.decide("A", {"inventory_status": status, "sales_assessment": "normal"})
    assert result["new_price"] <= price
    if status == "optimal":
        assert result["new_price"] == price
    assert result["capped_change_pct"] == pytest.approx(100 * (result["new_price"] / price - 1))


@pytest.mark.parametrize("excerpt", [False, True])
def test_conflicting_price_bounds_are_rejected(excerpt):
    from agents.ooda import OODAPricingAgent
    from demos.ooda_pricing_decide_excerpt import DecisionWeights, OrientationSignal, decide_price
    from models.pricing import PricingProduct

    with pytest.raises(ValueError, match="feasible"):
        if excerpt:
            decide_price(5, OrientationSignal(), DecisionWeights(), 10, 20, 5)
        else:
            agent = OODAPricingAgent()
            agent.update_products({"A": PricingProduct("A", "Tee", "apparel", 2, 5, 10, 20)})
            agent.decide("A", {"inventory_status": "optimal"})


def test_capstone_state_calendar_risk_and_contribution(monkeypatch):
    from demos.capstone_mdp_simulator_demo import Action, InventoryRiskMDP

    monkeypatch.setattr("random.random", lambda: 0.5)
    monkeypatch.setattr("random.gauss", lambda mean, sigma: 1)
    env = InventoryRiskMDP(initial_inventory=10, promo_weeks=(0, 2))
    assert env.state.risk_level == "high"
    assert env.state.promo_active
    for week in (1, 2, 3):
        state, reward, _, info = env.step(Action.WAIT)
        assert state.promo_active == (week == 2)
        assert info.contribution == info.sales * env.margin_per_unit
        assert reward == info.contribution - info.holding_cost - info.penalty - info.action_cost


@pytest.mark.parametrize("cost_gradient", [-2.0, 2.0])
def test_constrained_update_uses_cost_derivative_and_signed_dual_update(cost_gradient):
    from demos.constrained_rl_pricing_demo import constrained_update

    price, multiplier = constrained_update(25, 0, 0.4, 0.6, 1, cost_gradient=cost_gradient)
    assert multiplier == pytest.approx(0.98)
    assert price == pytest.approx(25 - 0.05 * 0.98 * cost_gradient)


def test_responses_executes_multiple_tool_rounds_and_fails_on_exhaustion():
    from demos.virtual_shopping_responses_demo import run_demo

    calls = []

    def create(**kwargs):
        calls.append(list(kwargs["input"]))
        n = len(calls)
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="recommend_outfit",
                    arguments='{"style":"summer"}',
                    call_id=str(n),
                )
            ]
            if n < 3
            else [],
            output_text="Done" if n == 3 else "",
        )

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    assert run_demo(client=client) == "Done"
    assert [
        x["call_id"] for x in calls[-1] if isinstance(x, dict) and x.get("type") == "function_call_output"
    ] == ["1", "2"]
    calls.clear()
    with pytest.raises(RuntimeError, match="round"):
        run_demo(client=client, max_tool_rounds=1)


def test_hitl_exact_twenty_percent_boundary():
    from demos.hitl_approval_api_demo import PriceProposal, propose_price

    result = propose_price(
        PriceProposal(product_id=1, current_price=10.05, suggested_price=8.04, requester="agent")
    )
    assert result.status == "auto_approved"


def test_weak_eval_exposes_false_positive_and_false_negative():
    from demos.agent_eval_hook_demo import AgentRun, eval_hook

    def run(answer):
        return AgentRun("trace", "Can I return this?", answer, False, 1, datetime.now(timezone.utc))

    assert eval_hook(run("policy: purple elephants")).passed
    assert not eval_hook(run("Returns accepted within 30 days with receipt.")).passed


def test_price_proposal_rejects_fractional_minor_units():
    from demos.structured_output_schema_demo import propose_price_update, validate_tool_payload

    with pytest.raises(ValidationError, match="minor unit"):
        validate_tool_payload({**propose_price_update().model_dump(), "new_price": 1.001})


@pytest.mark.parametrize(
    "price,status", [(8.04, "auto_approved"), (8.03, "pending"), (8.05, "auto_approved")]
)
def test_hitl_one_cent_boundaries(price, status):
    from demos.hitl_approval_api_demo import PriceProposal, propose_price

    assert (
        propose_price(
            PriceProposal(product_id=1, current_price=10.05, suggested_price=price, requester="planner")
        ).status
        == status
    )


def test_hitl_rejects_unsupported_money_precision():
    from demos.hitl_approval_api_demo import PriceProposal

    with pytest.raises(ValidationError, match="minor unit"):
        PriceProposal(product_id=1, current_price=10.051, suggested_price=8.04, requester="planner")


@pytest.mark.parametrize(
    "name,args",
    [("delete_order", "{}"), ("recommend_outfit", "not json"), ("recommend_outfit", '{"style":42}')],
)
def test_responses_rejects_unsupported_or_malformed_calls(name, args):
    from demos.virtual_shopping_responses_demo import run_demo

    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                output=[SimpleNamespace(type="function_call", name=name, arguments=args, call_id="one")],
                output_text="",
            )
        )
    )
    with pytest.raises(ValueError):
        run_demo(client=client)


@pytest.mark.parametrize("cost,allowed", [(0, True), (50, True), (50.01, False)])
def test_cost_ceiling_zero_and_exact_boundaries(cost, allowed):
    from agents.raom_minimal import ActionRequest, MaxCostRule

    reasons = MaxCostRule(50).evaluate(ActionRequest(tool_name="tool", payload={}, estimated_cost=cost))
    assert (not reasons) == allowed


def test_valid_preferences_and_ties_remain_meaningful():
    from demos.preference_pair_builder_demo import EvalCase, to_preference_pairs

    result = to_preference_pairs([EvalCase("valid", "A", "B", 0.9, 0.2), EvalCase("tie", "C", "D", 0.5, 0.5)])
    assert len(result) == 1 and result[0].chosen == "A" and result[0].rejected == "B"


def test_constrained_demo_exercises_clip_penalty_and_dual_recovery(capsys):
    from demos.constrained_rl_pricing_demo import run_demo

    run_demo()
    rows = capsys.readouterr().out.splitlines()
    multipliers = [float(row.split("lambda=")[1]) for row in rows]
    assert max(multipliers) > 0
    assert multipliers[-1] < max(multipliers)
    assert "price=32.00" in rows[0]


@pytest.mark.parametrize("text", ["", "   ", None])
def test_responses_requires_nonempty_terminal_text(text):
    from demos.virtual_shopping_responses_demo import run_demo

    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(output=[], output_text=text))
    )
    with pytest.raises(RuntimeError, match="text"):
        run_demo(client=client)
