import asyncio
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("status,want", [("low", 29.23), ("high", 28.65)])
def test_small_ooda_moves_are_not_swamped_by_price_endings(status, want):
    from agents.ooda import OODAPricingAgent
    from models.pricing import PricingProduct

    agent = OODAPricingAgent()
    agent.update_products({"P": PricingProduct("P", "P", "A", 12, 29, 19, 39)})
    result = agent.decide("P", {"inventory_status": status})
    assert result["new_price"] == want
    assert abs(result["price_ending_adjustment"]) <= 0.01


def test_short_ooda_demo_replays_after_unrelated_randomness(capsys):
    from demos.ooda_pricing_demo_short import demo_ooda_pricing

    random.seed(1)
    demo_ooda_pricing()
    first = capsys.readouterr().out
    random.seed(12)
    demo_ooda_pricing()
    assert capsys.readouterr().out == first


def test_stockout_value_uses_executed_price():
    from tests.agents.test_bdi_regressions import make_agent

    agent = make_agent(stock=1)
    agent.products["B"] = deepcopy(agent.products["A"])
    agent.inventory["B"] = deepcopy(agent.inventory["A"])
    agent.inventory["B"].current_stock = 100
    agent.sales_data["B"] = deepcopy(agent.sales_data["A"])
    before = agent._evaluate_stockout_prevention()
    agent.products["A"].current_price /= 2
    assert agent._evaluate_stockout_prevention() < before


def test_reorder_uses_decision_time_supplier_and_arrives_within_season(monkeypatch):
    from demos.capstone_mdp_simulator_demo import Action, InventoryRiskMDP, SupplierStatus, baseline_action

    env = InventoryRiskMDP()
    env.state.supplier_status = SupplierStatus.DELAYED
    monkeypatch.setattr(random, "random", lambda: 0.15)
    state, _, _, _ = env.step(Action.REORDER)
    assert state.pipeline[-1] == (3, 80)
    state.week, state.inventory_on_hand, state.pipeline = 6, 0, []
    assert baseline_action(env, state) == Action.WAIT


def test_frozen_pricing_evaluation_uses_selected_discount_factor():
    from demos.mdp_dynamic_pricing_book_demo import DynamicPricingMDP, evaluate_policy

    result = evaluate_policy(DynamicPricingMDP(season_length_weeks=2), {}, discount_factor=0.5)
    trace = result["learned"][0]["trace"]
    assert result["learned"][0]["discounted_return"] == pytest.approx(trace[0][2] + trace[1][2] * 0.5)


def test_policy_comparison_exposes_paired_interval_and_order_cap():
    from demos.capstone_policy_comparison_demo import compare_policies

    result = compare_policies(seeds=(1001, 1002, 1003), episodes=20, max_reorder_units=40)
    assert result["policies"]["rule"]["policy_violations"] > 0
    assert result["guardrail"] == "reorder_quantity<=40"
    assert len(result["paired_planner_minus_rule"]["confidence_interval"]) == 2
    assert "inventory_on_hand" in result["policies"]["learned"]["runs"][0]["trace"][0]


def test_governed_episode_keeps_accepted_stock_incoming_until_receipt():
    from demos.capstone_scaffold_demo import run_governed_episode

    result = run_governed_episode()
    assert result["inventory_trace"] == [
        {"event": "execution_confirmed", "on_hand": 20, "on_order": 80},
        {"event": "physical_receipt_recorded", "on_hand": 100, "on_order": 0},
    ]
    assert result["currency"] == "USD" and result["unit_cost"] == 75


def test_open_order_and_closed_season_prevent_duplicate_restock():
    from demos.inventory_agent_unit_test_demo import DemandForecast, InventoryAgent, InventorySnapshot

    agent = InventoryAgent(safety_stock=10)
    forecast = DemandForecast(demand={"SKU-123": 15})
    assert agent.evaluate_restock(
        InventorySnapshot(stock={"SKU-123": 5}, on_order={"SKU-123": 20}), forecast
    ).orders == {"SKU-123": 0}
    assert agent.evaluate_restock(
        InventorySnapshot(stock={"SKU-123": 5}, season_open=False), forecast
    ).orders == {"SKU-123": 0}


def test_empty_catalog_result_abstains():
    from demos.tool_calling_rag_demo import run_tool_call

    result = run_tool_call("not-in-catalog")
    assert result.sources == [] and result.confidence == 0
    assert result.answer == "No catalog matches; ask for another description."


def test_one_matching_pair_has_no_estimated_interval():
    from demos.promotion_causal_book_demo import PromotionCausalAnalyzer

    rows = pd.DataFrame({"sales_units": [10, 15], "x": [1, 1]})
    result = PromotionCausalAnalyzer._matching_diagnostics(
        [(rows.iloc[1], rows.iloc[0])], rows[["x"]], pd.Series([False, True]), np.array([0.5, 0.5]), 5
    )
    assert result["confidence_interval"] is None


def test_sales_window_identity_does_not_depend_on_sample_payload():
    from tests.demos.test_dynamic_pricing_feedback_book_demo import agent as fixture

    with pytest.MonkeyPatch.context() as mp:
        agent = fixture.__wrapped__(mp)
        agent.window_start_ms, agent.window_end_ms = 0, 3600000
        agent.compute_optimal_price([(1000, 2)])
        agent.compute_optimal_price([(1000, 2), (2000, 3)])
        assert agent.demand_history == [2]
        assert agent.observation_ids == [("sku", 0, 3600000)]


def test_fulfillment_demo_distributes_eligible_ambient_orders(monkeypatch, capsys):
    from demos.store_fulfillment_book_demo import FulfillmentPlanner, demo_fulfillment_system

    monkeypatch.setattr(FulfillmentPlanner, "visualize_plan", lambda self: None)
    planner = demo_fulfillment_system()
    capsys.readouterr()
    assert planner.assignments["A3"]
    assert len({order.order_id for orders in planner.assignments.values() for order in orders}) == 10


def test_sensor_entrypoint_runs_recovery_cycle(capsys):
    from demos.sensor_processor_demo import run_shelf_recovery

    result = asyncio.run(run_shelf_recovery())
    assert result == {"initial_units": 0, "recovered_units": 10, "open_discrepancies": 0, "ledger_writes": 0}


def test_capstone_budgets_block_execution_before_handler():
    from capstone.orchestrator import CapstoneOrchestrator
    from capstone.policies import PolicyEvaluator
    from capstone.schemas import ToolCall
    from capstone.tools import ToolSpec
    from capstone.tracing import TraceContext

    committed = []

    def commit():
        committed.append(1)
        return {"count": len(committed)}

    tool = ToolSpec(
        name="lookup", description="fixture", permission="read", schema={"type": "object"}, handler=commit
    )
    orchestrator = CapstoneOrchestrator([tool], PolicyEvaluator(), max_tool_calls=1, max_tokens=10)
    trace = TraceContext(route="test")
    call = ToolCall(name="lookup", args={})
    assert orchestrator.handle_tool_call(call, trace, token_count=10)[0].status == "ok"
    assert orchestrator.handle_tool_call(call, trace)[0].output == {"error": "budget_exhausted"}
    assert committed == [1]
    assert (
        orchestrator.handle_tool_call(call, TraceContext(route="test"), token_count=11)[0].status == "error"
    )


def test_structured_price_context_is_required_and_validated():
    from pydantic import ValidationError

    from demos.structured_output_schema_demo import propose_price_update, validate_tool_payload

    payload = propose_price_update().model_dump()
    assert payload["currency"] == "USD" and payload["current_price"] == 129
    assert payload["zone"] and payload["channel"] and payload["policy_version"]
    assert payload["effective_from"] < payload["effective_to"]
    for change in ({"currency": "EUR"}, {"effective_to": payload["effective_from"]}):
        with pytest.raises(ValidationError):
            validate_tool_payload({**payload, **change})


def test_inventory_reservation_sale_and_transfer_dispatch_conserve_stock():
    from demos import inventory_event_api_demo as demo

    demo.inventory_state.clear()
    common = dict(product_id="SKU-123", location_id="North", quantity=5)

    async def run():
        await demo.update_inventory_state(demo.InventoryReceived(**common, supplier_id="S"))
        reserved = demo.InventoryReserved(**common, reservation_id="R", order_id="O")
        await demo.update_inventory_state(reserved)
        await demo.update_inventory_state(reserved)
        sale = demo.InventorySold(**common, order_id="O", reservation_id="R")
        await demo.update_inventory_state(sale)
        await demo.update_inventory_state(sale)
        north = demo.inventory_state["SKU-123"]["North"]
        assert (north.quantity_available, north.quantity_reserved) == (0, 0)
        await demo.update_inventory_state(demo.InventoryReceived(**common, supplier_id="S"))
        transfer = dict(
            **common, source_location_id="North", destination_location_id="South", transfer_id="T"
        )
        await demo.update_inventory_state(demo.InventoryDispatched(**transfer))
        assert north.quantity_available == 0
        assert "South" not in demo.inventory_state["SKU-123"]
        await demo.update_inventory_state(demo.InventoryTransferred(**transfer))
        assert demo.inventory_state["SKU-123"]["South"].quantity_available == 5

    asyncio.run(run())


def test_gateway_preserves_compressed_bytes_and_repeated_headers(monkeypatch):
    import gzip

    import httpx
    from fastapi import BackgroundTasks, Request

    from demos import api_gateway_book_demo as demo

    compressed = gzip.compress(b"original", mtime=0)

    async def respond(request):
        return httpx.Response(
            200,
            stream=httpx.ByteStream(compressed),
            headers=[
                ("Content-Encoding", "gzip"),
                ("ETag", '"original"'),
                ("Set-Cookie", "a=1"),
                ("Set-Cookie", "b=2"),
            ],
        )

    factory = httpx.AsyncClient
    monkeypatch.setattr(demo.httpx, "AsyncClient", lambda: factory(transport=httpx.MockTransport(respond)))
    request = Request(dict(type="http", method="GET", path="/items", query_string=b"", headers=[]))
    request._body = b""
    response = asyncio.run(
        demo.proxy_request(
            request,
            "product-service",
            "/items",
            demo.Agent(agent_id="trusted", agent_name="agent", roles=["read"]),
            BackgroundTasks(),
        )
    )
    assert response.body == compressed
    assert response.headers["content-encoding"] == "gzip"
    assert response.headers.getlist("set-cookie") == ["a=1", "b=2"]


def test_gateway_unknown_service_failure_is_logged(caplog):
    import logging

    from fastapi import BackgroundTasks, HTTPException, Request

    from demos import api_gateway_book_demo as demo

    request = Request(dict(type="http", method="GET", path="/items", query_string=b"", headers=[]))
    request._body = b""
    with caplog.at_level(logging.INFO), pytest.raises(HTTPException) as error:
        asyncio.run(
            demo.proxy_request(
                request,
                "unknown",
                "/items",
                demo.Agent(agent_id="trusted", agent_name="agent", roles=["read"]),
                BackgroundTasks(),
            )
        )
    assert error.value.status_code == 404
    assert caplog.records[-1].outcome["status_code"] == 404


@pytest.mark.parametrize("change", ["price", "stock", "actor", "action", "policy", "expiry", "replay"])
def test_bound_approval_rejects_changed_or_unauthorized_execution(change):
    from datetime import datetime, timedelta, timezone

    from demos.approval_execution_demo import ApprovedPrice, PriceLedger

    now = datetime.now(timezone.utc)
    grant = ApprovedPrice(
        proposal_id="P",
        sku="SKU-123",
        current_price=100,
        stock=600,
        new_price=80,
        policy_version="v2",
        reviewer="pricing-lead",
        requester="agent",
        expires_at=now + timedelta(hours=24),
    )
    ledger = PriceLedger(
        prices={"SKU-123": 100}, stock={"SKU-123": 600}, policy_version="v2", approvals={"P": grant}
    )
    args = dict(proposal_id="P", actor="pricing-lead", new_price=80, now=now)
    if change == "price":
        ledger.prices["SKU-123"] = 99
    if change == "stock":
        ledger.stock["SKU-123"] = 599
    if change == "actor":
        args["actor"] = "unprivileged"
    if change == "action":
        args["new_price"] = 79
    if change == "policy":
        ledger.policy_version = "v3"
    if change == "expiry":
        args["now"] += timedelta(hours=24)
    if change == "replay":
        ledger.execute(**args)
    before = dict(ledger.prices)
    with pytest.raises(ValueError):
        ledger.execute(**args)
    assert ledger.prices == before


@pytest.mark.parametrize("cost,allowed", [(0, True), (100, True), (120, False)])
def test_printed_cost_ceiling_cases(cost, allowed):
    from agents.raom_minimal import ActionRequest, MaxCostRule

    assert (
        not MaxCostRule(100).evaluate(
            ActionRequest(tool_name="place_reorder", payload={}, estimated_cost=cost)
        )
    ) == allowed


def test_printed_unlisted_tool_is_denied():
    from agents.raom_minimal import ActionRequest, ToolAllowlistRule

    assert ToolAllowlistRule({"place_reorder"}).evaluate(
        ActionRequest(tool_name="delete_inventory", payload={}, estimated_cost=0)
    ) == ["tool_not_allowed:delete_inventory"]


def test_printed_pricing_contract_economics_and_scope():
    import json
    from pathlib import Path

    contract = json.loads(
        (Path(__file__).resolve().parents[2] / "demos/book_pricing_contracts.json").read_text()
    )["approval_request"]
    assert contract["price"]["currency"] == "USD"
    assert contract["scope"] == {
        "stores": ["store-north"],
        "channels": ["store_pos", "web"],
        "price_type": "advertised",
    }
    assert contract["product"] == {"sku": "SKU-123", "pack_quantity": 1, "pack_unit": "each"}
    assert contract["risk"]["rationale"]
    for reference in contract["evidence"]["signals"] + contract["evidence"]["policy_refs"]:
        file_name, pointer = reference.split("#", 1)
        record = json.loads((Path(__file__).resolve().parents[2] / file_name).read_text())
        for key in pointer.strip("/").split("/"):
            record = record[key]
        assert record["kind"] in {"synthetic_signal", "illustrative_policy"}
    checked = contract["constraint_checked"]
    assert checked["unit_cost_usd"] == 12
    assert checked["proposed_margin"] == pytest.approx(10 / 22)
    assert checked["proposed_margin"] == pytest.approx(
        (contract["price"]["proposed"] - checked["unit_cost_usd"]) / contract["price"]["proposed"]
    )
    assert checked["proposed_margin"] >= checked["floor"]
    contribution = contract["contribution"]
    assert contribution["other_variable_cost_usd"] == 4
    for price_type, expected in (("current", 4), ("proposed", 6)):
        assert contribution[f"{price_type}_per_unit_usd"] == expected
        assert contribution[f"{price_type}_per_unit_usd"] == (
            contract["price"][price_type] - checked["unit_cost_usd"] - contribution["other_variable_cost_usd"]
        )


def test_printed_pricing_contract_starts_friday_and_covers_promotion_remainder():
    import json
    from datetime import datetime, timedelta
    from pathlib import Path

    contract = json.loads(
        (Path(__file__).resolve().parents[2] / "demos/book_pricing_contracts.json").read_text()
    )["approval_request"]
    assert contract["effective"] == {
        "from": "2026-10-16T00:00:00Z",
        "to": "2026-10-19T00:00:00Z",
    }
    start, end = (
        datetime.fromisoformat(contract["effective"][key].replace("Z", "+00:00")) for key in ("from", "to")
    )
    assert start.weekday() == 4 and end.weekday() == 0
    assert end - start == timedelta(days=3)


def test_seeded_inventory_sharing_scores_ignore_caller_randomness():
    from agents.protocols.inventory_sharing import InventoryCollaborationNetwork
    from models.store import Store

    costs = []
    for seed in (1, 77):
        random.seed(seed)
        network = InventoryCollaborationNetwork()
        for name in ("A", "B"):
            network.register_store(Store(store_id=name, name=name, location=name))
        costs.append(network.transfer_costs[("A", "B")])
    assert costs[0] == costs[1]
