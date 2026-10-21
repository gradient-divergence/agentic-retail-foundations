from __future__ import annotations

from pydantic import BaseModel

from capstone.orchestrator import CapstoneOrchestrator
from capstone.policies import PolicyEvaluator
from capstone.schemas import ToolCall
from capstone.tools import ToolSpec
from capstone.tracing import TraceContext


class InventoryRisk(BaseModel):
    sku: str
    risk_score: float
    recommended_action: str


def lookup_inventory_risk(sku: str) -> InventoryRisk:
    return InventoryRisk(sku=sku, risk_score=0.42, recommended_action="replenish")


def main() -> None:
    trace = TraceContext(route="capstone/scaffold", tenant_id="retail-demo")
    tool = ToolSpec(
        name="inventory_risk_lookup",
        description="Return a risk score for the SKU.",
        permission="read",
        schema={
            "type": "object",
            "properties": {"sku": {"type": "string"}},
            "required": ["sku"],
        },
        handler=lookup_inventory_risk,
    )
    policy = PolicyEvaluator()
    orchestrator = CapstoneOrchestrator([tool], policy)
    call = ToolCall(name="inventory_risk_lookup", args={"sku": "SKU-123"})
    result, audit = orchestrator.handle_tool_call(call, trace)

    print("Tool result:", result.model_dump())
    print("Audit record:", audit.model_dump())
    print("Latency:", trace.finish())


def run_governed_episode() -> dict:
    """Synthetic buyer approval and receipt trace; identities are local fixtures."""
    from capstone.policies import PolicyInput, apply_supervisor_action
    from models.governance import SupervisorAction

    sku, proposal_id = "SKU-123", "proposal-123"
    risk = lookup_inventory_risk(sku)
    policy = PolicyEvaluator()
    proposal = PolicyInput(
        action="supplier_commitment",
        amount=6000,
        currency="USD",
        risk_score=risk.risk_score,
        evidence=[proposal_id, sku, "quantity=80"],
    )
    initial_stock, demand, order_quantity = 20, 60, 80
    proposals = [
        {"action": "hold", "projected_stockouts": max(0, demand - initial_stock)},
        {"action": "replenish", "projected_stockouts": max(0, demand - initial_stock - order_quantity)},
    ]
    selected = min(proposals, key=lambda proposal: proposal["projected_stockouts"])
    assert selected["action"] == "replenish"
    events = ["risk_lookup", "proposals_compared"]
    decision = policy.evaluate(proposal)
    assert not decision.allowed
    events.append("policy_rejected")
    audit = []
    approved = apply_supervisor_action(
        decision,
        SupervisorAction(action="approve", reviewer="buyer-2"),
        requester="planner-1",
        trace_id=proposal_id,
        audit_log=audit,
        action=proposal.action,
        evidence=proposal.evidence,
    )
    assert approved.allowed
    events.append("buyer_approved_exact_proposal")
    receipts, on_hand, on_order, inventory_trace = {}, initial_stock, 0, []

    def receive_once(identity: str, quantity: int) -> dict:
        nonlocal on_hand, on_order
        if identity in receipts:
            return receipts[identity]
        on_hand += quantity
        on_order -= quantity
        receipts[identity] = dict(proposal_id=identity, sku=sku, quantity=quantity)
        return receipts[identity]

    # The injected failure occurs before the synthetic operation commits.
    try:
        raise TimeoutError("fixture: supplier unreachable before acceptance")
    except TimeoutError:
        events += ["execution_failed_before_commit", "retry_authorized"]
    on_order = order_quantity
    events.append("execution_confirmed")
    inventory_trace.append(dict(event=events[-1], on_hand=on_hand, on_order=on_order))
    receipt = receive_once(proposal_id, order_quantity)
    events.append("physical_receipt_recorded")
    inventory_trace.append(dict(event=events[-1], on_hand=on_hand, on_order=on_order))
    assert receive_once(proposal_id, 80) is receipt
    events.append("duplicate_receipt_ignored")
    sales = min(demand, on_hand)
    ending_stock = on_hand - sales
    passed = (
        on_hand == 100
        and sales == demand
        and ending_stock >= 0
        and len(receipts) == 1
        and audit[0].actor == "buyer-2"
    )
    assert passed
    events.append("evaluation_passed")
    return dict(
        sku=sku,
        proposal_id=proposal_id,
        proposed_actions=proposals,
        baseline_stockouts=max(0, demand - initial_stock),
        demand_units=demand,
        sales_units=sales,
        ending_on_hand=ending_stock,
        measured_fill_rate=sales / demand,
        selected_action="replenish",
        events=events,
        on_hand=on_hand,
        on_order=on_order,
        inventory_trace=inventory_trace,
        currency=proposal.currency,
        unit_cost=proposal.amount / order_quantity,
        selling_price=100.0,  # Synthetic USD per unit; not part of the order-approval rule.
        receipt_count=len(receipts),
        evaluation_passed=passed,
    )


if __name__ == "__main__":
    main()
    print("Governed synthetic episode:", run_governed_episode())
