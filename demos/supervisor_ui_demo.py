from __future__ import annotations

from capstone.policies import PolicyEvaluator, PolicyInput, apply_supervisor_action
from capstone.schemas import AuditRecord
from capstone.tracing import TraceContext
from models.governance import SupervisorAction


def main() -> None:
    trace = TraceContext(route="capstone/supervisor", tenant_id="retail-demo")
    policy = PolicyEvaluator()
    policy_input = PolicyInput(
        action="price_change",
        amount=0.25,
        risk_score=0.8,
        evidence=["price_delta=0.25", "promo_window=true"],
    )
    decision = policy.evaluate(policy_input)

    supervisor_action = SupervisorAction(action="approve", reviewer="pricing-lead", notes="Promo approved.")
    # ponytail: process-local demo ledger; persist entries before multi-worker deployment.
    audit_log = []
    final_decision = apply_supervisor_action(
        decision,
        supervisor_action,
        requester="pricing-agent",
        trace_id=trace.trace_id,
        audit_log=audit_log,
        action=policy_input.action,
        evidence=policy_input.evidence,
    )
    entry = audit_log[-1]
    audit = AuditRecord(record_id=entry.entry_id, **entry.model_dump(exclude={"entry_id"}))

    print("Initial decision:", decision.model_dump())
    print("Supervisor action:", supervisor_action.model_dump())
    print("Final decision:", final_decision.model_dump())
    print("Audit record:", audit.model_dump())


if __name__ == "__main__":
    main()
