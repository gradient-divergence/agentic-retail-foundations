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
    final_decision = apply_supervisor_action(decision, supervisor_action)

    audit = AuditRecord(
        trace_id=trace.trace_id,
        actor=supervisor_action.reviewer,
        action=policy_input.action,
        decision=final_decision.reason,
        evidence=policy_input.evidence,
    )

    print("Initial decision:", decision.model_dump())
    print("Supervisor action:", supervisor_action.model_dump())
    print("Final decision:", final_decision.model_dump())
    print("Audit record:", audit.model_dump())


if __name__ == "__main__":
    main()
