from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from models.governance import AuditLogEntry, PolicyDecision, SupervisorAction


class PolicyInput(BaseModel):
    action: str
    amount: float | None = Field(default=None, allow_inf_nan=False)
    customer_tier: Literal["standard", "gold", "vip"] | None = None
    channel: Literal["email", "sms", "app"] | None = None
    risk_score: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False)
    evidence: list[str] = Field(default_factory=list)


class PolicyEvaluator(BaseModel):
    policy_id: str = "capstone-default"
    price_change_threshold: float = 0.1
    supplier_commitment_threshold: float = 5000.0
    high_risk_threshold: float = 0.75

    def evaluate(self, policy_input: PolicyInput) -> PolicyDecision:
        risk_level = self._risk_level(policy_input.risk_score)
        allowed = True
        reason = "approved"

        if policy_input.action == "price_change" and policy_input.amount is not None:
            if abs(policy_input.amount) >= self.price_change_threshold:
                allowed = False
                reason = "price_change_requires_approval"

        if policy_input.action == "supplier_commitment" and policy_input.amount is not None:
            if policy_input.amount >= self.supplier_commitment_threshold:
                allowed = False
                reason = "supplier_commitment_requires_approval"

        if risk_level == "high":
            allowed = False
            reason = "risk_level_high"

        return PolicyDecision(
            allowed=allowed,
            policy_id=self.policy_id,
            reason=reason,
            risk_level=risk_level,
        )

    def _risk_level(self, risk_score: float) -> Literal["low", "medium", "high"]:
        if risk_score >= self.high_risk_threshold:
            return "high"
        if risk_score >= self.high_risk_threshold / 2:
            return "medium"
        return "low"


def build_audit_entry(
    decision: PolicyDecision,
    trace_id: str,
    actor: str,
    action: str,
    evidence: list[str] | None = None,
) -> AuditLogEntry:
    return AuditLogEntry(
        trace_id=trace_id,
        actor=actor,
        action=action,
        decision=decision.reason,
        evidence=evidence or [],
    )


def apply_supervisor_action(
    decision: PolicyDecision,
    supervisor_action: SupervisorAction,
    *,
    requester: str,
    trace_id: str,
    audit_log: list[AuditLogEntry],
    action: str | None = None,
    evidence: list[str] | None = None,
) -> PolicyDecision:
    requester, reviewer = requester.strip(), supervisor_action.reviewer.strip()
    if not requester or not reviewer or requester == reviewer:
        raise ValueError("Supervisor review requires distinct, nonblank requester and reviewer")
    reason = {
        "approve": "approved_by_supervisor",
        "reject": "rejected_by_supervisor",
        "request_info": "supervisor_requested_info",
    }[supervisor_action.action]
    outcome = decision.model_copy(
        update={
            "allowed": supervisor_action.action == "approve",
            "reason": reason,
        }
    )
    audit_evidence = [*(evidence or []), f"requester={requester}"]
    if supervisor_action.notes:
        audit_evidence.append(supervisor_action.notes)
    audit_log.append(
        build_audit_entry(outcome, trace_id, reviewer, action or supervisor_action.action, audit_evidence)
    )
    return outcome
