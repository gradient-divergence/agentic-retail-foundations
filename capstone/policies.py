from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from models.governance import AuditLogEntry, PolicyDecision, SupervisorAction


class PolicyInput(BaseModel):
    action: str
    amount: float | None = None
    customer_tier: Literal["standard", "gold", "vip"] | None = None
    channel: Literal["email", "sms", "app"] | None = None
    risk_score: float = 0.0
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


def apply_supervisor_action(decision: PolicyDecision, supervisor_action: SupervisorAction) -> PolicyDecision:
    if supervisor_action.action == "approve":
        return decision.model_copy(
            update={
                "allowed": True,
                "reason": "approved_by_supervisor",
                "risk_level": decision.risk_level,
            }
        )
    if supervisor_action.action == "reject":
        return decision.model_copy(
            update={
                "allowed": False,
                "reason": "rejected_by_supervisor",
                "risk_level": decision.risk_level,
            }
        )
    return decision.model_copy(
        update={
            "allowed": False,
            "reason": "supervisor_requested_info",
            "risk_level": decision.risk_level,
        }
    )
