"""Process-local exact approval fixture; actor identity must come from a trusted caller.

This demonstrates the execution checks beyond the chapter's approval-recording API.
It is not an authentication service or a multi-worker ledger.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from demos.hitl_approval_api_demo import money_minor_units


class ApprovedPrice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    proposal_id: str
    sku: str
    current_price: float = Field(gt=0)
    stock: int = Field(ge=0)
    new_price: float = Field(ge=0)
    currency: Literal["USD"] = "USD"
    policy_version: str
    reviewer: str
    requester: str
    expires_at: datetime

    @field_validator("current_price", "new_price")
    @classmethod
    def validate_money(cls, value):
        return money_minor_units(value)


@dataclass
class PriceLedger:
    prices: dict[str, float]
    stock: dict[str, int]
    policy_version: str
    approvals: dict[str, ApprovedPrice]
    authorized_reviewers: frozenset[str] = frozenset({"pricing-lead"})
    executed: set[str] = field(default_factory=set)

    def execute(self, proposal_id: str, actor: str, new_price: float, now: datetime) -> float:
        grant = self.approvals.get(proposal_id)
        if grant is None or proposal_id in self.executed:
            raise ValueError("Missing or already executed approval")
        if actor not in self.authorized_reviewers or actor != grant.reviewer or actor == grant.requester:
            raise ValueError("Unauthorized actor")
        if now.tzinfo is None or grant.expires_at.tzinfo is None or now >= grant.expires_at:
            raise ValueError("Expired approval")
        if (self.prices.get(grant.sku), self.stock.get(grant.sku), self.policy_version) != (
            grant.current_price,
            grant.stock,
            grant.policy_version,
        ):
            raise ValueError("Stale proposal")
        if new_price != grant.new_price:
            raise ValueError("Modified action requires a new approval")
        self.prices[grant.sku] = new_price
        self.executed.add(proposal_id)
        return new_price
