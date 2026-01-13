from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class PolicyDecision(BaseModel):
    allowed: bool
    policy_id: str
    reason: str
    risk_level: Literal["low", "medium", "high"]


class AuditLogEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str
    actor: str
    action: str
    decision: str
    evidence: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SupervisorAction(BaseModel):
    action: Literal["approve", "reject", "request_info"]
    reviewer: str
    notes: str | None = None
