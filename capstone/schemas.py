from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, JsonValue


class ToolCall(BaseModel):
    name: str
    args: dict[str, JsonValue]


class ToolResult(BaseModel):
    name: str
    status: Literal["ok", "error"]
    output: dict[str, JsonValue]
    trace_id: str


class EventEnvelope(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    version: str = "1.0"
    payload: dict[str, JsonValue]
    trace_id: str


class AuditRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str
    actor: str
    action: str
    decision: str
    evidence: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
