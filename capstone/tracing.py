from __future__ import annotations

import time
from uuid import uuid4

from pydantic import BaseModel, Field


class TraceContext(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    route: str
    tenant_id: str | None = None
    start_time: float = Field(default_factory=time.perf_counter)

    def finish(self) -> dict[str, float]:
        return {"latency_ms": (time.perf_counter() - self.start_time) * 1000}
