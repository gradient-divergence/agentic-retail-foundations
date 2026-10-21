#!/usr/bin/env python3
# region book:session-state-demo
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class SessionItem(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_name: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SessionStore:
    session_id: str
    items: list[SessionItem] = field(default_factory=list)

    def add(self, item: SessionItem) -> None:
        self.items.append(item)

    def history(self) -> list[SessionItem]:
        return list(self.items)


def plan_next_step(user_request: str, history: list[SessionItem]) -> str:
    if "return" in user_request.lower():
        return "check_return_policy"
    if "deadline" in user_request.lower() and any(
        item.role == "assistant" and item.content == "Intent: check_return_policy" for item in history
    ):
        return "check_return_policy"
    return "search_catalog"


class SessionSummary(BaseModel):
    session_id: str
    last_intent: str
    item_count: int


def run_turn(session: SessionStore, user_request: str) -> SessionSummary:
    session.add(SessionItem(role="user", content=user_request))

    intent = plan_next_step(user_request, session.history())
    session.add(SessionItem(role="assistant", content=f"Intent: {intent}"))
    session.add(SessionItem(role="tool", content="ok", tool_name=intent))

    return SessionSummary(
        session_id=session.session_id,
        last_intent=intent,
        item_count=len(session.items),
    )


if __name__ == "__main__":
    session = SessionStore(session_id="sess_demo")
    summary = run_turn(session, "Can I return these shoes?")
    print(summary.model_dump())
    follow_up = run_turn(session, "What is the deadline?")
    assert summary.last_intent == follow_up.last_intent == "check_return_policy"
    assert follow_up.item_count == 6
    print(follow_up.model_dump())
# endregion book:session-state-demo
