import pytest

from capstone.policies import PolicyEvaluator, PolicyInput, apply_supervisor_action
from models.governance import SupervisorAction


@pytest.mark.parametrize("action", ["approve", "reject", "request_info"])
@pytest.mark.parametrize(
    "requester,reviewer", [("agent", "agent"), (" agent ", "agent"), ("", "lead"), ("agent", " ")]
)
def test_supervisor_requires_distinct_nonblank_actors(action, requester, reviewer):
    decision = PolicyEvaluator().evaluate(PolicyInput(action="price_change", amount=0.25))
    ledger = []
    with pytest.raises(ValueError):
        apply_supervisor_action(
            decision,
            SupervisorAction(action=action, reviewer=reviewer),
            requester=requester,
            trace_id="trace-test",
            audit_log=ledger,
        )
    assert ledger == []
    assert not decision.allowed


@pytest.mark.parametrize(
    "action,allowed,reason",
    [
        ("approve", True, "approved_by_supervisor"),
        ("reject", False, "rejected_by_supervisor"),
        ("request_info", False, "supervisor_requested_info"),
    ],
)
def test_supervisor_records_outcome_before_return(action, allowed, reason):
    decision = PolicyEvaluator().evaluate(PolicyInput(action="price_change", amount=0.25))
    ledger = []
    outcome = apply_supervisor_action(
        decision,
        SupervisorAction(action=action, reviewer=" lead ", notes="Promo reviewed"),
        requester=" agent ",
        trace_id="trace-test",
        audit_log=ledger,
    )
    assert outcome.allowed is allowed
    assert outcome.reason == reason
    assert not decision.allowed
    assert len(ledger) == 1
    assert ledger[0].actor == "lead"
    assert ledger[0].trace_id == "trace-test"
    assert ledger[0].action == action
    assert ledger[0].decision == reason
    assert ledger[0].evidence == ["requester=agent", "Promo reviewed"]


def test_failed_audit_cannot_return_approval():
    class FailedLedger(list):
        def append(self, item):
            raise OSError("audit unavailable")

    decision = PolicyEvaluator().evaluate(PolicyInput(action="price_change", amount=0.25))
    with pytest.raises(OSError, match="audit unavailable"):
        apply_supervisor_action(
            decision,
            SupervisorAction(action="approve", reviewer="lead"),
            requester="agent",
            trace_id="trace-test",
            audit_log=FailedLedger(),
        )
    assert not decision.allowed


def test_review_preserves_action_and_evidence_without_mutating_input():
    decision = PolicyEvaluator().evaluate(PolicyInput(action="price_change", amount=0.25))
    evidence = ["price_delta=0.25"]
    ledger = []
    apply_supervisor_action(
        decision,
        SupervisorAction(action="approve", reviewer="lead"),
        requester="agent",
        trace_id="trace-test",
        audit_log=ledger,
        action="price_change",
        evidence=evidence,
    )
    assert ledger[0].action == "price_change"
    assert ledger[0].evidence == ["price_delta=0.25", "requester=agent"]
    assert evidence == ["price_delta=0.25"]
