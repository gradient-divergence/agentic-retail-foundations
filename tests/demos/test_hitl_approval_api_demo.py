import pytest
from pydantic import ValidationError

from tests.demos.api_fakes import HTTPException, load_demo


@pytest.fixture
def demo(monkeypatch):
    return load_demo(monkeypatch, "hitl_approval_api_demo")


def proposal(demo, requester="agent", suggested_price=70):
    return demo.PriceProposal(
        product_id=1, current_price=100, suggested_price=suggested_price, requester=requester
    )


def test_self_approval_is_blocked_without_consuming_review(demo):
    review_id = demo.propose_price(proposal(demo)).review_id
    with pytest.raises(HTTPException) as exc:
        demo.approve_price(review_id, reviewer="agent")
    assert exc.value.status_code == 403
    assert review_id in demo.pending_reviews
    assert not demo.review_outcomes


def test_blank_reviewer_and_whitespace_self_approval_are_blocked(demo):
    review_id = demo.propose_price(proposal(demo, requester=" agent ")).review_id
    for reviewer in (" ", " agent "):
        with pytest.raises(HTTPException):
            demo.approve_price(review_id, reviewer=reviewer)
        assert review_id in demo.pending_reviews


def test_requester_cannot_impersonate_automatic_policy(demo):
    with pytest.raises(ValidationError):
        proposal(demo, requester="policy:discount_threshold", suggested_price=90)


def test_approval_is_recorded_and_cannot_be_replayed(demo):
    review_id = demo.propose_price(proposal(demo)).review_id
    outcome = demo.approve_price(review_id, reviewer="manager")
    assert outcome.new_price == 70
    assert demo.review_outcomes[review_id] == outcome
    assert outcome.reviewer == "manager"
    assert outcome.requester == "agent"
    with pytest.raises(HTTPException):
        demo.approve_price(review_id, reviewer="manager")


def test_missing_approval_and_self_modification_cannot_execute(demo):
    with pytest.raises(HTTPException):
        demo.approve_price(999, reviewer="manager")
    review_id = demo.propose_price(proposal(demo)).review_id
    with pytest.raises(HTTPException):
        demo.reject_price(review_id, adjustment=demo.ReviewAdjustment(new_price=80), reviewer="agent")
    assert review_id in demo.pending_reviews
    outcome = demo.reject_price(review_id, adjustment=demo.ReviewAdjustment(new_price=80), reviewer="manager")
    assert outcome.status == "modified"
    assert demo.review_outcomes[review_id] == outcome


def test_ids_do_not_overwrite_other_pending_reviews(demo):
    first = demo.propose_price(proposal(demo)).review_id
    second = demo.propose_price(proposal(demo)).review_id
    demo.approve_price(first, reviewer="manager")
    third = demo.propose_price(proposal(demo)).review_id
    assert len({first, second, third}) == 3
    assert second in demo.pending_reviews


def test_minor_change_has_recorded_policy_approval(demo):
    outcome = demo.propose_price(proposal(demo, suggested_price=90))
    assert outcome.status == "auto_approved"
    assert outcome.review_id in demo.review_outcomes
    assert demo.review_outcomes[outcome.review_id].reviewer == "policy:discount_threshold"


def test_concurrent_review_is_processed_once(demo):
    from concurrent.futures import ThreadPoolExecutor

    review_id = demo.propose_price(proposal(demo)).review_id

    def approve():
        try:
            return demo.approve_price(review_id, reviewer="manager").status
        except HTTPException as exc:
            return exc.status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: approve(), range(2)))
    assert results.count("approved") == 1
    assert results.count(404) == 1


@pytest.mark.parametrize(
    "field,value",
    [("current_price", 0), ("suggested_price", -1), ("suggested_price", float("nan")), ("requester", " ")],
)
def test_invalid_proposals_are_rejected(demo, field, value):
    payload = dict(product_id=1, current_price=100, suggested_price=70, requester="agent")
    payload[field] = value
    with pytest.raises(ValidationError):
        demo.PriceProposal(**payload)


def test_http_routes_when_web_stack_is_available():
    from fastapi.testclient import TestClient

    from demos.hitl_approval_api_demo import app

    client = TestClient(app)
    response = client.post(
        "/ai/propose_price", json=dict(product_id=1, current_price=100, suggested_price=70, requester="agent")
    )
    review_id = response.json()["review_id"]
    assert client.post(f"/admin/review/{review_id}/approve", params={"reviewer": "agent"}).status_code == 403
    assert (
        client.post(f"/admin/review/{review_id}/approve", params={"reviewer": "manager"}).status_code == 200
    )
