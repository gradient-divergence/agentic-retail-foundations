from __future__ import annotations

# region book:hitl-approval-api-models
from datetime import datetime, timezone
from itertools import count
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints, field_validator

app = FastAPI()
Actor = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PriceProposal(BaseModel):
    product_id: int
    current_price: float = Field(gt=0, allow_inf_nan=False)
    suggested_price: float = Field(ge=0, allow_inf_nan=False)
    requester: Actor

    @field_validator("requester")
    @classmethod
    def reserve_policy_identity(cls, value: str) -> str:
        if value.startswith("policy:"):
            raise ValueError("policy: identities are reserved for automatic approvals")
        return value


class PendingReview(BaseModel):
    review_id: int
    product_id: int
    current_price: float
    suggested_price: float
    reason: str
    requester: Actor


class PriceProposalResponse(BaseModel):
    status: Literal["pending", "auto_approved"]
    review_id: int | None = None
    new_price: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    message: str | None = None


class PendingReviewsResponse(BaseModel):
    reviews: list[PendingReview]


class ReviewAdjustment(BaseModel):
    new_price: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ReviewOutcome(BaseModel):
    status: Literal["approved", "modified", "rejected", "auto_approved"]
    product_id: int
    new_price: float | None = None
    reason: str | None = None
    reviewer: Actor
    requester: Actor
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


pending_reviews: dict[int, PendingReview] = {}
# ponytail: process-local ledger and lock; use a transactional store before multi-worker deployment.
review_outcomes: dict[int, ReviewOutcome] = {}
review_ids = count(1)
review_lock = Lock()
# Actors are synthetic demo identities; derive them from authenticated principals in a deployed API.
# endregion book:hitl-approval-api-models


# region book:hitl-approval-api-endpoints
@app.post("/ai/propose_price", response_model=PriceProposalResponse)
def propose_price(payload: PriceProposal) -> PriceProposalResponse:
    change_percent = (payload.current_price - payload.suggested_price) / payload.current_price * 100
    review_id = next(review_ids)
    if change_percent > 20:
        pending_review = PendingReview(
            review_id=review_id,
            product_id=payload.product_id,
            current_price=payload.current_price,
            suggested_price=payload.suggested_price,
            reason="High discount > 20%, pending approval",
            requester=payload.requester,
        )
        pending_reviews[review_id] = pending_review
        return PriceProposalResponse(
            status="pending",
            review_id=review_id,
            message="Escalated for human approval",
        )

    review_outcomes[review_id] = ReviewOutcome(
        status="auto_approved",
        product_id=payload.product_id,
        new_price=payload.suggested_price,
        reviewer="policy:discount_threshold",
        requester=payload.requester,
    )
    return PriceProposalResponse(
        status="auto_approved", review_id=review_id, new_price=payload.suggested_price
    )


@app.get("/admin/pending_reviews", response_model=PendingReviewsResponse)
def list_pending() -> PendingReviewsResponse:
    return PendingReviewsResponse(reviews=list(pending_reviews.values()))


def _review_for_actor(review_id: int, reviewer: str) -> PendingReview:
    review = pending_reviews.get(review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found or already processed")
    if not reviewer.strip():
        raise HTTPException(status_code=400, detail="Reviewer is required")
    if reviewer.strip() == review.requester:
        raise HTTPException(status_code=403, detail="Requester cannot review their own proposal")
    return review


@app.post("/admin/review/{review_id}/approve", response_model=ReviewOutcome)
def approve_price(review_id: int, reviewer: str) -> ReviewOutcome:
    with review_lock:
        review = _review_for_actor(review_id, reviewer)
        outcome = ReviewOutcome(
            status="approved",
            product_id=review.product_id,
            new_price=review.suggested_price,
            reviewer=reviewer,
            requester=review.requester,
        )
        review_outcomes[review_id] = outcome
        pending_reviews.pop(review_id)
        return outcome


@app.post("/admin/review/{review_id}/reject", response_model=ReviewOutcome)
def reject_price(
    review_id: int, adjustment: ReviewAdjustment | None = None, *, reviewer: str
) -> ReviewOutcome:
    with review_lock:
        review = _review_for_actor(review_id, reviewer)
        new_price = adjustment.new_price if adjustment is not None else None
        outcome = ReviewOutcome(
            status="modified" if new_price is not None else "rejected",
            product_id=review.product_id,
            new_price=new_price,
            reviewer=reviewer,
            requester=review.requester,
            reason=None if new_price is not None else "Human rejected AI suggestion",
        )
        review_outcomes[review_id] = outcome
        pending_reviews.pop(review_id)
        return outcome


# endregion book:hitl-approval-api-endpoints
