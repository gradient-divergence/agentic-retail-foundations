from __future__ import annotations

# region book:hitl-approval-api-models
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class PriceProposal(BaseModel):
    product_id: int
    current_price: float
    suggested_price: float


class PendingReview(BaseModel):
    review_id: int
    product_id: int
    current_price: float
    suggested_price: float
    reason: str


class PriceProposalResponse(BaseModel):
    status: Literal["pending", "auto_approved"]
    review_id: int | None = None
    new_price: float | None = None
    message: str | None = None


class PendingReviewsResponse(BaseModel):
    reviews: list[PendingReview]


class ReviewAdjustment(BaseModel):
    new_price: float | None = None


class ReviewOutcome(BaseModel):
    status: Literal["approved", "modified", "rejected"]
    product_id: int
    new_price: float | None = None
    reason: str | None = None


pending_reviews: dict[int, PendingReview] = {}
# endregion book:hitl-approval-api-models


# region book:hitl-approval-api-endpoints
@app.post("/ai/propose_price", response_model=PriceProposalResponse)
def propose_price(payload: PriceProposal) -> PriceProposalResponse:
    change_percent = (payload.current_price - payload.suggested_price) / payload.current_price * 100
    if change_percent > 20:
        review_id = len(pending_reviews) + 1
        pending_review = PendingReview(
            review_id=review_id,
            product_id=payload.product_id,
            current_price=payload.current_price,
            suggested_price=payload.suggested_price,
            reason="High discount > 20%, pending approval",
        )
        pending_reviews[review_id] = pending_review
        return PriceProposalResponse(
            status="pending",
            review_id=review_id,
            message="Escalated for human approval",
        )

    return PriceProposalResponse(status="auto_approved", new_price=payload.suggested_price)


@app.get("/admin/pending_reviews", response_model=PendingReviewsResponse)
def list_pending() -> PendingReviewsResponse:
    return PendingReviewsResponse(reviews=list(pending_reviews.values()))


@app.post("/admin/review/{review_id}/approve", response_model=ReviewOutcome)
def approve_price(review_id: int) -> ReviewOutcome:
    review = pending_reviews.pop(review_id, None)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found or already processed")
    return ReviewOutcome(
        status="approved",
        product_id=review.product_id,
        new_price=review.suggested_price,
    )


@app.post("/admin/review/{review_id}/reject", response_model=ReviewOutcome)
def reject_price(review_id: int, adjustment: ReviewAdjustment | None = None) -> ReviewOutcome:
    review = pending_reviews.pop(review_id, None)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found or already processed")
    if adjustment is None:
        adjustment = ReviewAdjustment()
    if adjustment.new_price is not None:
        return ReviewOutcome(
            status="modified",
            product_id=review.product_id,
            new_price=adjustment.new_price,
        )
    return ReviewOutcome(
        status="rejected",
        product_id=review.product_id,
        reason="Human rejected AI suggestion",
    )


# endregion book:hitl-approval-api-endpoints
