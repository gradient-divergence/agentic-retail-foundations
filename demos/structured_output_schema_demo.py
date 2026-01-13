#!/usr/bin/env python3
# region book:structured-output-demo
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PriceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str
    new_price: float = Field(..., gt=0)
    reason: str
    confidence: float = Field(..., ge=0, le=1)


class PriceUpdatePayload(BaseModel):
    sku: str
    new_price: float = Field(..., gt=0)
    reason: str
    confidence: float = Field(..., ge=0, le=1)


def propose_price_update() -> PriceUpdatePayload:
    return PriceUpdatePayload(
        sku="SKU-1001",
        new_price=109.0,
        reason="Competitor markdown within 24h",
        confidence=0.78,
    )


def validate_tool_payload(payload: PriceUpdatePayload) -> PriceUpdate:
    return PriceUpdate.model_validate(payload.model_dump())


def apply_price(update: PriceUpdate) -> None:
    print(f"Apply price: {update.sku} -> {update.new_price}")


def run_pipeline() -> None:
    payload = propose_price_update()
    try:
        update = validate_tool_payload(payload)
    except ValidationError as exc:
        print("route_to_human_review", exc.errors())
        return
    apply_price(update)


if __name__ == "__main__":
    run_pipeline()
# endregion book:structured-output-demo
