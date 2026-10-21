#!/usr/bin/env python3
# region book:structured-output-demo
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class PriceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    sku: str
    current_price: float = Field(gt=0)
    new_price: float = Field(..., gt=0)
    currency: Literal["USD"]
    zone: str = Field(min_length=1)
    channel: str = Field(min_length=1)
    effective_from: str
    effective_to: str
    policy_version: str = Field(min_length=1)
    sales_units: int = Field(ge=0)
    sales_days: int = Field(gt=0)
    reason: str
    confidence: float = Field(..., ge=0, le=1)

    @field_validator("current_price", "new_price")
    @classmethod
    def minor_units(cls, value: float) -> float:
        amount = Decimal(str(value))
        if amount.as_tuple().exponent < -2 and amount != amount.quantize(Decimal("0.01")):
            raise ValueError("Demo prices require a 0.01 currency minor unit")
        return value

    @model_validator(mode="after")
    def effective_window(self):
        start, end = datetime.fromisoformat(self.effective_from), datetime.fromisoformat(self.effective_to)
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError("Effective window requires increasing timezone-aware dates")
        return self


class PriceUpdatePayload(PriceUpdate):
    pass


def propose_price_update() -> PriceUpdatePayload:
    return PriceUpdatePayload(
        sku="SKU-1001",
        current_price=129.0,
        currency="USD",
        zone="North",
        channel="web",
        effective_from="2026-10-13T00:00:00+00:00",
        effective_to="2026-10-14T00:00:00+00:00",
        policy_version="pricing.v1",
        sales_units=20,
        sales_days=7,
        new_price=109.0,
        reason="Competitor markdown within 24h",
        confidence=0.78,
    )


def validate_tool_payload(payload: PriceUpdatePayload | dict) -> PriceUpdate:
    return PriceUpdate.model_validate(payload.model_dump() if isinstance(payload, BaseModel) else payload)


def apply_price(update: PriceUpdate) -> None:
    print(f"Apply price: {update.sku} -> {update.new_price} {update.currency}")


def run_pipeline() -> None:
    try:
        payload = propose_price_update()
        update = validate_tool_payload(payload)
    except ValidationError as exc:
        print("route_to_human_review", exc.errors())
        return
    apply_price(update)


if __name__ == "__main__":
    run_pipeline()
# endregion book:structured-output-demo
