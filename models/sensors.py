"""Validated demo readings; thresholds are illustrative, not safety certification."""

from datetime import datetime
from math import isfinite
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


class Location(BaseModel):
    zone: str = Field(min_length=1)
    section: str = Field(min_length=1)
    zone_type: str = "ambient"


class Reading(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    event_id: str = Field(min_length=1)
    sensor_id: str = Field(min_length=1)
    timestamp: datetime
    location: Location
    confidence: float = Field(ge=0, le=1, strict=True)

    @field_validator("timestamp")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Sensor timestamp must be timezone-aware")
        return value


class ShelfProduct(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    product_id: str = Field(min_length=1)
    unit_weight_grams: float = Field(gt=0, strict=True)


class ShelfReading(Reading):
    sensor_type: Literal["smart_shelf"]
    shelf_id: str = Field(min_length=1)
    current_weight_grams: float = Field(ge=0, strict=True)
    expected_weight_grams: float = Field(ge=0, strict=True)
    tare_weight_grams: float = Field(ge=0, strict=True)
    product_info: ShelfProduct


class EnvironmentalReading(Reading):
    sensor_type: Literal["environmental"]
    environmental_type: Literal["temperature", "humidity"]
    value: float = Field(strict=True)
    unit: Literal["C", "F", "%"]

    @model_validator(mode="after")
    def compatible_units(self) -> "EnvironmentalReading":
        if (self.environmental_type == "temperature" and self.unit not in {"C", "F"}) or (
            self.environmental_type == "humidity" and self.unit != "%"
        ):
            raise ValueError("Environmental reading has incompatible units")
        return self


class DetectedTag(BaseModel):
    product_id: str = Field(min_length=1)


class RFIDReading(Reading):
    sensor_type: Literal["rfid"]
    detected_products: list[DetectedTag]


class PriceTagReading(Reading):
    sensor_type: Literal["digital_price_tag"]
    tag_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    price_displayed: float = Field(ge=0, strict=True)
    battery_level: float = Field(default=100, ge=0, le=100, strict=True)


READING = TypeAdapter(
    Annotated[
        ShelfReading | EnvironmentalReading | RFIDReading | PriceTagReading,
        Field(discriminator="sensor_type"),
    ]
)


def is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and isfinite(value)
