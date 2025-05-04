"""
Customer dataclass for retail personalization and analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Customer:
    """
    Data model for a retail customer.
    """

    customer_id: str
    name: str = ""
    email: str = ""
    segment: str = ""
    registration_date: datetime | None = None
    order_history: list[dict] = field(default_factory=list)
    preferences: dict[str, float] = field(default_factory=dict)
    last_active: datetime | None = None
