from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SupportRequest:
    customer_id: str
    channel: str
    message: str


def triage_request(request: SupportRequest) -> str:
    text = request.message.lower()
    if "refund" in text or "billing" in text:
        return "BillingAgent"
    if "broken" in text or "defect" in text:
        return "ProductSupportAgent"
    if "where is" in text or "track" in text:
        return "OrderStatusAgent"
    return "GeneralSupportAgent"


def run_demo() -> None:
    sample = SupportRequest(
        customer_id="cust_001",
        channel="chat",
        message="I want a refund for a damaged item",
    )
    routed_agent = triage_request(sample)
    print(f"Routed to: {routed_agent}")


if __name__ == "__main__":
    run_demo()
