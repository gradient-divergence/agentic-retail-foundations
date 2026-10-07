"""AP2 test-mode sketch: build synthetic payment types locally; no payment is sent."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ap2.models.payment_request import PaymentRequest


def build_payment_request() -> PaymentRequest:
    try:
        from ap2.models.payment_request import (
            PaymentCurrencyAmount,
            PaymentDetailsInit,
            PaymentItem,
            PaymentMethodData,
            PaymentOptions,
            PaymentRequest,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'ap2 types package is not installed. Install with: uv pip install -e ".[agent_protocols]"'
        ) from exc

    item = PaymentItem(
        label="Seasonal jacket",
        amount=PaymentCurrencyAmount(currency="USD", value="129.99"),
    )
    details = PaymentDetailsInit(
        id="test_order_789",
        display_items=[item],
        total=PaymentItem(
            label="Total",
            amount=PaymentCurrencyAmount(currency="USD", value="129.99"),
        ),
    )
    method_data = PaymentMethodData(supported_methods="basic-card")
    options = PaymentOptions(request_payer_email=True)
    return PaymentRequest(
        method_data=[method_data],
        details=details,
        options=options,
    )


def main() -> None:
    payment_request = build_payment_request()
    print(payment_request.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
