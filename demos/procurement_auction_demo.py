"""
Demonstrates a reverse procurement auction for supplier selection.

This demo illustrates:
- supplier registration,
- bid submission under constraints (deadline, quality, budget),
- winner selection (lowest valid price for a reverse auction).
"""

# region book:procurement-auction-demo

import asyncio
import random
from datetime import datetime

from agents.protocols.auction import AuctionType, ProcurementAuction
from models.procurement import PurchaseOrder, SupplierBid
from models.supplier import Supplier, SupplierRating


def simulate_bid(supplier: Supplier, purchase_order: PurchaseOrder) -> SupplierBid | None:
    if purchase_order.product_id not in supplier.product_capabilities:
        return None

    base_price_per_unit = 15.0
    unit_price = base_price_per_unit * supplier.cost_factor * random.uniform(0.95, 1.05)
    total_price = unit_price * purchase_order.quantity

    base_days = 5
    delivery_days = int(
        base_days + (purchase_order.quantity / 200.0) * supplier.speed_factor * random.uniform(0.8, 1.2)
    )

    base_quality = 0.85
    quality_guarantee = min(
        0.99,
        base_quality + (1.0 / (supplier.quality_factor + 0.1)) * 0.15 * random.uniform(0.9, 1.1),
    )

    days_until_required = (purchase_order.required_delivery_date - datetime.now()).days
    if delivery_days > days_until_required:
        return None
    if quality_guarantee < purchase_order.quality_threshold:
        return None
    if total_price > purchase_order.maximum_acceptable_price:
        return None

    return SupplierBid(
        supplier_id=supplier.supplier_id,
        purchase_order_id=purchase_order.id,
        price=round(total_price, 2),
        delivery_days=delivery_days,
        quality_guarantee=round(quality_guarantee, 3),
    )


async def demo_procurement_auction() -> None:
    random.seed(7)

    suppliers = [
        Supplier(
            supplier_id="sup1",
            name="Alpha Supplies",
            rating=SupplierRating.STANDARD,
            product_capabilities=["PROD-XYZ", "PROD-ABC"],
            cost_factor=1.1,
            speed_factor=1.0,
            quality_factor=0.9,
        ),
        Supplier(
            supplier_id="sup2",
            name="Beta Goods Inc.",
            rating=SupplierRating.STANDARD,
            product_capabilities=["PROD-XYZ", "PROD-DEF"],
            cost_factor=1.0,
            speed_factor=0.8,
            quality_factor=1.0,
        ),
        Supplier(
            supplier_id="sup3",
            name="Gamma Distributors",
            rating=SupplierRating.PREFERRED,
            product_capabilities=["PROD-XYZ", "PROD-GHI"],
            cost_factor=1.2,
            speed_factor=0.9,
            quality_factor=0.8,
        ),
    ]

    purchase_order = PurchaseOrder(
        product_id="PROD-XYZ",
        quantity=1000,
        deadline_days=30,
        maximum_acceptable_price=20000.0,
        quality_threshold=0.90,
    )

    auction = ProcurementAuction(
        auction_id=f"AUC-{purchase_order.id}",
        purchase_order=purchase_order,
        auction_type=AuctionType.REVERSE,
        max_rounds=1,
    )
    for supplier in suppliers:
        auction.register_supplier(supplier)

    started = await auction.start_auction()
    if not started:
        print("Auction failed to start (not enough participants).")
        return

    submitted = 0
    for supplier in suppliers:
        bid = simulate_bid(supplier, purchase_order)
        if bid is None:
            continue
        if auction.submit_bid(bid):
            submitted += 1
            print(
                f"Bid: {supplier.name} price=${bid.price:.2f}, "
                f"days={bid.delivery_days}, quality={bid.quality_guarantee:.2%}"
            )

    print(f"Submitted bids: {submitted}")

    winner = await auction.finalize_auction()
    if winner is None:
        print("No winner (no valid bids).")
        return

    winning_supplier = next(s for s in suppliers if s.supplier_id == winner.supplier_id)
    print(
        f"Winner: {winning_supplier.name} price=${winner.price:.2f}, "
        f"days={winner.delivery_days}, quality={winner.quality_guarantee:.2%}"
    )


if __name__ == "__main__":
    asyncio.run(demo_procurement_auction())

# endregion book:procurement-auction-demo
