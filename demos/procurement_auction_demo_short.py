"""Short, print-friendly reverse auction demo."""

# region book:procurement-auction-setup
import asyncio

from agents.protocols.auction import AuctionType, ProcurementAuction
from models.procurement import PurchaseOrder, SupplierBid
from models.supplier import Supplier, SupplierRating


async def demo_procurement_auction() -> None:
    purchase_order = PurchaseOrder(
        product_id="PROD-XYZ",
        quantity=500,
        deadline_days=14,
        maximum_acceptable_price=12000.0,
    )
    auction = ProcurementAuction(
        auction_id=f"AUC-{purchase_order.id}",
        purchase_order=purchase_order,
        auction_type=AuctionType.REVERSE,
        max_rounds=1,
    )
    suppliers = [
        Supplier(
            "sup-alpha",
            "Alpha Supplies",
            SupplierRating.STANDARD,
            ["PROD-XYZ"],
            1.05,
            0.9,
            0.95,
        ),
        Supplier(
            "sup-beta",
            "Beta Goods",
            SupplierRating.PREFERRED,
            ["PROD-XYZ"],
            1.0,
            0.8,
            0.98,
        ),
    ]

    for supplier in suppliers:
        auction.register_supplier(supplier)

    # endregion book:procurement-auction-setup

    # region book:procurement-auction-run
    if not await auction.start_auction():
        print("Auction failed to start.")
        return

    bids = [
        SupplierBid(
            supplier_id="sup-alpha",
            purchase_order_id=purchase_order.id,
            price=11500.0,
            delivery_days=10,
            quality_guarantee=0.92,
        ),
        SupplierBid(
            supplier_id="sup-beta",
            purchase_order_id=purchase_order.id,
            price=10800.0,
            delivery_days=9,
            quality_guarantee=0.95,
        ),
    ]

    for bid in bids:
        auction.submit_bid(bid)

    winner = await auction.finalize_auction()
    if winner is None:
        print("No winner.")
        return

    print(f"Winner: {winner.supplier_id} price=${winner.price:.2f} days={winner.delivery_days}")


if __name__ == "__main__":
    asyncio.run(demo_procurement_auction())

# endregion book:procurement-auction-run
