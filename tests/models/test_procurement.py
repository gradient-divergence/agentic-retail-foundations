import asyncio
from datetime import datetime, timedelta

from models.procurement import ProcurementAuction, SupplierBid
from models.supplier import Supplier, SupplierRating


def test_delivery_can_use_remaining_partial_day():
    auction = ProcurementAuction()
    auction.register_supplier(Supplier("a", "A", SupplierRating.PREFERRED, ["sku"]))
    order_id = auction.create_purchase_order("sku", 150, deadline_days=2, budget=2000)
    order = auction.purchase_orders[order_id]
    order.required_delivery_date = datetime.now() + timedelta(days=1, hours=23)
    bids = asyncio.run(auction.collect_bids(order_id, bid_window_seconds=0))
    assert len(bids) == 1
    assert bids[0].delivery_days == 1


def test_weighted_auction_rejects_over_budget_bid():
    auction = ProcurementAuction()
    auction.register_supplier(Supplier("a", "A", SupplierRating.PREFERRED, ["sku"]))
    order_id = auction.create_purchase_order("sku", 10, deadline_days=7, budget=100)
    auction.bids[order_id] = [SupplierBid("a", order_id, 101, 1, 0.99)]
    assert auction.evaluate_bids(order_id) is None
