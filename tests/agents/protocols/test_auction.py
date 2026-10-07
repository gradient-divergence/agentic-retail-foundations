import asyncio
from dataclasses import replace

import pytest

from agents.protocols.auction import AuctionStatus, AuctionType, ProcurementAuction
from models.procurement import PurchaseOrder, PurchaseOrderStatus, SupplierBid
from models.supplier import Supplier, SupplierRating


def active_auction(auction_type=AuctionType.REVERSE, reserve_price=None):
    order = PurchaseOrder("sku", 10, 7, 100)
    auction = ProcurementAuction("auction", order, auction_type, reserve_price=reserve_price)
    for identifier in ["a", "b"]:
        auction.register_supplier(Supplier(identifier, identifier, SupplierRating.STANDARD, ["sku"]))
    assert asyncio.run(auction.start_auction())
    return auction


@pytest.mark.parametrize(
    "kind,prices,winner",
    [
        (AuctionType.REVERSE, [80, 70], "b"),
        (AuctionType.SEALED_BID, [80, 70], "b"),
        (AuctionType.ENGLISH, [70, 80], "b"),
        (AuctionType.DUTCH, [80, 70], "a"),
    ],
)
def test_auction_awards_by_its_declared_mechanism(kind, prices, winner):
    auction = active_auction(kind)
    for identifier, price in zip(["a", "b"], prices, strict=True):
        auction.submit_bid(SupplierBid(identifier, auction.purchase_order.id, price, 2, 0.95))
    result = asyncio.run(auction.finalize_auction())
    assert result.supplier_id == winner
    assert auction.purchase_order.status == PurchaseOrderStatus.AWARDED


@pytest.mark.parametrize("kind", list(AuctionType))
def test_tied_prices_keep_first_bid(kind):
    auction = active_auction(kind)
    first = SupplierBid("a", auction.purchase_order.id, 80, 2, 0.95)
    assert auction.submit_bid(first)
    auction.submit_bid(replace(first, supplier_id="b"))
    assert asyncio.run(auction.finalize_auction()) is first


@pytest.mark.parametrize("kind", list(AuctionType))
def test_no_bids_fail_auction(kind):
    auction = active_auction(kind)
    assert asyncio.run(auction.finalize_auction()) is None
    assert auction.status == AuctionStatus.FAILED
    assert auction.purchase_order.status == PurchaseOrderStatus.CANCELLED


@pytest.mark.parametrize("kind", [AuctionType.REVERSE, AuctionType.SEALED_BID, AuctionType.DUTCH])
@pytest.mark.parametrize("price,awarded", [(80, True), (81, False)])
def test_buyer_reserve_is_inclusive_ceiling(kind, price, awarded):
    auction = active_auction(kind, reserve_price=80)
    auction.submit_bid(SupplierBid("a", auction.purchase_order.id, price, 2, 0.95))
    assert (asyncio.run(auction.finalize_auction()) is not None) is awarded


@pytest.mark.parametrize(
    "field,value",
    [
        ("purchase_order_id", "wrong"),
        ("price", -1),
        ("price", float("nan")),
        ("price", 101),
        ("delivery_days", 8),
        ("delivery_days", -1),
        ("quality_guarantee", 0.1),
        ("quality_guarantee", float("nan")),
    ],
)
def test_auction_rejects_bids_that_violate_order_constraints(field, value):
    auction = active_auction()
    bid = replace(SupplierBid("a", auction.purchase_order.id, 80, 2, 0.95), **{field: value})
    assert not auction.submit_bid(bid)
    assert auction.current_best_bid is None
    assert asyncio.run(auction.finalize_auction()) is None
