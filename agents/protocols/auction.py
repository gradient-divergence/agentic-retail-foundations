"""
Auction protocol implementation for procurement in retail settings.

This module implements various auction protocols used in procurement and supplier bidding scenarios.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from models.procurement import PurchaseOrder, PurchaseOrderStatus, SupplierBid
from models.supplier import Supplier


class AuctionType(str, Enum):
    """Types of procurement auctions supported."""

    ENGLISH = "ENGLISH"  # Increasing price, highest bidder wins (for selling)
    DUTCH = "DUTCH"  # Decreasing price, first to accept wins
    SEALED_BID = "SEALED_BID"  # One round of sealed bids
    REVERSE = "REVERSE"  # Decreasing price, lowest bidder wins (for buying)


class AuctionStatus(str, Enum):
    """Status of an auction."""

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class ProcurementAuction:
    """
    Implements auction protocols for procurement scenarios.
    """

    def __init__(
        self,
        auction_id: str,
        purchase_order: PurchaseOrder,
        auction_type: AuctionType = AuctionType.REVERSE,
        max_rounds: int = 3,
        min_participants: int = 2,
        reserve_price: float | None = None,
    ):
        """
        Initialize a procurement auction.

        Args:
            auction_id: Unique identifier for this auction
            purchase_order: The purchase order being auctioned
            auction_type: Type of auction mechanism to use
            max_rounds: Maximum number of bidding rounds
            min_participants: Minimum participants for a valid auction
            reserve_price: Optional maximum price the buyer is willing to pay
        """
        self.auction_id = auction_id
        self.purchase_order = purchase_order
        self.auction_type = auction_type
        self.max_rounds = max_rounds
        self.min_participants = min_participants
        self.reserve_price = reserve_price

        self.status = AuctionStatus.PENDING
        self.participants: dict[str, Supplier] = {}  # supplier_id -> Supplier
        self.bids: dict[str, list[SupplierBid]] = {}  # supplier_id -> list of bids
        self.current_round = 0
        self.current_best_bid: SupplierBid | None = None
        self.auction_history: list[dict[str, Any]] = []

    def register_supplier(self, supplier: Supplier) -> bool:
        """
        Register a supplier as a participant in the auction.

        Args:
            supplier: The supplier to register

        Returns:
            True if registration was successful, False otherwise
        """
        if self.status != AuctionStatus.PENDING:
            return False

        self.participants[supplier.supplier_id] = supplier
        self.bids[supplier.supplier_id] = []

        # Record the registration
        event = {
            "timestamp": datetime.now(),
            "action": "supplier_registered",
            "supplier_id": supplier.supplier_id,
            "supplier_name": supplier.name,
        }
        self.auction_history.append(event)

        return True

    def submit_bid(self, bid: SupplierBid) -> bool:
        """
        Submit a bid from a supplier.

        Args:
            bid: The bid being submitted

        Returns:
            True if the bid was accepted, False otherwise
        """
        if self.status != AuctionStatus.ACTIVE:
            return False

        if bid.supplier_id not in self.participants:
            return False

        # For reverse auctions, new bids must be lower than the best bid so far
        if self.auction_type == AuctionType.REVERSE and self.current_best_bid is not None and bid.price >= self.current_best_bid.price:
            return False

        # Record the bid
        self.bids[bid.supplier_id].append(bid)

        # Update best bid if this is better (lower for reverse auction)
        if self.current_best_bid is None or (self.auction_type == AuctionType.REVERSE and bid.price < self.current_best_bid.price):
            self.current_best_bid = bid

        # Record the bid in history
        event = {
            "timestamp": datetime.now(),
            "action": "bid_submitted",
            "supplier_id": bid.supplier_id,
            "price": bid.price,
            "delivery_days": bid.delivery_days,
        }
        self.auction_history.append(event)

        return True

    async def start_auction(self) -> bool:
        """
        Start the auction if conditions are met.

        Returns:
            True if the auction was started, False otherwise
        """
        if self.status != AuctionStatus.PENDING:
            return False

        if len(self.participants) < self.min_participants:
            self.status = AuctionStatus.FAILED
            event = {
                "timestamp": datetime.now(),
                "action": "auction_failed",
                "reason": f"Not enough participants ({len(self.participants)} < {self.min_participants})",
            }
            self.auction_history.append(event)
            return False

        self.status = AuctionStatus.ACTIVE
        self.current_round = 1

        # Record the start
        event = {
            "timestamp": datetime.now(),
            "action": "auction_started",
            "participants": list(self.participants.keys()),
            "auction_type": self.auction_type.value,
        }
        self.auction_history.append(event)

        return True

    async def advance_round(self) -> bool:
        """
        Move to the next round of bidding.

        Returns:
            True if advanced to next round, False if auction ended
        """
        if self.status != AuctionStatus.ACTIVE:
            return False

        if self.current_round >= self.max_rounds:
            await self.finalize_auction()
            return False

        self.current_round += 1

        # Record the round change
        event = {
            "timestamp": datetime.now(),
            "action": "round_advanced",
            "new_round": self.current_round,
            "max_rounds": self.max_rounds,
            "current_best_price": (self.current_best_bid.price if self.current_best_bid else None),
        }
        self.auction_history.append(event)

        return True

    async def finalize_auction(self) -> SupplierBid | None:
        """
        Finalize the auction and determine the winner.

        Returns:
            The winning bid, if any
        """
        if self.status != AuctionStatus.ACTIVE:
            return None

        self.status = AuctionStatus.COMPLETED
        winning_bid = self.current_best_bid

        if winning_bid is None:
            self.status = AuctionStatus.FAILED
            self.purchase_order.status = PurchaseOrderStatus.CANCELLED

            event = {
                "timestamp": datetime.now(),
                "action": "auction_failed",
                "reason": "No valid bids received",
            }
            self.auction_history.append(event)
            return None

        # Check reserve price for reverse auctions
        if self.auction_type == AuctionType.REVERSE and self.reserve_price is not None and winning_bid.price > self.reserve_price:
            self.status = AuctionStatus.FAILED
            self.purchase_order.status = PurchaseOrderStatus.CANCELLED

            event = {
                "timestamp": datetime.now(),
                "action": "auction_failed",
                "reason": "All bids exceeded reserve price",
            }
            self.auction_history.append(event)
            return None

        # Update the purchase order with the winning supplier and bid details
        self.purchase_order.selected_supplier_id = winning_bid.supplier_id
        self.purchase_order.winning_bid_details = winning_bid
        self.purchase_order.status = PurchaseOrderStatus.AWARDED

        # Record the auction completion
        event = {
            "timestamp": datetime.now(),
            "action": "auction_completed",
            "winning_supplier": winning_bid.supplier_id,
            "winning_price": winning_bid.price,
            "delivery_days": winning_bid.delivery_days,
        }
        self.auction_history.append(event)

        return winning_bid

    def cancel_auction(self, reason: str) -> bool:
        """
        Cancel an active auction.

        Args:
            reason: Reason for cancellation

        Returns:
            True if cancelled successfully, False otherwise
        """
        if self.status not in [AuctionStatus.PENDING, AuctionStatus.ACTIVE]:
            return False

        self.status = AuctionStatus.CANCELLED
        self.purchase_order.status = PurchaseOrderStatus.CANCELLED

        # Record the cancellation
        event = {
            "timestamp": datetime.now(),
            "action": "auction_cancelled",
            "reason": reason,
        }
        self.auction_history.append(event)

        return True
