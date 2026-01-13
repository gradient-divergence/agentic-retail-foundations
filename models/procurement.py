"""
Procurement and auction mechanism classes for supplier selection and purchase order management in retail MAS.
"""

# region book:procurement-auction-procurement-models
import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# Import supplier models
from models.supplier import Supplier, SupplierStatus

logger = logging.getLogger(__name__)


@dataclass
class SupplierBid:
    """
    Data class for a supplier's bid on a purchase order.
    NOTE: This specific bid structure is related to Procurement Auction.
    Distinguish from the general `Bid` in `models.task` used for CNP.
    """

    supplier_id: str
    purchase_order_id: str
    price: float
    delivery_days: int
    quality_guarantee: float  # Example: 0.95 means 95% quality
    timestamp: datetime = field(default_factory=datetime.now)


class PurchaseOrderStatus(Enum):
    """Status of a Purchase Order"""

    CREATED = "CREATED"
    BIDDING = "BIDDING"
    AWARDED = "AWARDED"
    REJECTED = "REJECTED"  # If no suitable bid found
    FULFILLED = "FULFILLED"
    CANCELLED = "CANCELLED"


@dataclass
class PurchaseOrder:
    """
    Data class for a purchase order in the procurement process.
    """

    product_id: str
    quantity: int
    # Using deadline_days to calculate required_delivery_date on init
    deadline_days: int
    maximum_acceptable_price: float
    quality_threshold: float = 0.90  # Default minimum quality
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: PurchaseOrderStatus = PurchaseOrderStatus.CREATED
    required_delivery_date: datetime = field(init=False)
    creation_time: datetime = field(default_factory=datetime.now)
    selected_supplier_id: str | None = None
    winning_bid_details: SupplierBid | None = None  # Store the winning bid object

    def __post_init__(self) -> None:
        self.required_delivery_date = self.creation_time + timedelta(days=self.deadline_days)
        if self.quantity <= 0:
            raise ValueError("Purchase order quantity must be positive.")
        if self.maximum_acceptable_price < 0:
            raise ValueError("Maximum acceptable price cannot be negative.")
        if not (0 <= self.quality_threshold <= 1):
            raise ValueError("Quality threshold must be between 0 and 1.")


# endregion book:procurement-auction-procurement-models


class ProcurementAuction:
    """
    Manages procurement auctions, supplier registration, and bid evaluation for purchase orders.
    Uses Supplier, SupplierRating, SupplierBid, PurchaseOrder models.
    """

    def __init__(self) -> None:
        self.purchase_orders: dict[str, PurchaseOrder] = {}
        self.suppliers: dict[str, Supplier] = {}
        self.bids: dict[str, list[SupplierBid]] = defaultdict(list)  # PO ID -> List of Bids

    def register_supplier(self, supplier: Supplier) -> None:
        """
        Register a supplier for participation in auctions.
        """
        if not isinstance(supplier, Supplier):
            raise TypeError("Can only register Supplier objects.")
        if supplier.supplier_id in self.suppliers:
            logger.warning("Re-registering supplier %s.", supplier.supplier_id)
        self.suppliers[supplier.supplier_id] = supplier
        logger.info(
            "Supplier %s (%s) registered for auctions.",
            supplier.name,
            supplier.supplier_id,
        )

    def create_purchase_order(
        self,
        product_id: str,
        quantity: int,
        deadline_days: int,
        budget: float,  # Renamed from maximum_price for clarity
        quality_threshold: float = 0.9,
    ) -> str:
        """
        Create a new purchase order and add it to the auction manager.
        Returns the PO ID.
        """
        po = PurchaseOrder(
            product_id=product_id,
            quantity=quantity,
            deadline_days=deadline_days,
            maximum_acceptable_price=budget,
            quality_threshold=quality_threshold,
            status=PurchaseOrderStatus.BIDDING,  # Start in BIDDING state
        )
        self.purchase_orders[po.id] = po
        # Initialize bid list for this PO
        self.bids[po.id] = []
        logger.info(
            "Created Purchase Order %s for %sx%s, budget $%s, deadline %s days.",
            po.id,
            quantity,
            product_id,
            budget,
            deadline_days,
        )
        return po.id

    async def collect_bids(
        self,
        po_id: str,
        bid_window_seconds: float = 0.1,  # Reduced default window
    ) -> list[SupplierBid]:
        """
        Collect bids from all registered and capable suppliers for a specific PO.
        Simulates suppliers calculating and submitting bids.
        """
        if po_id not in self.purchase_orders:
            logger.error("Cannot collect bids for non-existent PO %s.", po_id)
            return []

        purchase_order = self.purchase_orders[po_id]
        if purchase_order.status != PurchaseOrderStatus.BIDDING:
            logger.warning(
                "PO %s is not in BIDDING state (current: %s). Cannot collect new bids.",
                po_id,
                purchase_order.status.name,
            )
            return self.bids.get(po_id, [])  # Return existing bids if any

        logger.info(
            "Collecting bids for PO %s (%sx%s).",
            po_id,
            purchase_order.quantity,
            purchase_order.product_id,
        )

        potential_bidders = [
            s
            for s in self.suppliers.values()
            if s.can_supply(purchase_order.product_id) and s.status == SupplierStatus.ACTIVE
        ]
        logger.info("Contacting %s potential suppliers...", len(potential_bidders))

        # Simulate bid calculation concurrently (if supplier logic allows)
        # For now, assume calculate_bid is synchronous within the Supplier model itself
        # In a real system, this might involve sending requests to supplier agents
        calculated_bids = []
        for supplier in potential_bidders:
            # Supplier internally decides if they *can* meet constraints
            bid = self._simulate_supplier_bid_calculation(supplier, purchase_order)
            if bid:
                calculated_bids.append(bid)
                logger.info("Received bid from %s.", supplier.name)
            # else: Supplier cannot meet constraints or chooses not to bid

        # Store the collected bids
        self.bids[po_id].extend(calculated_bids)

        # Simulate bidding window delay
        await asyncio.sleep(bid_window_seconds)
        logger.info(
            "Bid collection finished for PO %s. Total bids: %s.",
            po_id,
            len(self.bids[po_id]),
        )
        return self.bids[po_id]

    def _simulate_supplier_bid_calculation(
        self, supplier: Supplier, purchase_order: PurchaseOrder
    ) -> SupplierBid | None:
        """Internal helper to simulate a supplier generating a bid based on its factors."""
        # This replicates the logic previously inside Supplier.calculate_bid
        # It should ideally live within the Supplier agent/model or be called via message

        if not supplier.can_supply(purchase_order.product_id):
            return None

        # Example Pricing Logic (can be much more complex)
        base_price_per_unit = 10.0 * supplier.cost_factor
        total_price = base_price_per_unit * purchase_order.quantity
        # Volume discounts
        if purchase_order.quantity > 1000:
            total_price *= 0.90
        elif purchase_order.quantity > 500:
            total_price *= 0.95

        # Example Delivery Time Logic
        delivery_days = int(max(1, (purchase_order.quantity / 150.0) * supplier.speed_factor))

        # Example Quality Logic
        # Higher rating and lower quality_factor (lower defect rate) improve guarantee
        quality_guarantee = min(
            0.99,
            0.80 + (supplier.rating.value * 0.06) * (1.0 / max(0.1, supplier.quality_factor)),
        )

        # --- Check Constraints ---
        days_until_required = (purchase_order.required_delivery_date - datetime.now()).days
        if delivery_days > days_until_required:
            # print(f"Supplier {supplier.name} cannot meet deadline for PO {purchase_order.id}")
            return None
        if quality_guarantee < purchase_order.quality_threshold:
            # print(f"Supplier {supplier.name} cannot meet quality for PO {purchase_order.id}")
            return None
        if total_price > purchase_order.maximum_acceptable_price:
            # print(f"Supplier {supplier.name} price too high for PO {purchase_order.id}")
            return None

        # If all constraints met, create the bid object
        return SupplierBid(
            supplier_id=supplier.supplier_id,
            purchase_order_id=purchase_order.id,
            price=round(total_price, 2),
            delivery_days=delivery_days,
            quality_guarantee=round(quality_guarantee, 3),
        )

    def evaluate_bids(self, po_id: str) -> str | None:
        """
        Evaluate collected bids for a PO and select the best supplier based on a scoring function.
        Updates the PO status and selected supplier.
        Returns the ID of the winning supplier, or None if no winner.
        """
        if po_id not in self.purchase_orders:
            logger.error("Cannot evaluate bids for non-existent PO %s.", po_id)
            return None

        purchase_order = self.purchase_orders[po_id]
        bids_to_evaluate = self.bids.get(po_id, [])

        if not bids_to_evaluate:
            logger.warning("No bids found for PO %s to evaluate.", po_id)
            purchase_order.status = PurchaseOrderStatus.REJECTED
            return None

        logger.info(
            "Evaluating %s bids for PO %s.",
            len(bids_to_evaluate),
            po_id,
        )

        best_score = float("inf")
        winning_bid: SupplierBid | None = None

        # Scoring weights (configurable)
        W_PRICE = 0.6
        W_DELIVERY = 0.3
        W_QUALITY = 0.1

        for bid in bids_to_evaluate:
            # Normalize factors (0-1 range, lower is better generally)
            price_norm = bid.price / max(1.0, purchase_order.maximum_acceptable_price)

            days_allowed = max(
                1,
                (purchase_order.required_delivery_date - bid.timestamp.replace(tzinfo=None)).days,
            )
            delivery_norm = bid.delivery_days / days_allowed

            # Quality score: higher guarantee is better, so invert for scoring (lower score is better)
            quality_norm = 1.0 - bid.quality_guarantee

            # Weighted score (lower is better)
            score = (price_norm * W_PRICE) + (delivery_norm * W_DELIVERY) + (quality_norm * W_QUALITY)

            logger.info(
                "Bidder: %s, Score: %.4f (P:%.2f, D:%.2f, Q:%.2f)",
                self.suppliers[bid.supplier_id].name,
                score,
                price_norm,
                delivery_norm,
                quality_norm,
            )

            if score < best_score:
                best_score = score
                winning_bid = bid

        if winning_bid:
            winner_supplier_id = winning_bid.supplier_id
            winner_supplier_name = self.suppliers[winner_supplier_id].name
            purchase_order.status = PurchaseOrderStatus.AWARDED
            purchase_order.selected_supplier_id = winner_supplier_id
            purchase_order.winning_bid_details = winning_bid  # Store the winning bid
            logger.info(
                "PO %s awarded to %s (Score: %.4f).",
                po_id,
                winner_supplier_name,
                best_score,
            )
            logger.info(
                "Bid details: $%.2f, %s days, %.2f%% quality.",
                winning_bid.price,
                winning_bid.delivery_days,
                winning_bid.quality_guarantee * 100,
            )
            return winner_supplier_id
        else:
            logger.warning("No suitable winner found among bids for PO %s.", po_id)
            purchase_order.status = PurchaseOrderStatus.REJECTED
            return None
