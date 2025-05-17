"""
Demonstrates the Procurement Auction protocol.
"""

import asyncio
import logging
import random
from datetime import datetime  # Needed for bid simulation

import pandas as pd

from agents.protocols.auction import AuctionType, ProcurementAuction
from models.procurement import PurchaseOrder, SupplierBid  # Added SupplierBid

# Import necessary components from the project structure
from models.supplier import Supplier, SupplierRating

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Helper Function to Simulate Bid Calculation ---
def simulate_bid(supplier: Supplier, purchase_order: PurchaseOrder) -> SupplierBid | None:
    """Simulates a supplier calculating a bid, returning None if constraints not met."""
    # Basic check if supplier is capable (more checks could be added)
    if purchase_order.product_id not in supplier.product_capabilities:
        return None

    # Simulate price based on cost factor and random variation
    base_price_per_unit = 15.0  # Example base cost
    simulated_cost = base_price_per_unit * supplier.cost_factor * random.uniform(0.95, 1.05)
    total_price = simulated_cost * purchase_order.quantity

    # Simulate delivery days based on speed factor and quantity
    base_days = 5
    simulated_days = int(base_days + (purchase_order.quantity / 200.0) * supplier.speed_factor * random.uniform(0.8, 1.2))

    # Simulate quality guarantee based on quality factor (lower factor = better quality)
    base_quality = 0.85
    simulated_quality = min(
        0.99,
        base_quality + (1.0 / (supplier.quality_factor + 0.1)) * 0.15 * random.uniform(0.9, 1.1),
    )

    # --- Check Constraints (Must meet PO requirements) ---
    days_until_required = (purchase_order.required_delivery_date - datetime.now()).days
    if simulated_days > days_until_required:
        # print(f"DEBUG: {supplier.name} cannot meet deadline ({simulated_days} > {days_until_required})")
        return None  # Cannot meet deadline
    if simulated_quality < purchase_order.quality_threshold:
        # print(f"DEBUG: {supplier.name} cannot meet quality ({simulated_quality:.2f} < {purchase_order.quality_threshold})")
        return None  # Cannot meet quality
    if total_price > purchase_order.maximum_acceptable_price:
        # print(f"DEBUG: {supplier.name} price too high (${total_price:.2f} > ${purchase_order.maximum_acceptable_price})")
        return None  # Exceeds budget

    # If constraints met, create and return the bid
    return SupplierBid(
        supplier_id=supplier.supplier_id,
        purchase_order_id=purchase_order.id,
        price=round(total_price, 2),
        delivery_days=simulated_days,
        quality_guarantee=round(simulated_quality, 3),
    )


# --------------------------------------------------


async def demo_procurement_auction():
    """Runs the procurement auction demonstration."""
    logger.info("Initializing Procurement Auction Demo...")

    # Define Suppliers
    suppliers_data = {
        "sup1": Supplier(
            supplier_id="sup1",
            name="Alpha Supplies",
            rating=SupplierRating.STANDARD,
            product_capabilities=["PROD-XYZ", "PROD-ABC"],
            cost_factor=1.1,
            speed_factor=1.0,
            quality_factor=0.9,  # Lower is better?
        ),
        "sup2": Supplier(
            supplier_id="sup2",
            name="Beta Goods Inc.",
            rating=SupplierRating.STANDARD,
            product_capabilities=["PROD-XYZ", "PROD-DEF"],
            cost_factor=1.0,
            speed_factor=0.8,
            quality_factor=1.0,
        ),
        "sup3": Supplier(
            supplier_id="sup3",
            name="Gamma Distributors",
            rating=SupplierRating.PREFERRED,
            product_capabilities=["PROD-XYZ", "PROD-GHI"],
            cost_factor=1.2,
            speed_factor=0.9,
            quality_factor=0.8,
        ),
        "sup4": Supplier(
            supplier_id="sup4",
            name="Delta Partners",
            rating=SupplierRating.PROVISIONAL,
            product_capabilities=["PROD-XYZ"],
            cost_factor=0.9,
            speed_factor=1.2,
            quality_factor=1.1,
        ),
    }
    logger.info(f"Created {len(suppliers_data)} suppliers.")

    # Define the Purchase Order using correct field names
    purchase_order = PurchaseOrder(
        product_id="PROD-XYZ",
        quantity=1000,
        deadline_days=30,  # Example: delivery needed within 30 days
        maximum_acceptable_price=20000.0,  # Example budget
        quality_threshold=0.90,
    )
    # Access the generated ID
    po_id = purchase_order.id
    logger.info(f"Created Purchase Order {po_id} for {purchase_order.product_id}.")

    # Initialize Auction
    auction = ProcurementAuction(
        auction_id=f"AUC-{po_id}",
        purchase_order=purchase_order,
        auction_type=AuctionType.REVERSE,
        max_rounds=5,
    )

    # Register suppliers
    for supplier_id, supplier in suppliers_data.items():
        auction.register_supplier(supplier)
    logger.info("Initialized Reverse Procurement Auction and registered suppliers.")

    # Start the auction
    logger.info("Starting auction...")
    if not await auction.start_auction():
        logger.error("Auction failed to start (e.g., not enough participants).")
        return  # Exit if auction couldn't start
    logger.info("Auction active. Simulating bids...")

    # Simulate bid submission
    submitted_bids = 0
    for supplier_id, supplier in suppliers_data.items():
        bid = simulate_bid(supplier, purchase_order)  # Use helper to simulate
        if bid:
            if auction.submit_bid(bid):
                logger.info(
                    f"  Supplier {supplier.name} submitted bid: ${bid.price:.2f}, {bid.delivery_days} days, {bid.quality_guarantee:.2%} qual."
                )
                submitted_bids += 1
            else:
                logger.warning(
                    f"  Supplier {supplier.name}'s generated bid was rejected by auction protocol (e.g., too late, invalid round). Bid: ${bid.price:.2f}"
                )
        else:
            logger.info(f"  Supplier {supplier.name} did not submit a valid bid (constraints not met).")

    logger.info(f"{submitted_bids} bids were successfully submitted.")

    # Advance rounds if necessary (optional for simple demo)
    # while await auction.advance_round():
    #     logger.info(f"Advanced to round {auction.current_round}. Current best bid: {auction.current_best_bid}")
    #     # Simulate more bids for subsequent rounds if desired
    #     await asyncio.sleep(0.1)

    # Finalize the auction
    logger.info("Finalizing auction...")
    winning_bid = await auction.finalize_auction()  # finalize determines winner
    logger.info("Auction finished.")

    # Print results
    if winning_bid:
        winner_id = winning_bid.supplier_id
        # winning_bid = winner_info["winning_bid"] # Already have winning_bid
        logger.info(f"Auction Winner: {suppliers_data[winner_id].name} (ID: {winner_id})")
        logger.info(f"Winning Bid: Price=${winning_bid.price:.2f}, Days={winning_bid.delivery_days}, Quality={winning_bid.quality_guarantee:.2%}")
    else:
        logger.info("No winner determined in the auction.")

    # Print auction history
    try:
        history_df = pd.DataFrame(auction.auction_history)
        print("\n--- Procurement Auction Log ---")
        print(history_df.to_string())
        print("------------------------------\n")
    except Exception as e:
        logger.error(f"Could not display auction history as DataFrame: {e}")
        print("\n--- Raw Auction History ---")
        print(auction.auction_history)
        print("---------------------------\n")

    logger.info("Procurement Auction Demo completed.")


if __name__ == "__main__":
    asyncio.run(demo_procurement_auction())
