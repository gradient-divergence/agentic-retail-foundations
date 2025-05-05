"""
Demonstrates the Inventory Collaboration Network.
"""

import asyncio
import random
import logging
import pandas as pd

# Import necessary components from the project structure
from agents.protocols.inventory_sharing import (
    InventoryCollaborationNetwork,
)
from models.store import Store  # Assumed location

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Helper function to print store inventories
def print_inventories(network: InventoryCollaborationNetwork):
    data = []
    for sid, st in network.stores.items():
        if hasattr(st, "inventory") and isinstance(st.inventory, dict):
            for pid, pos in st.inventory.items():
                # Use InventoryPosition attributes/methods
                current_stock = getattr(pos, "current_stock", "N/A")
                target_stock = getattr(pos, "target_stock", "N/A")
                # status = getattr(pos, "get_status", lambda: None)() # get_status may not exist
                data.append(
                    {
                        "Store ID": sid,
                        "Product ID": pid,
                        "Current Stock": current_stock,
                        "Target Stock": target_stock,
                        # "Status": status,
                    }
                )
        else:
            data.append(
                {
                    "Store ID": sid,
                    "Product ID": "N/A",
                    "Current Stock": "N/A",
                    "Target Stock": "N/A",
                    # "Status": "N/A",
                }
            )
    if data:
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
    else:
        print("No inventory data available.")


async def demo_collaborative_inventory_sharing():
    """Runs the collaborative inventory sharing demonstration with multiple cycles."""
    logger.info("Initializing Collaborative Inventory Sharing Demo...")

    # --- Results Storage ---
    results = {
        "cycle1_opportunities": [],
        "cycle1_executed": [],
        "cycle2_opportunities": [],
        "cycle2_executed": [],
        "final_inventory": [],
        "log_messages": [],  # Capture log messages
    }
    results["log_messages"].append(
        "Initializing Collaborative Inventory Sharing Demo..."
    )

    # Create Network
    network = InventoryCollaborationNetwork(max_transfer_distance=50.0)
    results["log_messages"].append("Inventory Collaboration Network created.")
    logger.info("Inventory Collaboration Network created.")

    # Create and Register Stores
    stores = [
        Store(
            store_id="STORE_A",
            name="Store A",
            location=(10, 10),
            initial_cooperation_score=1.1,
        ),
        Store(
            store_id="STORE_B",
            name="Store B",
            location=(20, 30),
            initial_cooperation_score=0.9,
        ),
        Store(
            store_id="STORE_C",
            name="Store C",
            location=(50, 50),
            initial_cooperation_score=1.0,
        ),
        Store(
            store_id="STORE_D",
            name="Store D",
            location=(15, 15),
            initial_cooperation_score=1.3,
        ),
    ]
    for store in stores:
        network.register_store(store)
    results["log_messages"].append(f"Registered {len(stores)} stores.")
    logger.info(f"Registered {len(stores)} stores.")

    # Add Products and Initial Inventory (Simulated)
    products = {
        "P1": {"name": "Gadget", "unit_cost": 10.0},
        "P2": {"name": "Widget", "unit_cost": 5.0},
    }
    initial_inventory = {
        "STORE_A": {"P1": 50, "P2": 100},
        "STORE_B": {"P1": 10, "P2": 30},
        "STORE_C": {"P1": 150, "P2": 20},
        "STORE_D": {"P1": 5, "P2": 5},
    }
    target_levels = {
        "STORE_A": {"P1": 40, "P2": 80},
        "STORE_B": {"P1": 25, "P2": 50},
        "STORE_C": {"P1": 100, "P2": 40},
        "STORE_D": {"P1": 20, "P2": 30},
    }
    results["log_messages"].append("Initializing inventory...")
    logger.info("Initializing inventory...")
    for store in stores:
        for pid, pdata in products.items():
            current_stock = initial_inventory.get(store.store_id, {}).get(pid, 0)
            target_stock = target_levels.get(store.store_id, {}).get(pid, 0)
            sales_rate = target_stock / 10.0 if target_stock > 0 else 1.0
            if hasattr(store, "add_product"):
                store.add_product(pid, current_stock, target_stock, sales_rate)

    results["log_messages"].append("Initial inventory set.")
    logger.info("Initial inventory set.")
    print("\n--- Initial Inventory Levels ---")
    print_inventories(network)

    # --- Cycle 1 ---
    results["log_messages"].append("\n=== Cycle 1 ===")
    logger.info("\n=== Cycle 1 ===")
    results["log_messages"].append("Identifying transfer opportunities...")
    logger.info("Identifying transfer opportunities...")
    opportunities_cycle1 = await network.identify_transfer_opportunities()
    results["log_messages"].append(
        f"Found {len(opportunities_cycle1)} potential transfers."
    )
    logger.info(f"Found {len(opportunities_cycle1)} potential transfers.")

    if opportunities_cycle1:
        # Store and log opportunities
        for i, opp in enumerate(opportunities_cycle1):
            sender = network.stores.get(opp["sender_id"])
            receiver = network.stores.get(opp["receiver_id"])
            if sender and receiver:
                log_msg = f"{i + 1}. {sender.name} -> {receiver.name}: {opp['quantity']} of {opp['product_id']} (Value: {opp['net_value']:.2f})"
                results["log_messages"].append(log_msg)
                results["cycle1_opportunities"].append(
                    {**opp, "sender_name": sender.name, "receiver_name": receiver.name}
                )
            else:
                results["log_messages"].append(
                    f"Warning: Sender/Receiver not found for opp: {opp}"
                )
                logger.warning(f"Sender or Receiver not found for opportunity: {opp}")

        # Simulate Execution (Warning: execute_transfer not fully implemented on Network)
        approved_transfers_c1 = [o for o in opportunities_cycle1 if o["net_value"] > 0]
        results["log_messages"].append(
            f"\nExecuting {len(approved_transfers_c1)} approved transfers..."
        )
        logger.info(f"Executing {len(approved_transfers_c1)} approved transfers...")
        executed_count_c1 = 0
        for proposal in approved_transfers_c1:
            if random.random() > 0.1:  # Simulate 90% chance
                if hasattr(network, "execute_transfer"):  # Check if method exists
                    # This won't actually change inventory state until implemented
                    # network.execute_transfer(proposal)
                    executed_count_c1 += 1
                    # Store mock executed data
                    results["cycle1_executed"].append(proposal)
                else:
                    logger.warning("Network does not have execute_transfer method.")
                    results["log_messages"].append(
                        "Warning: Network.execute_transfer not implemented."
                    )
            else:
                skip_msg = f"Simulated skip/fail for transfer: {proposal.get('product_id')} from {proposal.get('sender_id')} to {proposal.get('receiver_id')}"
                results["log_messages"].append(skip_msg)
                logger.info(skip_msg)
        results["log_messages"].append(
            f"Simulated execution of {executed_count_c1} transfers."
        )
        logger.info(f"Simulated execution of {executed_count_c1} transfers.")
    else:
        results["log_messages"].append("No transfer opportunities found in Cycle 1.")
        logger.info("No transfer opportunities found in Cycle 1.")

    # --- Simulate Market Changes ---
    results["log_messages"].append("\n=== Simulating Market Changes ===")
    logger.info("\n=== Simulating Market Changes ===")
    try:
        stores[2].update_sales_rate("P1", 15)  # Store C Gadget sales increase
        log_msg_c = "- Store C (Mall Store) Gadget sales increased."
        results["log_messages"].append(log_msg_c)
        logger.info(log_msg_c)
        stores[1].update_sales_rate("P2", 3)  # Store B Widget sales decrease
        log_msg_b = "- Store B (Suburban Store) Widget sales decreased."
        results["log_messages"].append(log_msg_b)
        logger.info(log_msg_b)
    except Exception as e:
        logger.error(f"Error simulating market changes: {e}")
        results["log_messages"].append(f"Error simulating market changes: {e}")

    # --- Cycle 2 ---
    results["log_messages"].append("\n=== Cycle 2 ===")
    logger.info("\n=== Cycle 2 ===")
    results["log_messages"].append(
        "Identifying transfer opportunities after changes..."
    )
    logger.info("Identifying transfer opportunities after changes...")
    opportunities_cycle2 = await network.identify_transfer_opportunities()
    results["log_messages"].append(
        f"Found {len(opportunities_cycle2)} potential transfers."
    )
    logger.info(f"Found {len(opportunities_cycle2)} potential transfers.")

    if opportunities_cycle2:
        # Store and log opportunities
        for i, opp in enumerate(opportunities_cycle2):
            sender = network.stores.get(opp["sender_id"])
            receiver = network.stores.get(opp["receiver_id"])
            if sender and receiver:
                log_msg = f"{i + 1}. {sender.name} -> {receiver.name}: {opp['quantity']} of {opp['product_id']} (Value: {opp['net_value']:.2f})"
                results["log_messages"].append(log_msg)
                results["cycle2_opportunities"].append(
                    {**opp, "sender_name": sender.name, "receiver_name": receiver.name}
                )
            else:
                results["log_messages"].append(
                    f"Warning: Sender/Receiver not found for opp: {opp}"
                )
                logger.warning(f"Sender or Receiver not found for opportunity: {opp}")

        # Simulate Execution
        approved_transfers_c2 = [o for o in opportunities_cycle2 if o["net_value"] > 0]
        results["log_messages"].append(
            f"\nExecuting {len(approved_transfers_c2)} approved transfers..."
        )
        logger.info(f"Executing {len(approved_transfers_c2)} approved transfers...")
        executed_count_c2 = 0
        for proposal in approved_transfers_c2:
            if random.random() > 0.1:  # Simulate 90% chance
                if hasattr(network, "execute_transfer"):  # Check if method exists
                    # network.execute_transfer(proposal) # Still not implemented
                    executed_count_c2 += 1
                    results["cycle2_executed"].append(proposal)
                else:
                    logger.warning("Network does not have execute_transfer method.")
                    results["log_messages"].append(
                        "Warning: Network.execute_transfer not implemented."
                    )
            else:
                skip_msg = f"Simulated skip/fail for transfer: {proposal.get('product_id')} from {proposal.get('sender_id')} to {proposal.get('receiver_id')}"
                results["log_messages"].append(skip_msg)
                logger.info(skip_msg)
        results["log_messages"].append(
            f"Simulated execution of {executed_count_c2} transfers."
        )
        logger.info(f"Simulated execution of {executed_count_c2} transfers.")
    else:
        results["log_messages"].append("No transfer opportunities found in Cycle 2.")
        logger.info("No transfer opportunities found in Cycle 2.")

    # --- Final Inventory Status ---
    results["log_messages"].append("\n=== Final Status ===")
    logger.info("\n=== Final Status ===")
    final_inventory_summary = []
    try:
        for st in stores:
            store_inv_details = []
            transfer_history = getattr(st, "transfer_history", [])
            out_trans = len(
                [t for t in transfer_history if t.get("direction") == "out"]
            )
            in_trans = len([t for t in transfer_history if t.get("direction") == "in"])

            if hasattr(st, "inventory") and isinstance(st.inventory, dict):
                for pid, pos in st.inventory.items():
                    current_stock = getattr(pos, "current_stock", "N/A")
                    status = getattr(pos, "get_status", lambda: None)()
                    status_value = status.value if hasattr(status, "value") else "N/A"
                    days_supply = getattr(pos, "days_of_supply", lambda: -1)()
                    days_supply_str = (
                        f"{days_supply:.1f}"
                        if isinstance(days_supply, (int, float)) and days_supply >= 0
                        else "N/A"
                    )
                    store_inv_details.append(
                        {
                            "Product ID": pid,
                            "Stock": current_stock,
                            "Days Supply": days_supply_str,
                            "Status": status_value,
                        }
                    )
            final_inventory_summary.append(
                {
                    "Store Name": st.name,
                    "Cooperation Score": f"{st.cooperation_score:.2f}",
                    "Transfers Out": out_trans,
                    "Transfers In": in_trans,
                    "Inventory": store_inv_details,
                }
            )
        results["final_inventory"] = final_inventory_summary
    except Exception as e:
        logger.error(f"Error gathering final inventory status: {e}", exc_info=True)
        results["log_messages"].append(f"Error gathering final inventory status: {e}")

    # --- Print Results ---
    print("\n--- Collaborative Inventory Sharing Results ---")
    # Cycle 1
    print("\n### Cycle 1")
    if results["cycle1_opportunities"]:
        print("**Potential Transfers Identified:**")
        opp1_df = pd.DataFrame(results["cycle1_opportunities"])
        print(
            opp1_df[
                ["sender_name", "receiver_name", "product_id", "quantity", "net_value"]
            ].to_string(index=False)
        )
    if results["cycle1_executed"]:
        print(f"\n**Simulated Executed Transfers:** {len(results['cycle1_executed'])}")
    else:
        print("\nNo transfers simulated executed.")

    # Cycle 2
    print("\n### Cycle 2 (After Market Changes)")
    if results["cycle2_opportunities"]:
        print("**Potential Transfers Identified:**")
        opp2_df = pd.DataFrame(results["cycle2_opportunities"])
        print(
            opp2_df[
                ["sender_name", "receiver_name", "product_id", "quantity", "net_value"]
            ].to_string(index=False)
        )
    if results["cycle2_executed"]:
        print(f"\n**Simulated Executed Transfers:** {len(results['cycle2_executed'])}")
    else:
        print("\nNo transfers simulated executed.")

    # Final Inventory
    print("\n### Final Inventory Status")
    if results["final_inventory"]:
        flat_inventory = []
        for store_info in results["final_inventory"]:
            for item in store_info["Inventory"]:
                flat_inventory.append(
                    {
                        "Store Name": store_info["Store Name"],
                        "Product ID": item["Product ID"],
                        "Stock": item["Stock"],
                        "Days Supply": item["Days Supply"],
                        "Status": item["Status"],
                        "Coop Score": store_info["Cooperation Score"],
                        "Transfers In": store_info["Transfers In"],
                        "Transfers Out": store_info["Transfers Out"],
                    }
                )
        if flat_inventory:
            final_inv_df = pd.DataFrame(flat_inventory)
            print(final_inv_df.to_string(index=False))
        else:
            print("No detailed inventory data to display.")
    else:
        print("No final inventory data generated.")

    # Optionally print detailed logs
    # print("\n--- Detailed Log ---")
    # print("\n".join(results["log_messages"]))
    print("-----------------------------------------")

    logger.info("Collaborative Inventory Sharing Demo completed.")


if __name__ == "__main__":
    # Add missing import

    asyncio.run(demo_collaborative_inventory_sharing())
