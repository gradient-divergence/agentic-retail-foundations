"""
Demonstration of the fulfillment planning system.
"""

import logging
import random
import time

# Import the refactored components
from models.fulfillment import Associate, Item, Order, OrderLineItem
from utils.planning import FulfillmentPlanner, StoreLayout

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def demo_fulfillment_system(mo_instance=None):  # Allow passing mo for notebook context
    """Demonstrate the fulfillment optimization system with a sample scenario."""
    logger.info("\n--- Starting Fulfillment Optimization Demo ---")
    # Create store layout
    store = StoreLayout(width=50, height=40)
    logger.info("Store layout created.")
    # Add sections
    store.add_section((5, 15), (5, 15), "Grocery")
    store.add_section((20, 30), (5, 15), "Produce")
    store.add_section((35, 45), (5, 15), "Dairy")
    store.add_section((5, 15), (20, 30), "Frozen")
    store.add_section((20, 30), (20, 30), "Electronics")
    store.add_section((35, 45), (20, 30), "Apparel")
    logger.info("Store sections defined.")
    # Add obstacles (walls, displays, etc.)
    for x in range(0, 50, 10):
        for y in range(0, 40):
            if y % 5 != 0:  # Leave gaps for aisles
                store.add_obstacle(x, y)
    logger.info("Obstacles added.")
    # Create items
    items = []
    # Grocery items
    for i in range(20):
        x = random.randint(6, 14)
        y = random.randint(6, 14)
        items.append(Item(f"G{i}", f"Grocery Item {i}", "grocery", (x, y)))
    # Produce items
    for i in range(15):
        x = random.randint(21, 29)
        y = random.randint(6, 14)
        items.append(
            Item(
                f"P{i}",
                f"Produce Item {i}",
                "produce",
                (x, y),
                temperature_zone="refrigerated",
                handling_time=1.2,
            )
        )
    # Dairy items
    for i in range(10):
        x = random.randint(36, 44)
        y = random.randint(6, 14)
        items.append(
            Item(
                f"D{i}",
                f"Dairy Item {i}",
                "dairy",
                (x, y),
                temperature_zone="refrigerated",
                handling_time=1.1,
            )
        )
    # Frozen items
    for i in range(12):
        x = random.randint(6, 14)
        y = random.randint(21, 29)
        items.append(
            Item(
                f"F{i}",
                f"Frozen Item {i}",
                "frozen",
                (x, y),
                temperature_zone="frozen",
                handling_time=1.3,
            )
        )
    # Electronics items
    for i in range(8):
        x = random.randint(21, 29)
        y = random.randint(21, 29)
        items.append(
            Item(
                f"E{i}",
                f"Electronics Item {i}",
                "electronics",
                (x, y),
                handling_time=1.5,
                fragility=0.8,
            )
        )
    # Apparel items
    for i in range(15):
        x = random.randint(36, 44)
        y = random.randint(21, 29)
        items.append(
            Item(
                f"A{i}",
                f"Apparel Item {i}",
                "apparel",
                (x, y),
                handling_time=1.4,
                fragility=0.3,
            )
        )
    logger.info(f"Created {len(items)} sample items.")
    # Create orders
    orders = []
    for i in range(10):
        num_items_in_order = random.randint(3, 8)
        # Sample physical Item objects to decide which products are in the order
        sampled_physical_items = random.sample(items, num_items_in_order)

        # Create OrderLineItem objects based on the sampled physical items
        order_line_items = []
        for physical_item in sampled_physical_items:
            # Use item_id from the physical Item as the product_id for the OrderLineItem
            # Assign random quantity and price for the demo
            order_line_items.append(
                OrderLineItem(
                    product_id=physical_item.item_id,  # Use the correct ID
                    quantity=random.randint(1, 5),
                    price=round(random.uniform(1.0, 50.0), 2),
                )
            )

        # Create the Order object using the correct constructor signature
        # Add a dummy customer_id and remove the invalid priority/due_time args
        orders.append(
            Order(
                order_id=f"ORD{i}",
                items=order_line_items,  # Pass the list of OrderLineItem
                customer_id=f"CUST{random.randint(100, 999)}",
                # created_at, status, etc., will use defaults from Order definition
            )
        )
    logger.info(f"Created {len(orders)} sample orders.")
    # Create associates
    associates = [
        Associate(
            "A1",
            "Alex",
            efficiency=1.2,
            authorized_zones=["ambient", "refrigerated", "frozen"],
            current_location=(0, 0),
            shift_end_time=240,
        ),
        Associate(
            "A2",
            "Bailey",
            efficiency=1.0,
            authorized_zones=["ambient", "refrigerated"],
            current_location=(0, 20),
            shift_end_time=180,
        ),
        Associate(
            "A3",
            "Casey",
            efficiency=0.9,
            authorized_zones=["ambient"],
            current_location=(25, 0),
            shift_end_time=120,
        ),
    ]
    logger.info(f"Created {len(associates)} associates.")
    # Create fulfillment planner
    planner = FulfillmentPlanner(store)
    # Add orders and associates
    for order in orders:
        planner.add_order(order)
    for associate in associates:
        planner.add_associate(associate)
    logger.info("Added orders and associates to planner.")
    # Generate plan
    logger.info("Generating fulfillment plan...")
    start_time = time.time()
    planner.plan()  # Call plan
    end_time = time.time()
    logger.info(f"Plan generated in {end_time - start_time:.3f} seconds")
    # Explain plan
    explanation_text = planner.explain_plan()
    print(explanation_text)  # Print explanation to console
    # Visualize plan
    logger.info("Visualizing plan...")
    plan_figure = planner.visualize_plan()

    # Handle notebook vs script context
    if mo_instance:
        # Combine outputs for Marimo
        return mo_instance.vstack(
            [
                mo_instance.md("### Fulfillment Plan Explanation"),
                mo_instance.md(f"```\n{explanation_text}\n```"),  # Display explanation as code block
                mo_instance.md("### Fulfillment Plan Visualization"),
                (plan_figure if plan_figure else mo_instance.md("_Could not generate visualization._")),
            ]
        )
    else:
        # If running as script, just print explanation
        # (Visualization requires further handling like saving to file)
        if plan_figure:
            print("(Visualization generated, not shown in script output)")
        return explanation_text  # Return text explanation


if __name__ == "__main__":
    demo_fulfillment_system()
