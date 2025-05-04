"""
Demonstrates the Product Launch Coordinator.
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta

# Import necessary components from the project structure
from agents.coordinators.product_launch import ProductLaunchCoordinator
from agents.cross_functional import (
    CustomerServiceAgent,
    MarketingAgent,
    PricingAgent,
    StoreOpsAgent,
    SupplyChainAgent,
)
from utils.planning import calculate_remediation_timeline  # Import the helper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demo_product_launch():
    """Runs the product launch coordination demonstration."""
    logger.info("Initializing Product Launch Demo...")

    # Initialize Functional Agents
    supply_chain = SupplyChainAgent()
    pricing = PricingAgent()
    marketing = MarketingAgent()
    store_ops = StoreOpsAgent()
    customer_service = CustomerServiceAgent()
    logger.info("Functional agents initialized.")

    # Initialize Coordinator
    coordinator = ProductLaunchCoordinator(
        supply_chain_agent=supply_chain,
        pricing_agent=pricing,
        marketing_agent=marketing,
        store_ops_agent=store_ops,
        customer_service_agent=customer_service,
    )
    logger.info("Product Launch Coordinator initialized.")

    # Define Product Launch Data (Example)
    product_launch_data = {
        "id": "PROD-007",
        "name": "Smart Thermostat Gen 3",
        "target_segments": ["tech_enthusiasts", "homeowners", "eco_conscious"],
        "first_month_forecast": 5000,
        "unit_cost": 45.00,
        "margin_targets": {"min_margin": 0.35, "target_margin": 0.45},
        "competitor_products": {
            "COMP-A": {"price": 129.99},
            "COMP-B": {"price": 149.50},
        },
        "messaging_guidelines": {
            "primary_message": "Save energy effortlessly",
            "key_features": ["AI learning", "Remote control", "Voice integration"],
        },
        "planogram": {"location": "Aisle 5, Shelf 3", "facings": 2},
        "training_materials": ["doc1.pdf", "video_intro.mp4"],
        "support_materials": ["faq.html", "troubleshooting_guide.pdf"],
        "anticipated_questions": ["Compatibility?", "Installation?", "Savings?"],
        "planned_launch_date": datetime.now() + timedelta(days=45),
        "lead_time_days": 30,  # Supply chain lead time
    }
    logger.info(
        f"Defined launch data for {product_launch_data['name']} (ID: {product_launch_data['id']})."
    )

    # Simulate initial planning phase (optional)
    logger.info("Simulating initial planning phase...")
    planning_tasks = [
        supply_chain.plan_inventory(product_launch_data),
        pricing.set_initial_price(product_launch_data),
        marketing.plan_launch_campaign(product_launch_data),
        store_ops.prepare_store_layout(product_launch_data),
        customer_service.prepare_support_team(product_launch_data),
    ]
    planning_results = await asyncio.gather(*planning_tasks)
    logger.info("Initial planning phase complete.")
    print("\n--- Initial Planning Results ---")
    print(pd.DataFrame(planning_results).to_string())
    print("------------------------------")

    # Coordinate Launch Readiness Check
    logger.info("\nStarting launch readiness coordination...")
    launch_status = await coordinator.coordinate_product_launch(product_launch_data)
    logger.info("Readiness check complete.")

    # Print Launch Status Summary using the new structure
    print("\n--- Launch Readiness Status ---")
    overall_status = launch_status.get("launch_status", "Unknown")
    projected_date_key = (
        "final_launch_date"
        if overall_status == "confirmed"
        else "suggested_new_launch_date"
    )
    projected_date = launch_status.get(projected_date_key, "N/A")

    print(f"Overall Status: {overall_status.upper()}")
    print(f"Projected Launch Date: {projected_date}")

    blockers = launch_status.get("blockers", [])
    if blockers:
        print("\nBlockers:")
        for blocker_str in blockers:  # Blockers are now strings
            print(f"- {blocker_str}")

    # Display agent statuses (more robustly)
    agent_results = launch_status.get("agent_statuses", [])
    agent_summary_data = []
    agent_name_list = [
        a.__class__.__name__
        for a in [supply_chain, pricing, marketing, store_ops, customer_service]
    ]  # Get names in order

    for i, result in enumerate(agent_results):
        agent_name = agent_name_list[i]
        if isinstance(result, BaseException):
            status = "ERROR"
            details = f"{type(result).__name__}: {result}"
            readiness_date_str = "N/A"
        else:
            status = result.get("status", "unknown")
            details = result.get("details", "-")
            readiness_date = result.get("readiness_date")
            readiness_date_str = (
                readiness_date.strftime("%Y-%m-%d")
                if isinstance(readiness_date, datetime)
                else "N/A"
            )

        agent_summary_data.append(
            {
                "agent": agent_name,
                "status": status,
                "readiness_date": readiness_date_str,
                "details": details,
            }
        )

    if agent_summary_data:
        print("\nAgent Readiness Details:")
        summary_df = pd.DataFrame(agent_summary_data)
        # Select and rename columns for better display
        display_df = summary_df[
            ["agent", "status", "readiness_date", "details"]
        ].rename(columns={"readiness_date": "Est. Ready Date"})
        print(display_df.to_string(index=False))

    # Display Remediation Plan if needed
    if overall_status == "delayed":
        remediation_plan = launch_status.get("remediation_plan", {})
        print("\nRemediation Plan:")
        print(
            f"- Latest Estimated Readiness: {remediation_plan.get('latest_estimated_readiness', 'N/A')}"
        )
        print(
            f"- Suggested New Launch Date: {remediation_plan.get('suggested_launch_date', 'N/A')}"
        )
        remediation_steps = remediation_plan.get("steps", [])
        if remediation_steps:
            print("\nRemediation Steps:")
            steps_df = pd.DataFrame(remediation_steps)
            print(steps_df.to_string(index=False))

    print("-----------------------------\n")

    logger.info("Product Launch Demo completed.")


if __name__ == "__main__":
    asyncio.run(demo_product_launch())
