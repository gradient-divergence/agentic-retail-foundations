"""
SupplyChainAgent for planning initial distribution and supply chain remediation in retail MAS.
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Any


class SupplyChainAgent:
    """
    Agent responsible for planning initial distribution and suggesting supply chain remediation.
    """

    def __init__(self):
        print("SupplyChainAgent initialized")

    async def plan_inventory(self, product_data: dict[str, Any]):
        """Placeholder: Plan initial inventory based on forecast."""
        forecast = product_data.get("first_month_forecast", 0)
        lead_time = product_data.get("lead_time_days", 30)
        print(f"SupplyChain: Planning inventory for {forecast} units (lead time: {lead_time} days)...")
        await asyncio.sleep(0.3)  # Simulate planning time
        print("SupplyChain: Initial inventory plan complete.")
        return {"status": "inventory_planned", "initial_order_placed": True}

    async def check_readiness(self, product_data: dict[str, Any]) -> dict[str, Any]:
        """Simulate checking if supply chain is ready for launch based on lead time."""
        agent_name = self.__class__.__name__
        product_id = product_data.get("id", "Unknown Product")
        planned_launch_date = product_data.get("planned_launch_date")
        lead_time = product_data.get("lead_time_days", 30)  # Default lead time if not specified
        print(f"SupplyChain: Checking readiness for {product_id}...")
        await asyncio.sleep(random.uniform(0.1, 0.3))

        status = "blocked"
        details = ""
        readiness_date = None

        if not isinstance(planned_launch_date, datetime):
            details = "Planned launch date missing or invalid."
            readiness_date = None  # Cannot determine readiness
        elif lead_time is None or not isinstance(lead_time, int) or lead_time < 0:
            details = "Product lead time missing or invalid."
            readiness_date = None  # Cannot determine readiness
        else:
            required_order_date = planned_launch_date - timedelta(days=lead_time)
            # Simulate that orders are placed roughly on time, but sometimes delays occur
            simulated_order_date = required_order_date - timedelta(days=random.randint(-3, 5))  # Order placed up to 3 days early or 5 days late
            estimated_arrival_date = simulated_order_date + timedelta(days=lead_time)

            if estimated_arrival_date <= planned_launch_date:
                status = "ready"
                details = f"Initial inventory order confirmed. Estimated arrival: {estimated_arrival_date.strftime('%Y-%m-%d')}."
                readiness_date = estimated_arrival_date
            else:
                status = "blocked"
                details = f"Potential supply delay. Estimated arrival {estimated_arrival_date.strftime('%Y-%m-%d')} is after launch date."
                readiness_date = estimated_arrival_date

        print(
            f"  - {agent_name}: {status} ({details}) - Est. Ready Date: {readiness_date.strftime('%Y-%m-%d') if isinstance(readiness_date, datetime) else 'N/A'}"
        )
        return {
            "agent": agent_name,
            "status": status,
            "details": details,
            "readiness_date": readiness_date,
        }

    async def plan_initial_distribution(
        self,
        product_id: str,
        target_date: datetime,
        forecast_units: int,
        store_allocation: dict[str, int],
    ) -> dict[str, Any]:
        """
        Plan initial distribution of product to stores.
        """
        print(f"Supply Chain: Planning distribution for {product_id}, {forecast_units} units")
        await asyncio.sleep(0.5)
        return {
            "status": "ready",
            "summary": f"Distribution plan ready for {forecast_units} units across {len(store_allocation)} stores",
            "allocation_by_store": store_allocation,
            "completion_date": datetime.now() + timedelta(days=3),
        }

    async def suggest_remediation(self, product_id: str, current_status: str) -> dict[str, Any]:
        """
        Suggest remediation steps for supply chain issues.
        """
        return {
            "action": "accelerate_distribution",
            "estimated_completion": datetime.now() + timedelta(days=5),
            "resource_needs": ["additional_warehouse_staff", "priority_shipping"],
        }
