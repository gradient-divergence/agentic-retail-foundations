"""
Product Launch Coordinator agent.
Coordinates activities across different functional agents for a product launch.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

# Import dependent agent types (adjust paths if needed)
from agents.cross_functional import (
    CustomerServiceAgent,
    MarketingAgent,
    PricingAgent,
    StoreOpsAgent,
    SupplyChainAgent,
)


class ProductLaunchCoordinator:
    """
    Coordinates the product launch process across various functional agents.
    """

    def __init__(
        self,
        supply_chain_agent: SupplyChainAgent,
        pricing_agent: PricingAgent,
        marketing_agent: MarketingAgent,
        store_ops_agent: StoreOpsAgent,
        customer_service_agent: CustomerServiceAgent,
    ):
        self.supply_chain_agent = supply_chain_agent
        self.pricing_agent = pricing_agent
        self.marketing_agent = marketing_agent
        self.store_ops_agent = store_ops_agent
        self.customer_service_agent = customer_service_agent
        # print("ProductLaunchCoordinator initialized (Placeholder)") # Quieter init

    async def coordinate_product_launch(self, product_data: dict[str, Any]) -> dict[str, Any]:
        """
        Coordinates the product launch by checking readiness across functional agents.

        Args:
            product_data: Dictionary containing details about the product to launch.

        Returns:
            Dictionary summarizing the launch status, blockers, and dates.
        """
        product_id = product_data.get("id", "Unknown Product")
        planned_launch_date = product_data.get("planned_launch_date", datetime.now() + timedelta(days=30))
        print(f"Coordinating launch for product {product_id}...")

        # --- Call Agent Readiness Checks ---
        # Store agents along with their check coroutines
        agent_check_map = {
            self.supply_chain_agent: self.supply_chain_agent.check_readiness(product_data),
            self.pricing_agent: self.pricing_agent.check_readiness(product_data),
            self.marketing_agent: self.marketing_agent.check_readiness(product_data),
            self.store_ops_agent: self.store_ops_agent.check_readiness(product_data),
            self.customer_service_agent: self.customer_service_agent.check_readiness(product_data),
        }
        agent_list = list(agent_check_map.keys())
        check_coroutines = list(agent_check_map.values())

        agent_statuses = await asyncio.gather(*check_coroutines, return_exceptions=True)

        # --- Process Results ---
        blockers: list[str] = []
        all_ready = True
        latest_readiness_date = planned_launch_date  # Start with planned date
        failed_agents: list[str] = []

        for i, result in enumerate(agent_statuses):
            agent_instance = agent_list[i]  # Get the agent instance
            agent_name = agent_instance.__class__.__name__

            if isinstance(result, BaseException):
                all_ready = False
                error_msg = f"{agent_name}: Failed readiness check - {type(result).__name__}: {result}"
                blockers.append(error_msg)
                failed_agents.append(agent_name)
                print(f"  - ERROR: {error_msg}")
                # Assume a significant delay if an agent check fails entirely
                failure_readiness_date = planned_launch_date + timedelta(days=14)
                if failure_readiness_date > latest_readiness_date:
                    latest_readiness_date = failure_readiness_date
                continue  # Skip processing this result further

            # Safely access keys now we know it's not an exception
            status = result.get("status", "unknown")
            details = result.get("details", "No details provided.")
            readiness_date = result.get("readiness_date")

            if status != "ready":
                all_ready = False
                blockers.append(f"{agent_name}: {details}")
                if isinstance(readiness_date, datetime) and readiness_date > latest_readiness_date:
                    latest_readiness_date = readiness_date
                elif readiness_date is None:  # If blocker has no date, assume worst case
                    worst_case_date = planned_launch_date + timedelta(days=21)  # Arbitrary long delay
                    if worst_case_date > latest_readiness_date:
                        latest_readiness_date = worst_case_date
            elif isinstance(readiness_date, datetime) and readiness_date > planned_launch_date:
                # Even if ready, if their readiness date is after planned launch, it's a blocker
                all_ready = False
                blockers.append(f"{agent_name}: Ready but after planned launch ({readiness_date.strftime('%Y-%m-%d')})")
                if readiness_date > latest_readiness_date:
                    latest_readiness_date = readiness_date

        # --- Determine Final Status and Remediation (Improved) ---
        final_status: dict[str, Any] = {
            "product_id": product_id,
            "original_launch_date": planned_launch_date.strftime("%Y-%m-%d"),
            "blockers": blockers,
            "agent_statuses": agent_statuses,  # Include raw results for details
        }

        if all_ready:
            final_status["launch_status"] = "confirmed"
            final_status["final_launch_date"] = planned_launch_date.strftime("%Y-%m-%d")
            print(f"--> Launch Confirmed for {final_status['final_launch_date']}")
        else:
            final_status["launch_status"] = "delayed"
            # Suggest new launch date based on latest readiness + buffer
            suggested_launch_date = latest_readiness_date + timedelta(days=7)
            final_status["suggested_new_launch_date"] = suggested_launch_date.strftime("%Y-%m-%d")
            # Basic remediation plan based on blockers
            remediation_steps = []
            for result in agent_statuses:
                if not isinstance(result, BaseException) and result.get("status") != "ready":
                    remediation_steps.append(
                        {
                            "agent": result.get("agent"),
                            "blocker": result.get("details"),
                            "estimated_ready": (
                                result.get("readiness_date").strftime("%Y-%m-%d")  # type: ignore[union-attr]
                                if isinstance(result.get("readiness_date"), datetime)
                                else "Unknown"
                            ),
                        }
                    )
                elif isinstance(result, BaseException):
                    # Find which agent failed using the original list order
                    failed_agent_instance = agent_list[agent_statuses.index(result)]
                    failed_agent_name = failed_agent_instance.__class__.__name__
                    remediation_steps.append(
                        {
                            "agent": failed_agent_name,
                            "blocker": f"Agent communication failed: {type(result).__name__}",
                            "estimated_ready": "Unknown",
                        }
                    )

            final_status["remediation_plan"] = {
                "latest_estimated_readiness": latest_readiness_date.strftime("%Y-%m-%d"),
                "suggested_launch_date": suggested_launch_date.strftime("%Y-%m-%d"),
                "steps": remediation_steps,
            }
            print(f"--> Launch Delayed. Blockers found. Suggested new date: {final_status['suggested_new_launch_date']}")

        return final_status
