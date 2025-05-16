"""
Cross-functional agent classes and product launch orchestration logic for retail MAS.
"""

from datetime import datetime, timedelta
from typing import Any

from agents.cross_functional.customer_service import CustomerServiceAgent
from agents.cross_functional.marketing import MarketingAgent
from agents.cross_functional.pricing import PricingAgent
from agents.cross_functional.store_ops import StoreOpsAgent
from agents.cross_functional.supply_chain import SupplyChainAgent


def calculate_remediation_timeline(
    steps_by_domain: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate a timeline for remediation steps across domains.
    """
    all_durations = []
    for domain, step in steps_by_domain.items():
        duration = (step["estimated_completion"] - datetime.now()).days
        all_durations.append((domain, duration))
    all_durations.sort(key=lambda x: x[1], reverse=True)
    critical_path = [d[0] for d in all_durations[:2]]  # The two longest steps
    max_duration = all_durations[0][1] if all_durations else 0
    return {
        "critical_path": critical_path,
        "completion_date": datetime.now() + timedelta(days=max_duration),
        "suggested_launch_date": datetime.now() + timedelta(days=max_duration + 7),
    }


async def coordinate_product_launch(product_data: dict[str, Any]) -> dict[str, Any]:
    """
    Orchestrate a product launch across supply chain, pricing, marketing, store operations, and customer service.
    """
    agents = {
        "supply_chain": SupplyChainAgent(),
        "pricing": PricingAgent(),
        "marketing": MarketingAgent(),
        "store_ops": StoreOpsAgent(),
        "customer_service": CustomerServiceAgent(),
    }
    product_id = product_data["id"]
    launch_date = product_data["planned_launch_date"]
    print(f"\nCoordinating launch for product {product_id} planned for {launch_date.strftime('%Y-%m-%d')}\n")
    inventory_plan = await agents["supply_chain"].plan_initial_distribution(
        product_id=product_id,
        target_date=launch_date,
        forecast_units=product_data["first_month_forecast"],
        store_allocation=product_data["store_allocation_strategy"],
    )
    price_strategy = await agents["pricing"].develop_launch_pricing(
        product_id=product_id,
        cost=product_data["unit_cost"],
        competitor_data=product_data["competitor_products"],
        margin_targets=product_data["margin_targets"],
        price_elasticity=product_data["price_elasticity_estimate"],
    )
    campaign_plan = await agents["marketing"].create_launch_campaign(
        product_id=product_id,
        launch_date=launch_date,
        product_features=product_data["key_features"],
        target_segments=product_data["target_segments"],
        price_point=price_strategy["recommended_price"],
        messaging=product_data["messaging_guidelines"],
    )
    store_readiness = await agents["store_ops"].prepare_for_launch(
        product_id=product_id,
        launch_date=launch_date,
        planogram_updates=product_data["planogram"],
        staff_training=product_data["training_materials"],
        display_requirements=product_data["display_guidelines"],
    )
    cs_readiness = await agents["customer_service"].prepare_product_support(
        product_id=product_id,
        launch_date=launch_date,
        product_specs=product_data["specifications"],
        support_docs=product_data["support_materials"],
        faq=product_data["anticipated_questions"],
        return_policy=product_data["return_policy"],
    )
    statuses = {
        "supply_chain": inventory_plan["status"],
        "pricing": price_strategy["status"],
        "marketing": campaign_plan["status"],
        "store_ops": store_readiness["status"],
        "customer_service": cs_readiness["status"],
    }
    print("\nLaunch Readiness by Department:")
    for dept, status in statuses.items():
        print(f"- {dept.replace('_', ' ').title()}: {status.upper()}")
    all_ready = all(s == "ready" for s in statuses.values())
    if all_ready:
        print("\n✅ All departments ready! Product launch is CONFIRMED.")
        return {
            "product_id": product_id,
            "launch_status": "confirmed",
            "launch_date": launch_date,
            "inventory_plan": inventory_plan["summary"],
            "pricing_strategy": price_strategy["summary"],
            "marketing_plan": campaign_plan["summary"],
            "store_readiness": store_readiness["summary"],
            "support_readiness": cs_readiness["summary"],
        }
    else:
        blockers = [domain for domain, stat in statuses.items() if stat != "ready"]
        print(f"\n⚠️ Launch DELAYED due to {len(blockers)} departments not ready: {', '.join(blockers)}")
        print("\nGenerating remediation plan...")
        remediation_plan = {}
        for blocker in blockers:
            remediation_plan[blocker] = await agents[blocker].suggest_remediation(product_id=product_id, current_status=statuses[blocker])
        timeline = calculate_remediation_timeline(remediation_plan)
        print("\nRemediation Timeline:")
        print(f"- Critical path: {' → '.join(timeline['critical_path'])}")
        print(f"- Estimated completion: {timeline['completion_date'].strftime('%Y-%m-%d')}")
        print(f"- Suggested new launch: {timeline['suggested_launch_date'].strftime('%Y-%m-%d')}")
        return {
            "product_id": product_id,
            "launch_status": "delayed",
            "original_date": launch_date,
            "blockers": blockers,
            "remediation_plan": {
                "steps_by_domain": remediation_plan,
                "critical_path": timeline["critical_path"],
                "estimated_completion": timeline["completion_date"],
                "suggested_new_launch": timeline["suggested_launch_date"],
            },
        }
