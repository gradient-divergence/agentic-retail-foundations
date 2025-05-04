import pytest
from datetime import datetime, timedelta
from agents.cross_functional.supply_chain import SupplyChainAgent


@pytest.mark.asyncio
async def test_plan_initial_distribution():
    agent = SupplyChainAgent()
    product_id = "P123"
    target_date = datetime.now() + timedelta(days=10)
    forecast_units = 100
    store_allocation = {"store1": 60, "store2": 40}
    result = await agent.plan_initial_distribution(
        product_id, target_date, forecast_units, store_allocation
    )
    assert result["status"] == "ready"
    assert result["allocation_by_store"] == store_allocation
    assert forecast_units == sum(store_allocation.values())
    assert "summary" in result
    assert isinstance(result["completion_date"], datetime)


@pytest.mark.asyncio
async def test_suggest_remediation():
    agent = SupplyChainAgent()
    product_id = "P123"
    current_status = "delayed"
    result = await agent.suggest_remediation(product_id, current_status)
    assert result["action"] == "accelerate_distribution"
    assert "estimated_completion" in result
    assert isinstance(result["estimated_completion"], datetime)
    assert "resource_needs" in result
    assert isinstance(result["resource_needs"], list)
