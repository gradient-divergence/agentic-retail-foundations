import pytest
from agents.cross_functional.pricing import PricingAgent


@pytest.mark.asyncio
async def test_develop_launch_pricing():
    agent = PricingAgent()
    product_id = "P456"
    cost = 10.0
    competitor_data = {"comp1": {"price": 12.0}, "comp2": {"price": 11.0}}
    margin_targets = {"target_margin": 0.2}
    price_elasticity = 1.5
    result = await agent.develop_launch_pricing(
        product_id, cost, competitor_data, margin_targets, price_elasticity
    )
    assert result["status"] == "ready"
    assert "recommended_price" in result
    assert result["recommended_price"] > 0
    assert "summary" in result
    assert "margin" in result
    assert 0 < result["margin"] < 1
    assert "competitor_analysis" in result


@pytest.mark.asyncio
async def test_suggest_remediation():
    agent = PricingAgent()
    product_id = "P456"
    current_status = "delayed"
    result = await agent.suggest_remediation(product_id, current_status)
    assert result["action"] == "revise_pricing_strategy"
    assert "estimated_completion" in result
    assert "resource_needs" in result
    assert isinstance(result["resource_needs"], list)
