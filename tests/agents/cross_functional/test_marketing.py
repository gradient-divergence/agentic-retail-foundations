from datetime import datetime, timedelta

import pytest

from agents.cross_functional.marketing import MarketingAgent


@pytest.mark.asyncio
async def test_create_launch_campaign():
    agent = MarketingAgent()
    product_id = "P789"
    launch_date = datetime.now() + timedelta(days=30)
    product_features = ["feature1", "feature2", "feature3"]
    target_segments = ["millennials", "professionals"]
    price_point = 19.99
    messaging = {"main": "Exciting new product!"}
    result = await agent.create_launch_campaign(
        product_id,
        launch_date,
        product_features,
        target_segments,
        price_point,
        messaging,
    )
    assert result["status"] in ("ready", "delayed")
    assert "channels" in result
    assert isinstance(result["channels"], list)
    assert "summary" in result
    assert "start_date" in result
    assert "key_messages" in result
    assert isinstance(result["key_messages"], list)


@pytest.mark.asyncio
async def test_suggest_remediation():
    agent = MarketingAgent()
    product_id = "P789"
    current_status = "delayed"
    result = await agent.suggest_remediation(product_id, current_status)
    assert result["action"] == "expand_campaign_reach"
    assert "estimated_completion" in result
    assert "resource_needs" in result
    assert isinstance(result["resource_needs"], list)
