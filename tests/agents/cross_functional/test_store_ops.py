from datetime import datetime, timedelta

import pytest

from agents.cross_functional.store_ops import StoreOpsAgent


@pytest.mark.asyncio
async def test_prepare_for_launch():
    agent = StoreOpsAgent()
    product_id = "P321"
    launch_date = datetime.now() + timedelta(days=15)
    planogram_updates = {"aisle": "A1"}
    staff_training = ["product_overview", "sales_techniques"]
    display_requirements = ["standard_display"]
    result = await agent.prepare_for_launch(product_id, launch_date, planogram_updates, staff_training, display_requirements)
    assert result["status"] in ("ready", "delayed")
    assert "planogram_updated" in result
    assert "staff_trained" in result
    assert "displays_ready" in result
    assert "summary" in result


@pytest.mark.asyncio
async def test_suggest_remediation():
    agent = StoreOpsAgent()
    product_id = "P321"
    current_status = "delayed"
    result = await agent.suggest_remediation(product_id, current_status)
    assert result["action"] == "expedite_staff_training"
    assert "estimated_completion" in result
    assert "resource_needs" in result
    assert isinstance(result["resource_needs"], list)
