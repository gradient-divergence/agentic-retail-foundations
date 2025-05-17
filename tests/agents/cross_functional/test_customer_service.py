from datetime import datetime, timedelta

import pytest

from agents.cross_functional.customer_service import CustomerServiceAgent


@pytest.mark.asyncio
async def test_prepare_product_support():
    agent = CustomerServiceAgent()
    product_id = "P654"
    launch_date = datetime.now() + timedelta(days=20)
    product_specs = {"color": "red", "size": "M"}
    support_docs = ["user_manual", "quick_start"]
    faq = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    return_policy = {"extended_period": True}
    result = await agent.prepare_product_support(product_id, launch_date, product_specs, support_docs, faq, return_policy)
    assert result["status"] in ("ready", "delayed")
    assert "knowledge_base_updated" in result
    assert "support_team_briefed" in result
    assert "return_policy_configured" in result
    assert "summary" in result


@pytest.mark.asyncio
async def test_suggest_remediation():
    agent = CustomerServiceAgent()
    product_id = "P654"
    current_status = "delayed"
    result = await agent.suggest_remediation(product_id, current_status)
    assert result["action"] == "develop_additional_support_materials"
    assert "estimated_completion" in result
    assert "resource_needs" in result
    assert isinstance(result["resource_needs"], list)
