import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

from agents.cv import ShelfMonitoringAgent
from demos.shelf_monitoring_book_demo import ShelfMonitoringAgent as BookShelfAgent


@pytest.mark.parametrize("agent_type", [ShelfMonitoringAgent, BookShelfAgent])
def test_recovered_shelf_clears_prior_issues(monkeypatch, agent_type):
    monkeypatch.setattr(torch.jit, "load", lambda path: SimpleNamespace(eval=lambda: None))
    db = SimpleNamespace(
        get_section_camera=AsyncMock(return_value="CAM01"),
        get_section_planogram=AsyncMock(return_value={"products": []}),
    )
    agent = agent_type("fake", db, SimpleNamespace(report_visual_audit=AsyncMock()), {})
    agent.detection_model = lambda tensor: {}
    agent._preprocess_image = lambda frame: None
    agent._process_detections = lambda *args: []
    agent.detected_issues["SEC001"] = [{"type": "OUT_OF_STOCK"}]
    stream = SimpleNamespace(isOpened=lambda: True, read=lambda: (True, np.zeros((2, 3, 3))))
    agent.active_streams["CAM01"] = stream
    if agent_type is BookShelfAgent:
        asyncio.run(agent._check_section("LOC1", "SEC001"))
    else:
        asyncio.run(agent._check_section("LOC1", "SEC001", "CAM01", stream))
    assert agent.detected_issues["SEC001"] == []


def test_failed_recovery_notification_is_retried(monkeypatch):
    monkeypatch.setattr(torch.jit, "load", lambda path: SimpleNamespace(eval=lambda: None))
    db = SimpleNamespace(
        get_section_camera=AsyncMock(return_value="CAM01"),
        get_section_planogram=AsyncMock(return_value={"products": []}),
    )
    inventory = SimpleNamespace(report_visual_audit=AsyncMock(side_effect=[RuntimeError("failed"), None]))
    agent = BookShelfAgent("fake", db, inventory, {})
    agent.detection_model = lambda tensor: {}
    agent._preprocess_image = lambda frame: None
    agent._process_detections = lambda *args: []
    agent.detected_issues["SEC001"] = [{"type": "OUT_OF_STOCK"}]
    agent.active_streams["CAM01"] = SimpleNamespace(
        isOpened=lambda: True, read=lambda: (True, np.zeros((2, 3, 3)))
    )
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(agent._check_section("LOC1", "SEC001"))
    asyncio.run(agent._check_section("LOC1", "SEC001"))
    assert inventory.report_visual_audit.await_count == 2
    assert agent.detected_issues["SEC001"] == []
