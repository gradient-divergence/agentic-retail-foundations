from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.cv import ShelfMonitoringAgent
from demos.shelf_monitoring_book_demo import ShelfMonitoringAgent as BookShelfAgent


@pytest.mark.parametrize("agent_type", [ShelfMonitoringAgent, BookShelfAgent])
def test_recorded_detections_preserve_yxyx_coordinates(monkeypatch, agent_type):
    monkeypatch.setattr(torch.jit, "load", lambda path: SimpleNamespace(eval=lambda: None))
    agent = agent_type("fake", None, None, {})
    detections = {
        "detection_boxes": torch.tensor([[[0.1, 0.2, 0.6, 0.8]]], dtype=torch.float64),
        "detection_classes": torch.tensor([[1]]),
        "detection_scores": torch.tensor([[0.9]]),
    }
    product = agent._process_detections(detections, 200, 100)[0]
    if hasattr(product, "model_dump"):
        product = product.model_dump()
    assert product["bounding_box"] == [10, 40, 60, 160]
    assert product["shelf_position"] == {"x": 0.5, "y": 0.35}


@pytest.mark.parametrize("agent_type", [ShelfMonitoringAgent, BookShelfAgent])
def test_preprocess_uses_rgb_nhwc(monkeypatch, agent_type):
    monkeypatch.setattr(torch.jit, "load", lambda path: SimpleNamespace(eval=lambda: None))
    agent = agent_type("fake", None, None, {})
    bgr = np.full((2, 3, 3), [0, 128, 255], dtype=np.uint8)
    tensor = agent._preprocess_image(bgr)
    assert tensor.shape == (1, 640, 640, 3)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor[0, 0, 0], torch.tensor([1.0, 128 / 255, 0.0]))


def test_missing_model_does_not_produce_false_stock_audits(monkeypatch):
    def fail_load(path):
        raise OSError("model not found")

    monkeypatch.setattr(torch.jit, "load", fail_load)
    with pytest.raises(OSError, match="model not found"):
        ShelfMonitoringAgent("missing", None, None, {})
