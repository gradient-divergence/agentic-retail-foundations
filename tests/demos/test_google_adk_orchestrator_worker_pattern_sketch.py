from demos.google_adk_orchestrator_worker_pattern_sketch import PriceUpdatePayload, Task, orchestrator


def test_price_update_payload_retains_its_type():
    task = Task(task_id="t2", intent="update_price", payload={"sku": "SKU456"})
    assert isinstance(task.payload, PriceUpdatePayload)
    assert isinstance(orchestrator([task])[0].result, PriceUpdatePayload)
