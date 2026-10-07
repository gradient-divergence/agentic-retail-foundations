import asyncio
from unittest.mock import Mock

import pytest

from capstone.orchestrator import CapstoneOrchestrator
from capstone.schemas import ToolCall
from demos import capstone_gateway_demo


@pytest.mark.parametrize("quantity", [0, -1, True])
def test_reservation_demo_gateway_rejects_invalid_quantity(monkeypatch, quantity):
    calls = []
    monkeypatch.setattr(capstone_gateway_demo, "reserve_inventory", Mock(return_value={"status": "reserved"}))

    class CaptureGateway(CapstoneOrchestrator):
        def handle_tool_call(self, call, trace, actor="capstone-agent"):
            result = super().handle_tool_call(
                ToolCall(name=call.name, args={"sku": "sku", "quantity": quantity}), trace, actor
            )
            calls.append(result)
            return result

    monkeypatch.setattr(capstone_gateway_demo, "CapstoneOrchestrator", CaptureGateway)
    asyncio.run(capstone_gateway_demo.main())
    assert calls[0][0].status == "error"
    capstone_gateway_demo.reserve_inventory.assert_not_called()
