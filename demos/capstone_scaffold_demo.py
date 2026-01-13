from __future__ import annotations

from pydantic import BaseModel

from capstone.orchestrator import CapstoneOrchestrator
from capstone.policies import PolicyEvaluator
from capstone.schemas import ToolCall
from capstone.tools import ToolSpec
from capstone.tracing import TraceContext


class InventoryRisk(BaseModel):
    sku: str
    risk_score: float
    recommended_action: str


def lookup_inventory_risk(sku: str) -> InventoryRisk:
    return InventoryRisk(sku=sku, risk_score=0.42, recommended_action="replenish")


def main() -> None:
    trace = TraceContext(route="capstone/scaffold", tenant_id="retail-demo")
    tool = ToolSpec(
        name="inventory_risk_lookup",
        description="Return a risk score for the SKU.",
        permission="read",
        schema={
            "type": "object",
            "properties": {"sku": {"type": "string"}},
            "required": ["sku"],
        },
        handler=lookup_inventory_risk,
    )
    policy = PolicyEvaluator()
    orchestrator = CapstoneOrchestrator([tool], policy)
    call = ToolCall(name="inventory_risk_lookup", args={"sku": "SKU-123"})
    result, audit = orchestrator.handle_tool_call(call, trace)

    print("Tool result:", result.model_dump())
    print("Audit record:", audit.model_dump())
    print("Latency:", trace.finish())


if __name__ == "__main__":
    main()
