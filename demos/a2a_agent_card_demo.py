from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from a2a.types import AgentCard


class InventoryLookupRequest(BaseModel):
    sku: str


class InventoryLookupResponse(BaseModel):
    sku: str
    in_stock: bool
    available_units: int


def build_agent_card() -> AgentCard:
    try:
        from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'a2a-sdk is not installed. Install with: uv pip install -e ".[agent_protocols]"'
        ) from exc

    skill = AgentSkill(
        id="inventory_lookup",
        name="Inventory availability",
        description="Returns availability for a given SKU.",
        tags=["inventory", "availability"],
        examples=["Is SKU-123 in stock?"],
    )
    return AgentCard(
        name="Retail Inventory Agent",
        description="Answers SKU availability questions for retail operations.",
        supported_interfaces=[
            AgentInterface(url="http://localhost:9999/", protocol_binding="JSONRPC", protocol_version="1.0")
        ],
        version="1.0.0",
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=True, extended_agent_card=True),
        skills=[skill],
    )


def main() -> None:
    from google.protobuf.json_format import MessageToJson

    card = build_agent_card()
    print(MessageToJson(card, indent=2))


if __name__ == "__main__":
    main()
