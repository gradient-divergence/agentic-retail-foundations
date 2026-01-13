"""
Print-friendly excerpt of the LLM-powered customer service flow.
"""

# region book:customer-service-agent-excerpt
from __future__ import annotations

from pydantic import BaseModel, JsonValue, RootModel

from agents.llm import RetailCustomerServiceAgent


class AgentAction(RootModel[dict[str, JsonValue]]):
    pass


class AgentReply(BaseModel):
    message: str
    intent: str
    actions: list[AgentAction]


class Inquiry(BaseModel):
    customer_id: str
    message: str


async def handle_inquiry(agent: RetailCustomerServiceAgent, inquiry: Inquiry) -> AgentReply:
    customer_info = await agent.customer_db.get_customer(inquiry.customer_id)
    recent_orders = await agent.order_system.get_recent_orders(inquiry.customer_id, limit=3)

    intent = await agent._classify_intent(inquiry.message)
    context_data: dict[str, JsonValue] = {}

    if intent == "order_status":
        order_id = await agent._extract_order_id(inquiry.message, recent_orders)
        if order_id:
            context_data["order_details"] = await agent.order_system.get_order_details(order_id)
    elif intent == "product_question":
        product_id = await agent._extract_product_id(inquiry.message)
        if product_id:
            context_data["product_details"] = await agent.product_db.get_product(product_id)
            context_data["inventory"] = await agent.product_db.get_inventory(product_id)

    response = await agent._generate_response(
        customer_info=customer_info,
        intent=intent,
        message=inquiry.message,
        context_data=context_data,
        conversation_history=[],
    )
    return AgentReply.model_validate(response)


# endregion book:customer-service-agent-excerpt
