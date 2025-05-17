"""Retrieves context data needed for LLM customer service agent based on intent."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Define dummy classes for Database/System interfaces if they aren't defined elsewhere
# In a real application, these would be proper implementations or interfaces.
class DummyProductDB:
    async def get_product(self, product_id: str) -> dict[str, Any] | None:
        return {"name": f"Product {product_id}", "price": 10.0} if product_id else None

    async def get_inventory(self, product_id: str) -> dict[str, Any] | None:
        return {"stock_level": 50} if product_id else None

    async def resolve_product_id(self, identifier: str) -> str | None:
        return identifier if identifier and "fail" not in identifier else None


class DummyOrderSystem:
    async def get_order_details(self, order_id: str) -> dict[str, Any] | None:
        return {"order_id": order_id, "status": "Shipped", "items": []} if order_id else None

    async def check_return_eligibility(self, order_id: str) -> dict[str, Any] | None:
        return {"eligible": True, "reason": None} if order_id else None

    async def get_recent_orders(self, customer_id: str, limit: int = 3) -> list[dict]:
        return []  # Needs real implementation or fixture data


class DummyCustomerDB:
    async def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        return {"name": f"Cust {customer_id}"} if customer_id else None


class LLMContextRetriever:
    """
    Retrieves context information from various backend systems
    based on the classified intent and extracted entities.
    """

    def __init__(
        self,
        product_database: Any,  # Should be Protocol/Interface for ProductDB
        order_management_system: Any,  # Should be Protocol/Interface for OrderSystem
        # customer_database: Any, # Not directly used in context fetching per intent
        policy_guidelines: dict[str, Any],
    ):
        """
        Initializes the context retriever with necessary system connectors.
        """
        self.product_db = product_database
        self.order_system = order_management_system
        self.policies = policy_guidelines
        logger.info("LLMContextRetriever initialized.")

    async def get_context(self, intent: str, entities: dict[str, Any]) -> dict[str, Any]:
        """
        Fetches and returns context data relevant to the intent and entities.

        Args:
            intent: The classified intent of the customer message.
            entities: A dictionary of extracted entities (e.g., {'order_id': '123', 'product_id': 'ABC'}).

        Returns:
            A dictionary containing the fetched context data.
        """
        context_data: dict[str, Any] = {}
        logger.debug(f"Getting context for intent: '{intent}', entities: {entities}")

        try:
            if intent == "order_status":
                order_id = entities.get("order_id")
                if order_id:
                    details = await self.order_system.get_order_details(order_id)
                    if details:
                        context_data["order_details"] = details
                        logger.debug(f"Retrieved order details for {order_id}.")
                    else:
                        logger.warning(f"Failed to retrieve details for order ID {order_id}.")
                else:
                    logger.warning("No order_id entity found for order_status intent.")

            elif intent == "product_question":
                product_identifier = entities.get("product_identifier")  # Name or SKU
                product_id = entities.get("product_id")  # Specific ID if resolved

                resolved_id = product_id  # Prefer specific ID if already resolved
                if not resolved_id and product_identifier:
                    logger.debug(f"Attempting to resolve product identifier: {product_identifier}")
                    resolved_id = await self.product_db.resolve_product_id(product_identifier)

                if resolved_id:
                    details = await self.product_db.get_product(resolved_id)
                    inventory = await self.product_db.get_inventory(resolved_id)
                    if details:
                        context_data["product_details"] = details
                        logger.debug(f"Retrieved product details for {resolved_id}.")
                    else:
                        logger.warning(f"Failed to retrieve details for resolved product ID {resolved_id}.")
                    if inventory:
                        context_data["inventory"] = inventory
                        logger.debug(f"Retrieved inventory for {resolved_id}.")
                    else:
                        logger.warning(f"Failed to retrieve inventory for resolved product ID {resolved_id}.")
                elif product_identifier:
                    logger.warning(f"Could not resolve product identifier '{product_identifier}' to an ID.")
                else:
                    logger.warning("No product_identifier or product_id entity found for product_question intent.")

            elif intent == "return_request":
                order_id = entities.get("order_id")
                if order_id:
                    details = await self.order_system.get_order_details(order_id)
                    eligibility = await self.order_system.check_return_eligibility(order_id)
                    policy = self.policies.get("returns", {})

                    if details:
                        context_data["order_details"] = details
                        logger.debug(f"Retrieved order details for return request: {order_id}.")
                    else:
                        logger.warning(f"Failed to retrieve order details for return request: {order_id}.")

                    if eligibility:
                        context_data["return_eligibility"] = eligibility
                        logger.debug(f"Retrieved return eligibility for {order_id}.")
                    else:
                        logger.warning(f"Failed to retrieve return eligibility for order ID {order_id}.")

                    context_data["return_policy"] = policy
                    logger.debug("Added return policy to context.")
                else:
                    logger.warning("No order_id entity found for return_request intent.")

            # Add other intent handlers here...
            # elif intent == "account_update": ...
            # elif intent == "general_inquiry": ... (maybe no specific context needed)

        except Exception as e:
            logger.error(
                f"Error retrieving context data for intent '{intent}' with entities {entities}: {e}",
                exc_info=True,
            )
            # Return partially gathered context or empty dict depending on desired error handling

        logger.debug(f"Returning context data: {list(context_data.keys())}")
        return context_data
