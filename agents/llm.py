"""
Module: agents.llm

Contains the RetailCustomerServiceAgent class for LLM-powered customer service in retail.
"""

from typing import Any, Dict, List, Set, Optional
from datetime import datetime
import logging
import asyncio
import os
import re
from collections import defaultdict, deque

from openai import AsyncOpenAI, OpenAI
from utils.openai_utils import safe_chat_completion

# NLP helpers
from utils.nlp import (
    classify_intent as nlp_classify_intent,
    extract_product_identifier as nlp_extract_product_identifier,
    analyze_sentiment as nlp_analyze_sentiment,
    extract_order_id_via_llm as nlp_extract_order_id_llm,
)

# Utils
from agents.response_builder import build_response_prompt, extract_actions


class RetailCustomerServiceAgent:
    """
    LLM-powered customer service agent for retail.
    Handles customer inquiries, order status, product questions, and returns using an LLM backend.
    Robustness additions:
    - API key now read from `OPENAI_API_KEY` environment variable if not passed.
    - Bounded per‑customer conversation history (default 50 turns) stored in `deque`.
    - Simple exponential back‑off retry wrapper for all OpenAI chat completions.
    - Per‑customer `asyncio.Lock` to avoid race conditions when multiple async
      requests arrive for the same customer concurrently.
    - Response and utility model names are now configurable (default: gpt‑4o and gpt‑4o‑mini).
    """

    client: Optional[AsyncOpenAI] = None

    def __init__(
        self,
        product_database,
        order_management_system,
        customer_database,
        policy_guidelines,
        api_key: str | None = None,
        *,
        max_history_per_user: int = 50,
        retry_attempts: int = 3,
        retry_backoff: float = 1.0,
        response_model: str = "gpt-4o",
        utility_model: str = "gpt-4o-mini",
    ):
        """Initializes the RetailCustomerServiceAgent."""
        self.product_db = product_database
        self.order_system = order_management_system
        self.customer_db = customer_database
        self.policies = policy_guidelines
        self.logger = logging.getLogger(__name__)
        self.max_history_per_user = max_history_per_user
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        self.response_model = response_model
        self.utility_model = utility_model
        self.conversation_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_history_per_user)
        )
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if resolved_key and resolved_key != "YOUR_API_KEY_HERE":
            try:
                self.client = AsyncOpenAI(api_key=resolved_key)
                self.logger.info("AsyncOpenAI client initialized successfully.")
            except Exception as e:
                self.logger.error("Failed to initialize OpenAI client: %s", e)
        else:
            self.logger.warning(
                "OpenAI API key missing or placeholder. LLM features will be disabled."
            )
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def process_customer_inquiry(
        self, customer_id: str, message: str
    ) -> dict[str, Any]:
        """Process a customer inquiry and generate an appropriate response."""
        if not self.client:
            self.logger.error("OpenAI client not initialized. Cannot process inquiry.")
            return {
                "message": "I apologize, our AI assistance system is currently unavailable. Please contact support directly.",
                "intent": "error",
                "actions": [],
                "error": "LLM client not available",
            }
        self.logger.info(
            f"Processing inquiry for customer {customer_id}: '{message[:50]}...'"
        )
        customer_info = None
        recent_orders = []
        try:
            customer_info = await self.customer_db.get_customer(customer_id)
            recent_orders_raw = await self.order_system.get_recent_orders(
                customer_id, limit=3
            )
            recent_orders = (
                recent_orders_raw if isinstance(recent_orders_raw, list) else []
            )
            self.logger.debug(
                f"Retrieved context for customer {customer_id}. Recent orders: {[o.get('order_id', 'N/A') for o in recent_orders]}"
            )
        except Exception as e:
            self.logger.error(
                f"Error retrieving customer context for {customer_id}: {e}"
            )
            if customer_info is None:
                customer_info = {
                    "name": f"Customer {customer_id}",
                    "loyalty_tier": "Standard",
                }
        if customer_id not in self.conversation_history:
            # defaultdict handles initialization
            self.logger.debug(
                f"Initialized conversation history for customer {customer_id}."
            )
        async with self._locks[customer_id]:
            self.conversation_history[customer_id].append(
                {
                    "role": "customer",
                    "content": message,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        intent = await self._classify_intent(message)
        self.logger.debug(f"Classified intent for message as: {intent}")
        context_data = {}
        try:
            if intent == "order_status":
                order_id = await self._extract_order_id(message, recent_orders)
                if order_id:
                    self.logger.debug(
                        f"Extracted order ID: {order_id} for order_status intent."
                    )
                    context_data["order_details"] = (
                        await self.order_system.get_order_details(order_id)
                    )
                    if context_data["order_details"]:
                        self.logger.debug(f"Retrieved order details for {order_id}.")
                    else:
                        self.logger.warning(
                            f"Failed to retrieve details for extracted order ID {order_id}."
                        )
                        context_data.pop("order_details", None)
                else:
                    self.logger.warning(
                        f"Could not extract valid order ID for order_status intent from message: '{message}'"
                    )
            elif intent == "product_question":
                product_identifier = await self._extract_product_identifier(message)
                if product_identifier:
                    self.logger.debug(
                        f"Extracted product identifier: {product_identifier}"
                    )
                    product_id = await self.product_db.resolve_product_id(
                        product_identifier
                    )
                    if product_id:
                        context_data["product_details"] = (
                            await self.product_db.get_product(product_id)
                        )
                        context_data["inventory"] = await self.product_db.get_inventory(
                            product_id
                        )
                        if context_data["product_details"]:
                            self.logger.debug(
                                f"Retrieved product details and inventory for {product_id}."
                            )
                        else:
                            self.logger.warning(
                                f"Failed to retrieve details for resolved product ID {product_id}."
                            )
                            context_data.pop("product_details", None)
                            context_data.pop("inventory", None)
                    else:
                        self.logger.warning(
                            f"Could not resolve product identifier '{product_identifier}' to a product ID."
                        )
                else:
                    self.logger.warning(
                        "Could not extract product identifier for product_question intent."
                    )
            elif intent == "return_request":
                order_id = await self._extract_order_id(message, recent_orders)
                if order_id:
                    self.logger.debug(
                        f"Extracted order ID: {order_id} for return_request intent."
                    )
                    context_data["order_details"] = (
                        await self.order_system.get_order_details(order_id)
                    )
                    context_data["return_eligibility"] = (
                        await self.order_system.check_return_eligibility(order_id)
                    )
                    context_data["return_policy"] = self.policies.get("returns", {})
                    if (
                        context_data["order_details"]
                        and context_data["return_eligibility"]
                    ):
                        self.logger.debug(
                            f"Retrieved order details and return eligibility for {order_id}."
                        )
                    else:
                        self.logger.warning(
                            f"Failed to retrieve full context for return request for order ID {order_id}."
                        )
                else:
                    self.logger.warning(
                        f"Could not extract valid order ID for return_request intent from message: '{message}'"
                    )
        except Exception as e:
            self.logger.error(
                f"Error retrieving context data for intent '{intent}': {e}",
                exc_info=True,
            )
        recent_history = list(self.conversation_history[customer_id])[-5:]
        response = await self._generate_response(
            customer_info=customer_info or {},
            intent=intent,
            message=message,
            context_data=context_data,
            conversation_history=recent_history,
        )
        self.logger.info(
            f"Generated response for customer {customer_id}. Intent: {intent}. Response: '{response.get('message', '')[:50]}...'"
        )
        if "message" in response and "error" not in response:
            async with self._locks[customer_id]:
                self.conversation_history[customer_id].append(
                    {
                        "role": "agent",
                        "content": response["message"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )
        try:
            await self._log_interaction(customer_id, intent, message, response)
            self.logger.debug(f"Logged interaction for customer {customer_id}.")
        except Exception as e:
            self.logger.error(f"Failed to log interaction: {e}")
        return response

    async def _safe_response_create(self, **kwargs):
        """Thin wrapper delegating to :pyfunc:`utils.openai_utils.safe_chat_completion`."""
        return await safe_chat_completion(
            self.client,
            logger=self.logger,
            retry_attempts=self.retry_attempts,
            retry_backoff=self.retry_backoff,
            **kwargs,
        )

    async def _classify_intent(self, message: str) -> str:
        """Use LLM to classify the customer's intent."""
        if not self.client:
            return "general_inquiry"
        try:
            return await nlp_classify_intent(
                self.client,
                message=message,
                model=self.utility_model,
                logger=self.logger,
                retry_attempts=self.retry_attempts,
                retry_backoff=self.retry_backoff,
            )
        except Exception as e:
            self.logger.error(f"LLM intent classification failed: {e}")
            return "general_inquiry"

    async def _extract_order_id(
        self, message: str, recent_orders: list[dict]
    ) -> str | None:
        """Extract order ID from message or infer from recent orders."""
        match = re.search(
            r"(?:#|order\s*|order number\s*)?([a-z0-9]{6,})", message, re.IGNORECASE
        )
        plausible_id = None
        if match:
            potential_id = match.group(1)
            if not re.fullmatch(
                r"(status|product|item|sku|mat|shoes|bottle)",
                potential_id,
                re.IGNORECASE,
            ):
                plausible_id = potential_id.upper()
                self.logger.debug(
                    f"Extracted potential order ID via regex: {plausible_id}"
                )
                for order in recent_orders:
                    order_id_val = order.get("order_id")
                    if order_id_val and order_id_val.upper() == plausible_id:
                        self.logger.debug(
                            f"Regex extracted ID {plausible_id} matches recent order."
                        )
                        return order_id_val
        if not self.client:
            return plausible_id
        if not recent_orders:
            self.logger.debug("No recent orders available for LLM inference.")
            return plausible_id
        recent_order_ids = [order.get("order_id", "N/A") for order in recent_orders]
        try:
            result = await nlp_extract_order_id_llm(
                self.client,
                message=message,
                recent_order_ids=recent_order_ids,
                model=self.utility_model,
                logger=self.logger,
                retry_attempts=self.retry_attempts,
                retry_backoff=self.retry_backoff,
            )
            self.logger.debug(f"LLM order ID extraction result: '{result}'")
            # If the LLM cannot uniquely determine an order (or returns a helper keyword),
            # do **not** propagate that string as a real order ID.  Fallback to the regex
            # extracted `plausible_id` which may be ``None``.
            if result.lower() in {"ambiguous", "not_found", "not_recent"}:
                return plausible_id
            return result
        except Exception as e:
            self.logger.error(f"LLM order ID extraction failed via helper: {e}")
            return plausible_id

    async def _extract_product_identifier(self, message: str) -> str | None:
        """Extract potential product ID or name from customer message."""
        if not self.client:
            return None
        try:
            result = await nlp_extract_product_identifier(
                self.client,
                message=message,
                model=self.utility_model,
                logger=self.logger,
                retry_attempts=self.retry_attempts,
                retry_backoff=self.retry_backoff,
            )
            self.logger.debug(f"LLM product identifier extraction result: '{result}'")
            return result # type: ignore[no-any-return]
        except Exception as e:
            self.logger.error(f"LLM product identifier extraction failed: {e}")
            return None # type: ignore[no-any-return]

    async def _generate_response(
        self,
        customer_info: dict,
        intent: str,
        message: str,
        context_data: dict,
        conversation_history: list,
    ) -> dict[str, Any]:
        """Generate a response using the LLM based on intent and context."""
        if not self.client:
            return {
                "message": "I apologize, our AI assistance is currently unavailable. Can I help with anything else?",
                "intent": intent,
                "actions": [],
                "error": "LLM client not available",
            }

        final_system_prompt = build_response_prompt(
            customer_info=customer_info,
            intent=intent,
            message=message,
            context_data=context_data,
            conversation_history=conversation_history,
        )
        try:
            # Revert to using messages for chat completions
            completion = await safe_chat_completion(
                self.client,
                model=self.response_model,
                messages=[{"role": "system", "content": final_system_prompt}], # Use messages
                logger=self.logger,
                retry_attempts=self.retry_attempts,
                retry_backoff=self.retry_backoff,
                max_tokens=250,
                temperature=0.7,
                stop=None,
            )
            # Use choices[0].message.content
            generated_message = completion.choices[0].message.content.strip() if completion.choices[0].message.content else ""
            self.logger.debug(f"LLM generated response: '{generated_message[:100]}...'")
            
            actions = await extract_actions(
                self.client,
                intent=intent,
                response_text=generated_message,
                context_data=context_data,
                model=self.utility_model,
                logger=self.logger,
                retry_attempts=self.retry_attempts,
                retry_backoff=self.retry_backoff,
            )
            self.logger.debug(f"Extracted actions: {actions}")
            customer_sentiment = await self._analyze_sentiment(message)
            return {
                "message": generated_message,
                "intent": intent,
                "actions": actions,
                "customer_sentiment": customer_sentiment,
            }
        except Exception as e:
            self.logger.error(f"LLM response generation failed: {e}", exc_info=True)
            return {
                "message": "I apologize, but I'm having technical difficulties and can't generate a full response right now. Could you please rephrase your request, or contact our support team directly?",
                "intent": intent,
                "actions": [],
                "error": str(e),
            }

    async def _analyze_sentiment(self, message: str) -> str:
        """Analyze message sentiment."""
        if not self.client:
            return "neutral"
        try:
            sentiment = await nlp_analyze_sentiment(
                self.client,
                message=message,
                model=self.utility_model,
                logger=self.logger,
                retry_attempts=self.retry_attempts,
                retry_backoff=self.retry_backoff,
            )
            self.logger.debug(f"Analyzed sentiment as: {sentiment}")
            return sentiment
        except Exception as e:
            self.logger.error(f"LLM sentiment analysis failed: {e}")
            return "neutral"

    async def _log_interaction(
        self, customer_id: str, intent: str, message: str, response: dict
    ):
        """Log the interaction details."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "customer_id": customer_id,
            "intent": intent,
            "customer_message_len": len(message),
            "agent_response_len": len(response.get("message", "")),
            "actions_taken": response.get("actions", []),
            "error": response.get("error"),
            "customer_sentiment": response.get("customer_sentiment"),
        }
        self.logger.info(
            f"Interaction logged: C:{customer_id}, Intent:{intent}, Actions:{len(log_entry['actions_taken'])}, Error:{log_entry['error'] is not None}"
        )
        # Extend this method to persist logs to a database or analytics system as needed.
        pass
