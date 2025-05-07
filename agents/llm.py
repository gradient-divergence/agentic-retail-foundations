"""
Module: agents.llm

Contains the RetailCustomerServiceAgent class for LLM-powered customer service in retail.
Orchestrates calls to specific components for context retrieval, LLM subtasks, and conversation management.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging
import asyncio
import os

from openai import AsyncOpenAI
from utils.openai_utils import safe_chat_completion

# NLP utilities
from utils import nlp

# Agent components
from .conversation_manager import ConversationManager
from .llm_context_retriever import LLMContextRetriever
# Import from the new components module (replace response_builder)
from .llm_components import (
    build_response_prompt,
    extract_actions,
    generate_agent_response
)

# Dummy Interfaces (replace with actual imports or protocols if available)
from .llm_context_retriever import (
    DummyProductDB, DummyOrderSystem, DummyCustomerDB
)

class RetailCustomerServiceAgent:
    """
    LLM-powered customer service agent orchestrator.
    Handles customer inquiries by coordinating context retrieval, LLM tasks,
    and conversation history management.
    """

    def __init__(
        self,
        # Dependencies for context/data systems
        product_database: Any, # e.g., DummyProductDB()
        order_management_system: Any, # e.g., DummyOrderSystem()
        customer_database: Any, # e.g., DummyCustomerDB()
        policy_guidelines: Dict[str, Any],
        # LLM Configuration
        api_key: str | None = None,
        response_model: str = "gpt-4o",
        utility_model: str = "gpt-4o-mini",
        # Agent Behavior Configuration
        max_history_per_user: int = 50,
        retry_attempts: int = 3,
        retry_backoff: float = 1.0,
    ):
        """Initializes the RetailCustomerServiceAgent orchestrator."""
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.response_model = response_model
        self.utility_model = utility_model
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff

        # Initialize OpenAI client (required for sub-components)
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if resolved_key and resolved_key != "YOUR_API_KEY_HERE":
            try:
                self.client = AsyncOpenAI(api_key=resolved_key)
                self.logger.info("AsyncOpenAI client initialized successfully.")
            except Exception as e:
                self.client = None
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            self.client = None
            self.logger.warning(
                "OpenAI API key missing or placeholder. LLM features will be disabled."
            )

        # Initialize dependency components
        self.customer_db = customer_database
        self.conversation_manager = ConversationManager(max_history_per_user)
        self.context_retriever = LLMContextRetriever(
            product_database=product_database,
            order_management_system=order_management_system,
            # customer_database=customer_database, # Not needed by retriever directly
            policy_guidelines=policy_guidelines
        )

        self.logger.info("RetailCustomerServiceAgent initialized.")

    async def process_customer_inquiry(
        self, customer_id: str, message: str
    ) -> Dict[str, Any]:
        """
        Process a customer inquiry by orchestrating sub-components.

        Args:
            customer_id: The unique ID of the customer.
            message: The customer's message.

        Returns:
            A dictionary containing the agent's response, intent, actions, etc.
        """
        if not self.client:
            self.logger.error("OpenAI client not initialized. Cannot process inquiry.")
            # Return standard error response
            return {
                "message": "I apologize, our AI assistance system is currently unavailable. Please contact support directly.",
                "intent": "error",
                "actions": [],
                "error": "LLM client not available",
            }

        self.logger.info(
            f"Processing inquiry for customer {customer_id}: '{message[:50]}...'"
        )

        # Use context manager for lock
        async with self.conversation_manager.get_lock(customer_id):
            # 1. Add user message to history
            self.conversation_manager.add_message(customer_id, "customer", message)

            # 2. Basic context retrieval (customer info)
            customer_info: Dict[str, Any] = {}
            try:
                cust_info = await self.customer_db.get_customer(customer_id)
                customer_info = cust_info if cust_info else {}
            except Exception as e:
                self.logger.error(f"Error retrieving customer info for {customer_id}: {e}")
                # Proceed with default/empty info

            # 3. Classify Intent
            intent = await nlp.classify_intent(client=self.client, message=message)
            self.logger.debug(f"Classified intent: {intent}")

            # 4. Extract Entities (can be combined or specific)
            entities: Dict[str, Any] = {}
            try:
                # Example: Extract order ID (needs recent orders)
                # Note: Getting recent orders here might be slightly inefficient if not always needed
                # Could be moved into context retriever if desired.
                recent_orders = await self.context_retriever.order_system.get_recent_orders(customer_id, limit=3)
                extracted_order_id = await nlp.extract_order_id_llm(
                    client=self.client,
                    message=message,
                    recent_order_ids=[o["order_id"] for o in recent_orders if "order_id" in o],
                    model=self.utility_model,
                    logger=self.logger, # Pass agent logger
                    retry_attempts=self.retry_attempts,
                    retry_backoff=self.retry_backoff,
                )
                if extracted_order_id:
                    entities["order_id"] = extracted_order_id
                    self.logger.debug(f"Extracted entity order_id: {extracted_order_id}")

                # Example: Extract product identifier
                if intent == "product_question":
                     extracted_product_id = await nlp.extract_product_id(
                        client=self.client,
                        message=message,
                        model=self.utility_model,
                        logger=self.logger,
                        retry_attempts=self.retry_attempts,
                        retry_backoff=self.retry_backoff,
                    )
                     if extracted_product_id:
                        entities["product_identifier"] = extracted_product_id
                        self.logger.debug(f"Extracted entity product_identifier: {extracted_product_id}")
                # Add more entity extractions as needed...

            except Exception as e:
                self.logger.error(f"Error during entity extraction: {e}", exc_info=True)

            # 5. Get Context Data based on intent & entities
            context_data = await self.context_retriever.get_context(intent, entities)

            # 6. Get Recent Conversation History
            recent_history = self.conversation_manager.get_recent_history(customer_id, n=5)

            # 7. Build the Prompt
            system_prompt = build_response_prompt(
                customer_info=customer_info,
                intent=intent,
                message=message,
                context_data=context_data,
                conversation_history=recent_history,
                # brand_name=... # Could be configurable
            )

            # 8. Generate Response Text
            generated_message = await generate_agent_response(
                client=self.client,
                model=self.response_model,
                system_prompt=system_prompt,
                logger=self.logger,
                retry_attempts=self.retry_attempts,
                retry_backoff=self.retry_backoff,
                # max_tokens, temperature can be passed if needed
            )

            final_response: Dict[str, Any] = {
                "message": "I apologize, I encountered an issue generating a response.",
                "intent": intent,
                "actions": [],
            }

            if generated_message:
                final_response["message"] = generated_message
                # 9. Extract Actions from Response
                actions = await extract_actions(
                    client=self.client,
                    intent=intent,
                    response_text=generated_message,
                    context_data=context_data,
                    model=self.utility_model,
                    logger=self.logger,
                    retry_attempts=self.retry_attempts,
                    retry_backoff=self.retry_backoff,
                )
                final_response["actions"] = actions

                # 10. Add Agent message to history
                self.conversation_manager.add_message(customer_id, "agent", generated_message)
            else:
                 # Handle case where response generation failed
                 final_response["error"] = "LLM response generation failed."

            # 11. Analyze Sentiment (optional, could be done earlier)
            customer_sentiment = await nlp.sentiment_analysis(
                client=self.client,
                message=message,
                model=self.utility_model,
                logger=self.logger,
                retry_attempts=self.retry_attempts,
                retry_backoff=self.retry_backoff,
            )
            final_response["customer_sentiment"] = customer_sentiment

            # 12. Log Interaction
            try:
                await self._log_interaction(customer_id, intent, message, final_response)
                self.logger.debug(f"Logged interaction for customer {customer_id}.")
            except Exception as e:
                self.logger.error(f"Failed to log interaction: {e}")

            # Lock released automatically by context manager

        self.logger.info(
            f"Finished processing inquiry for customer {customer_id}. Intent: {intent}. Actions: {len(final_response.get('actions', []))}"
        )
        return final_response

    # --- Internal Helper Methods --- #

    # Note: _safe_response_create is removed as safe_chat_completion is now used
    # directly by the sub-components (nlp, llm_components)

    # Note: Specific LLM subtask methods are removed as they are handled by nlp/llm_components
    # _classify_intent -> nlp.classify_intent
    # _extract_order_id -> nlp.extract_order_id_llm
    # _extract_product_identifier -> nlp.extract_product_id
    # _generate_response -> llm_components.generate_agent_response
    # _analyze_sentiment -> nlp.sentiment_analysis

    async def _log_interaction(
        self, customer_id: str, intent: str, message: str, response: dict
    ):
        """Log the interaction details (placeholder for actual logging)."""
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
        # In a real system, persist this log_entry to a database, file, or monitoring service.
        pass

# Example Usage (Illustrative - requires setting up dummy dependencies)
# async def main():
#     logging.basicConfig(level=logging.INFO)
#     # Load API key from environment
#     # os.environ["OPENAI_API_KEY"] = "sk-..."

#     agent = RetailCustomerServiceAgent(
#         product_database=DummyProductDB(),
#         order_management_system=DummyOrderSystem(),
#         customer_database=DummyCustomerDB(),
#         policy_guidelines={"returns": {"return_window_days": 30, "return_methods": ["Mail", "In-Store"]}}
#     )

#     customer_id = "cust123"
#     messages = [
#         "Hi, what's the status of my order #ORD987?",
#         "Tell me about the blue widget SKU:BLUE-WDGT-01",
#         "I want to return order ORD987",
#         "This is taking too long!"
#     ]

#     for msg in messages:
#         print(f"\n--- Customer: {msg} ---")
#         response = await agent.process_customer_inquiry(customer_id, msg)
#         print(f"Agent: {response.get('message')}")
#         print(f"(Intent: {response.get('intent')}, Actions: {response.get('actions')}, Sentiment: {response.get('customer_sentiment')})")
#         await asyncio.sleep(1) # Small delay between messages

# if __name__ == "__main__":
#     asyncio.run(main())
