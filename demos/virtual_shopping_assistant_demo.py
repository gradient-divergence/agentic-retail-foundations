"""
Demo script for a virtual shopping assistant using OpenAI function calling.

This script demonstrates how an AI assistant can recommend outfits by calling
a predefined function when prompted by the user.
"""

import json
import logging

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()

# --- Tool Definition ---


def recommend_outfit(style: str) -> list:
    """
    Recommend fashion items based on the given style or occasion.
    (Simulated function - in real life, this would query a database or model)
    """
    logger.debug(f"Tool 'recommend_outfit' called with style: {style}")
    suggestions = []
    style_lower = style.lower()
    if "summer" in style_lower:
        suggestions = [
            "Red sundress with floral prints",
            "Lightweight beige linen blazer",
            "White sneakers",
        ]
    elif "formal" in style_lower:
        suggestions = [
            "Navy blue suit jacket",
            "Silk tie in matching color",
            "Oxford dress shoes",
        ]
    else:
        suggestions = ["Classic blue jeans", "Comfy cotton t-shirt", "Denim jacket"]
    logger.info(f"Recommendation function generated: {suggestions}")
    return suggestions


# --- OpenAI Responses API Tool Calling Setup ---

tools = [
    {
        "type": "function",
        "name": "recommend_outfit",
        "description": "Recommend fashion items based on style or occasion",
        "parameters": {
            "type": "object",
            "properties": {
                "style": {
                    "type": "string",
                    "description": "The user's style preference or occasion.",
                }
            },
            "required": ["style"],
        },
    }
]

# --- Demo Execution ---


def run_assistant_demo(
    user_message: str = "I need an outfit idea for a summer party.",
) -> str:
    """Runs the virtual shopping assistant demo with the given user message."""
    logger.info("--- Starting Virtual Shopping Assistant Demo ---")
    logger.info(f"User Message: {user_message}")
    assistant_reply: str | None = None  # Initialize to allow for None return on error

    try:
        # Initialize OpenAI client (ensure OPENAI_API_KEY is set in environment)
        client = OpenAI()
        if not client.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        input_items = [{"role": "user", "content": user_message}]

        # First API call: let the model decide if it should call the function
        logger.info("Calling OpenAI API (initial request)...")
        response = client.responses.create(
            model="gpt-5.2",
            input=input_items,
            tools=tools,
        )

        input_items += response.output or []
        tool_calls = [item for item in response.output or [] if item.type == "function_call"]

        if tool_calls:
            logger.info("AI decided to call a function.")
            for tool_call in tool_calls:
                logger.info(
                    "Function to call: %s, Args: %s",
                    tool_call.name,
                    tool_call.arguments,
                )
                if tool_call.name == "recommend_outfit":
                    try:
                        args = json.loads(tool_call.arguments)
                        result = recommend_outfit(**args)
                        logger.info("Function executed successfully.")
                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": tool_call.call_id,
                                "output": json.dumps(result),
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error executing function: {e}")
                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": tool_call.call_id,
                                "output": json.dumps({"error": str(e)}),
                            }
                        )
                else:
                    logger.warning("AI requested unknown function: %s", tool_call.name)
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": json.dumps({"error": "Unknown function"}),
                        }
                    )

            logger.info("Calling OpenAI API (with function result)...")
            final_response = client.responses.create(
                model="gpt-5.2",
                input=input_items,
                tools=tools,
            )
            assistant_reply = final_response.output_text
        else:
            logger.info("AI did not call a function. Returning its direct response.")
            assistant_reply = response.output_text

    except OpenAIError as e:
        logger.error(f"OpenAI API Error: {e}")
        assistant_reply = f"Sorry, there was an error communicating with the AI service: {e}"
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        assistant_reply = f"Sorry, there was a configuration error: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        assistant_reply = f"Sorry, an unexpected error occurred: {e}"

    logger.info(f"Assistant Response: {assistant_reply}")
    logger.info("--- Virtual Shopping Assistant Demo Finished ---")
    return assistant_reply if assistant_reply is not None else "An unknown error occurred."


if __name__ == "__main__":
    # Example of running the demo directly
    run_assistant_demo()
    # Example with a different query
    # run_assistant_demo("What should I wear for a formal event?")
