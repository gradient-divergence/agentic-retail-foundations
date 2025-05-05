"""
Demo script for a virtual shopping assistant using OpenAI function calling.

This script demonstrates how an AI assistant can recommend outfits by calling
a predefined function when prompted by the user.
"""

import json
import logging
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

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

# --- OpenAI Function Calling Setup ---

tool_schema = [
    {
        "type": "function",
        "function": {
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
        },
    }
]

# --- Demo Execution ---

def run_assistant_demo(user_message: str = "I need an outfit idea for a summer party.") -> str:
    """Runs the virtual shopping assistant demo with the given user message."""
    logger.info("--- Starting Virtual Shopping Assistant Demo ---")
    logger.info(f"User Message: {user_message}")
    assistant_reply: str | None = None # Initialize to allow for None return on error

    try:
        # Initialize OpenAI client (ensure OPENAI_API_KEY is set in environment)
        client = OpenAI()
        if not client.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        # First API call: let the model decide if it should call the function
        logger.info("Calling OpenAI API (initial request)...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_message}],
            tools=tool_schema,
            tool_choice="auto" # Let the model decide
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Check if the model wants to call a function
        if tool_calls:
            logger.info("AI decided to call a function.")
            # For this demo, assume only one tool call
            tool_call = tool_calls[0]
            func_name = tool_call.function.name
            func_args = tool_call.function.arguments

            logger.info(f"Function to call: {func_name}, Args: {func_args}")

            # Execute the function
            if func_name == "recommend_outfit":
                try:
                    args = json.loads(func_args)
                    result = recommend_outfit(**args)
                    result_json = json.dumps(result)
                    logger.info("Function executed successfully.")
                    message_history = [
                        {"role": "user", "content": user_message},
                        response_message, # Include the assistant's first message (with tool call)
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": result_json,
                        }
                    ]

                except Exception as e:
                    logger.error(f"Error executing function: {e}")
                    result_json = json.dumps({"error": str(e)})
                    # If function failed, still send result back so model knows
                    message_history = [
                        {"role": "user", "content": user_message},
                        response_message,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": result_json, # Send error back to model
                        }
                    ]


                # Send the function result back to the model
                logger.info("Calling OpenAI API (with function result)...")
                final_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=message_history, # Correctly typed messages
                )
                assistant_reply = final_response.choices[0].message.content
            else:
                logger.warning(f"AI requested unknown function: {func_name}")
                assistant_reply = "Sorry, I tried to use a tool I don't recognize."

        else:
            logger.info("AI did not call a function. Returning its direct response.")
            assistant_reply = response_message.content

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
