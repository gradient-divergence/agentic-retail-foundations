from __future__ import annotations

# region book:virtual-shopping-responses-setup
import json

from openai import OpenAI


def recommend_outfit(style: str) -> list[str]:
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
        suggestions = [
            "Classic blue jeans",
            "Comfy cotton t-shirt",
            "Denim jacket",
        ]
    return suggestions


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

client = OpenAI()
user_message = "I need an outfit idea for a summer party."
# endregion book:virtual-shopping-responses-setup

# region book:virtual-shopping-responses-run
input_items = [{"role": "user", "content": user_message}]

response = client.responses.create(
    model="gpt-5.2",
    input=input_items,
    tools=tools,
)

input_items += response.output or []
tool_calls = [item for item in response.output or [] if item.type == "function_call"]
if tool_calls:
    for tool_call in tool_calls:
        if tool_call.name == "recommend_outfit":
            args = json.loads(tool_call.arguments)
            result = recommend_outfit(**args)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(result),
                }
            )

    final_response = client.responses.create(
        model="gpt-5.2",
        input=input_items,
        tools=tools,
    )
    assistant_reply = final_response.output_text
    print(assistant_reply)

# endregion book:virtual-shopping-responses-run
