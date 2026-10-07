from __future__ import annotations

# region book:virtual-shopping-responses-setup
import json
import os


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

user_message = "I need an outfit idea for a summer party."
# endregion book:virtual-shopping-responses-setup


# region book:virtual-shopping-responses-run
def run_demo(user_message: str = user_message, *, client=None) -> str:
    if client is None:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise RuntimeError("Set OPENAI_API_KEY to run this provider demo.")
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "openai is not installed. Install the openai package to run this demo."
            ) from exc
        client = OpenAI()

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
        response = final_response

    assistant_reply = response.output_text
    print(assistant_reply)
    return assistant_reply


if __name__ == "__main__":
    try:
        run_demo()
    except (RuntimeError, ImportError) as exc:
        raise SystemExit(str(exc)) from None

# endregion book:virtual-shopping-responses-run
