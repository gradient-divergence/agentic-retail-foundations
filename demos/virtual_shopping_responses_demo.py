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


def _outfit_arguments(arguments: str) -> dict[str, str]:
    try:
        args = json.loads(arguments)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed recommend_outfit arguments") from exc
    if not isinstance(args, dict) or set(args) != {"style"} or not isinstance(args["style"], str):
        raise ValueError("recommend_outfit requires exactly one string style")
    return args


# region book:virtual-shopping-responses-run
def run_demo(user_message: str = user_message, *, client=None, max_tool_rounds: int = 4) -> str:
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

    for round_number in range(max_tool_rounds + 1):
        response = client.responses.create(model="gpt-5.2", input=list(input_items), tools=tools)
        input_items += response.output or []
        tool_calls = [item for item in response.output or [] if item.type == "function_call"]
        if not tool_calls:
            if not isinstance(response.output_text, str) or not response.output_text.strip():
                raise RuntimeError("Terminal response produced no final text")
            print(response.output_text)
            return response.output_text
        if round_number == max_tool_rounds:
            raise RuntimeError("Tool round limit exhausted before a final response")
        for call in tool_calls:
            if call.name != "recommend_outfit":
                raise ValueError(f"Unsupported tool: {call.name}")
            result = recommend_outfit(**_outfit_arguments(call.arguments))
            input_items.append(
                {"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result)}
            )
    raise RuntimeError("Tool round limit must be nonnegative")


if __name__ == "__main__":
    try:
        run_demo()
    except (RuntimeError, ImportError) as exc:
        raise SystemExit(str(exc)) from None

# endregion book:virtual-shopping-responses-run
