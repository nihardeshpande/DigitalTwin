from openai import OpenAI
from context import TWIN_SYSTEM_PROMPT
from tools import tools, handle_tool_calls
from styles import CSS, JS, EXAMPLES
from dotenv import load_dotenv
import gradio as gr
import os
import time

load_dotenv(override=True)

MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b:free"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openai = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key)

system = [{"role": "system", "content": TWIN_SYSTEM_PROMPT}]


def _as_openai_messages(history):
    """Gradio 6 history is message dicts; older UIs used [user, assistant] pairs."""
    messages = list(system)
    for item in history or []:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            content = item.get("content")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if content:
                messages.append({"role": item["role"], "content": content})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            user, assistant = item[0], item[1]
            if user:
                messages.append({"role": "user", "content": user})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})
    return messages


def _response_error(response):
    dumped = response.model_dump() if hasattr(response, "model_dump") else {}
    return dumped.get("error") or getattr(response, "error", None) or dumped


def _create(messages, retries=3):
    last_error = None
    for attempt in range(1, retries + 1):
        response = openai.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools
        )
        if response.choices:
            return response
        last_error = _response_error(response)
        print(f"OpenRouter returned no choices (attempt {attempt}/{retries}): {last_error}", flush=True)
        time.sleep(1.5 * attempt)
    raise RuntimeError(
        "The model returned an empty completion. This is usually an OpenRouter "
        f"free-model rate limit or provider error: {last_error}"
    )


def chat(message, history):
    messages = _as_openai_messages(history) + [{"role": "user", "content": message}]
    response = _create(messages)
    while True:
        choice = response.choices[0]
        assistant_message = choice.message
        if choice.finish_reason == "tool_calls" or assistant_message.tool_calls:
            results = handle_tool_calls(assistant_message.tool_calls)
            messages.append(assistant_message)
            messages.extend(results)
            response = _create(messages)
            continue
        return assistant_message.content or "I wasn't able to generate a reply. Please try again."


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(css=CSS, js=JS, theme=gr.themes.Base())
